from typing import Iterator, List, OrderedDict, Tuple
from pytorch_lightning import LightningModule
from gym import Env
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from sionpy.experience import ExperienceSourceDataset
from sionpy.mcts import MCTS, Node

from sionpy.network import SionNetwork
from sionpy.replay_buffer import GameHistory, ReplayBuffer, SimpleBatch
import numpy as np
import torch
from torch import Tensor, optim
from torch.optim.optimizer import Optimizer
from torch.functional import F


class A2C(LightningModule):
    def __init__(
        self,
        env: Env,
        observation_shape: Tuple,
        action_space: List,
        lr: float = 1e-3,
        batch_size: int = 16,
        encoding_size: int = 30,
        max_moves: int = 2500,
        max_episodes: int = 10,
        stacked_observations: int = 32,
        window_size: int = 1e6,
        num_unroll_steps: int = 5,
        td_steps: int = 10,
        discount: float = 0.997,
        simulations: int = 30,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["env"])
        self.env = env
        self.net = SionNetwork(
            observation_shape, len(action_space), stacked_observations, encoding_size
        )
        self.buffer = ReplayBuffer(window_size)

    def loss(self, batch: SimpleBatch) -> Tensor:
        policy_logits, value, hidden_state, reward = self.net.initial_inference(
            batch.observation_batch
        )

        predictions = [(value, reward, policy_logits)]

        for i in range(1, batch.action_batch.shape[1]):
            policy_logits, value, hidden_state, reward = self.net.recurrent_inference(
                hidden_state, batch.action_batch[:, i]
            )
            predictions.append((value, reward, policy_logits))

        value_loss, reward_loss, policy_loss = (0, 0, 0)
        value, reward, policy_logits = predictions[0]

        current_value_loss, _, current_policy_loss = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            batch.target_value[:, 0],
            batch.target_reward[:, 0],
            batch.target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss

        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                batch.target_value[:, i],
                batch.target_reward[:, i],
                batch.target_policy[:, i],
            )

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

        loss = value_loss * 0.25 + reward_loss + policy_loss
        loss = loss.mean()

        return (loss, value_loss, reward_loss, policy_loss)

    def loss_function(
        self, value, reward, policy_logits, target_value, target_reward, target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = F.cross_entropy(
            value.unsqueeze(0), target_value.unsqueeze(0)
        ).sum()
        reward_loss = F.cross_entropy(
            reward.unsqueeze(0), target_reward.unsqueeze(0)
        ).sum()
        policy_loss = F.cross_entropy(
            policy_logits.unsqueeze(0), target_policy.unsqueeze(0)
        ).sum()
        return value_loss, reward_loss, policy_loss

    def training_step(self, batch: SimpleBatch, batch_idx: int):
        loss, value_loss, reward_loss, policy_loss = self.loss(batch)

        avg_reward, avg_mean_value, avg_length = self.buffer.get_statistic()

        self.log(
            "1.Total_reward/1.Avg_reward", avg_reward, on_step=False, on_epoch=True,
        )
        self.log(
            "1.Total_reward/2.Avg_mean_value",
            avg_mean_value,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "1.Total_reward/3.Avg_episode_length",
            avg_length,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "2.Loss/1.Total_loss", loss, on_step=False, on_epoch=True,
        )
        self.log(
            "2.Loss/2.Value_loss", value_loss, on_step=False, on_epoch=True,
        )
        self.log(
            "2.Loss/3.Reward_loss", reward_loss, on_step=False, on_epoch=True,
        )
        self.log(
            "2.Loss/4.Policy_loss", policy_loss, on_step=False, on_epoch=True,
        )

        self.log("avg_reward", avg_reward, logger=False, prog_bar=True)

        self.log("loss", loss, logger=False)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.loss(batch)
        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def test_dataloader(self):
        dataset = ExperienceSourceDataset(
            self.env, self.net, self.buffer, self.hparams, self.device, test=True
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_wrapper,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers >= 1 else False 
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataset = ExperienceSourceDataset(
            self.env, self.net, self.buffer, self.hparams, self.device
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_wrapper,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers >= 1 else False 
        )
        return dataloader

    def collate_wrapper(self, batch):
        return SimpleBatch(batch, self.device)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("A2C")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_moves", type=int, default=2500)
        parser.add_argument("--encoding_size", type=int, default=30)
        parser.add_argument("--max_episodes", type=int, default=10)
        parser.add_argument("--window_size", type=int, default=1e6)
        parser.add_argument("--stacked_observations", type=int, default=32)
        parser.add_argument("--num_unroll_steps", type=int, default=5)
        parser.add_argument("--td_steps", type=int, default=10)
        parser.add_argument("--discount", type=float, default=0.997)
        parser.add_argument("--simulations", type=int, default=30)
        parser.add_argument("--num_workers", type=int, default=0)
        return parent_parser
