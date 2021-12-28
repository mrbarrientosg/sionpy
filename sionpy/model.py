from typing import Iterator, List, OrderedDict, Tuple
from pytorch_lightning import LightningModule
from gym import Env
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from sionpy.agent import ActorCriticAgent

from sionpy.network import MlpNetwork
from sionpy.replay_buffer import ExperienceSourceDataset, ReplayBuffer
import numpy as np
import torch
from torch import Tensor, optim
from torch.optim.optimizer import Optimizer


class A2C(LightningModule):
    def __init__(
        self,
        env: Env,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        entropy_beta: float = 0.01,
        critic_beta: float = 0.5,
        max_episode_len: int = 1024,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=["env", "batch_size", "steps_per_epoch", "max_episode_len"]
        )
        self.max_episode_len = max_episode_len
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.env = env
        self.net = MlpNetwork(
            self.env.observation_space.shape, self.env.action_space.n, 32, 30
        )
        self.agent = ActorCriticAgent()
        self.buffer = ReplayBuffer()

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.eps = np.finfo(np.float32).eps.item()

        self.state = self.env.reset()

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        if not isinstance(x, Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float)
        
        x = x.unsqueeze(0)
        logprobs, values, hidden_state = self.net.initial_inference(x)
        action = self.agent(logprobs)
        return action, logprobs, values

    def train_batch(self) -> Iterator[Tuple[np.ndarray, int, Tensor]]:
        self.buffer.states.append(self.state)
        self.buffer.actions.append(0)
        for step in range(self.steps_per_epoch):
            observation = self.buffer.get_stacked_observations(-1, 32)
            with torch.no_grad():
                action, _, value = self.forward(observation)
                action = action[0]

            next_state, reward, done, _ = self.env.step(action)

            self.episode_step += 1

            self.buffer.add(self.state, action, reward, done)
            self.state = next_state

            self.ep_rewards.append(reward)
            self.ep_values.append(value.detach())

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len
            over = done or terminal

            if epoch_end or over:
                if terminal or epoch_end:
                    observation = self.buffer.get_stacked_observations(-1, 32)
                    with torch.no_grad():
                        _, _, value = self.forward(observation)
                        last_value = value.detach()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                self.state = self.env.reset()
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards.clear()
                self.ep_values.clear()
                self.episode_step = 0

            if epoch_end:
                returns = self.buffer.compute_returns(last_value, self.hparams.gamma)

                train_data = zip(self.buffer.states, self.buffer.actions, returns)

                for state, action, _return in train_data:
                    yield self.buffer.get_stacked_observations(-1, 32), action, _return

                self.buffer.reset()
                self.buffer.states.append(self.state)
                self.buffer.actions.append(0)
                
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                if nb_episodes != 0:
                    self.avg_ep_reward = total_epoch_reward / nb_episodes
                    self.avg_ep_len = (
                        self.steps_per_epoch - steps_before_cutoff
                    ) / nb_episodes
                else:
                    self.avg_ep_reward = 0.0
                    self.avg_ep_len = 0.0

                self.epoch_rewards.clear()


    def loss(self, states: Tensor, actions: Tensor, returns: Tensor,) -> Tensor:
        """Calculates the loss for A2C which is a weighted sum of actor loss (MSE), critic loss (PG), and entropy
        (for exploration)
        Args:
            states: tensor of shape (batch_size, state dimension)
            actions: tensor of shape (batch_size, )
            returns: tensor of shape (batch_size, )
        """

        logprobs, values, _ = self.net.initial_inference(states)

        # calculates (normalized) advantage
        with torch.no_grad():
            # critic is trained with normalized returns, so we need to scale the values here
            advs = returns - values * returns.std() + returns.mean()
            # normalize advantages to train actor
            advs = (advs - advs.mean()) / (advs.std() + self.eps)
            # normalize returns to train critic
            targets = (returns - returns.mean()) / (returns.std() + self.eps)

        # entropy loss
        entropy = -logprobs.exp() * logprobs
        entropy = self.hparams.entropy_beta * entropy.sum(1).mean()

        # actor loss
        logprobs = logprobs[range(self.batch_size), actions]
        actor_loss = -(logprobs * advs).mean()

        # critic loss
        critic_loss = self.hparams.critic_beta * torch.square(targets - values).mean()

        # total loss (weighted sum)
        total_loss = actor_loss + critic_loss - entropy

        return total_loss

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> OrderedDict:
        """Perform one actor-critic update using a batch of data.
        Args:
            batch: a batch of (states, actions, returns)
        """
        states, actions, returns = batch
        loss = self.loss(states, actions, returns)

        self.log(
            "avg_ep_len", self.avg_ep_len, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "avg_ep_reward",
            self.avg_ep_reward,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "avg_reward", self.avg_reward, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log("loss", loss, prog_bar=False, on_step=False, on_epoch=True)

        return OrderedDict({"loss": loss, "avg_reward": self.avg_reward})

    def configure_optimizers(self) -> List[Optimizer]:
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def train_dataloader(self) -> DataLoader:
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def get_device(self, batch) -> str:
        return batch[0][0][0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("A2C")
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--steps_per_epoch", type=int, default=2048)
        parser.add_argument("--entropy_beta", type=float, default=0.01)
        parser.add_argument("--critic_beta", type=float, default=0.5)
        parser.add_argument("--max_episode_len", type=int, default=256)
        return parent_parser
