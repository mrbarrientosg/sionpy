import time
import numpy as np
import ray
import torch
from sionpy.buffer import ReplayBuffer
from sionpy.config import Config
import copy
from torch import Tensor
from sionpy.network import ActorCriticModel
from sionpy.shared_storage import SharedStorage
from torch.functional import F

from sionpy.transformation import transform_to_logits


@ray.remote
class Trainer:
    def __init__(
        self,
        initial_checkpoint,
        config: Config,
        replay_buffer: ReplayBuffer,
        shared_storage: SharedStorage,
    ):
        self.config = config

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.replay_buffer = replay_buffer
        self.shared_storage = shared_storage

        self.model = ActorCriticModel(config)
        self.model.load_state_dict(initial_checkpoint["weights"])
        self.model.to(config.device)
        self.model.eval()

        self.training_step = initial_checkpoint["training_step"]

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=1e-4
        )

        if initial_checkpoint["optimizer_state"] is not None:
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[config.lr_decay_steps], gamma=config.lr_decay_rate
        )

    def train(self):
        while ray.get(self.shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = self.replay_buffer.sample.remote(self.config.batch_size)

        while self.training_step < self.config.steps and not ray.get(
            self.shared_storage.get_info.remote("terminate")
        ):
            batch = ray.get(next_batch)
            next_batch = self.replay_buffer.sample.remote(self.config.batch_size)

            (total_loss, value_loss, reward_loss, policy_loss,) = self.update_weights(
                batch
            )

            self.scheduler.step()

            if self.training_step % self.config.checkpoint_interval == 0:
                self.shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.state_dict()),
                        "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
                    }
                )

                self.shared_storage.save_checkpoint.remote()

            self.shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                }
            )

    def initial_mse(
        self, value, policy_logits, target_value, target_policy,
    ):
        value_loss = F.cross_entropy(value, target_value).sum()
        policy_loss = F.cross_entropy(policy_logits, target_policy).sum()
        # policy_loss = (-target_policy * F.softmax(policy_logits, dim=1)).mean()
        return value_loss, policy_loss

    def recurrent_mse(
        self, value, reward, policy_logits, target_value, target_reward, target_policy,
    ):
        value_loss = F.cross_entropy(value, target_value).sum()
        reward_loss = F.cross_entropy(reward, target_reward).sum()
        policy_loss = F.cross_entropy(policy_logits, target_policy).sum()
        return value_loss, reward_loss, policy_loss

    def update_weights(self, batch) -> Tensor:
        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
        ) = batch

        observation_batch = (
            torch.tensor(np.array(observation_batch)).float().to(self.config.device)
        )
        action_batch = (
            torch.tensor(action_batch).long().to(self.config.device).unsqueeze(-1)
        )
        target_value = torch.tensor(target_value).float().to(self.config.device)
        target_reward = torch.tensor(target_reward).float().to(self.config.device)
        target_policy = torch.tensor(target_policy).float().to(self.config.device)

        target_value = transform_to_logits(target_value, self.config.support_size)
        target_reward = transform_to_logits(target_reward, self.config.support_size)

        value, reward, policy_logits, encoded_state = self.model.initial_inference(
            observation_batch
        )

        value_loss, reward_loss, policy_loss = (0, 0, 0)

        current_value_loss, current_policy_loss = self.initial_mse(
            value.squeeze(-1), policy_logits, target_value[:, 0], target_policy[:, 0],
        )

        value_loss += current_value_loss
        policy_loss += current_policy_loss

        for i in range(1, self.config.num_unroll_steps + 1):
            (
                value,
                reward,
                policy_logits,
                encoded_state,
            ) = self.model.recurrent_inference(encoded_state, action_batch[:, i - 1])

            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
            ) = self.recurrent_mse(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
            )

            encoded_state.register_hook(lambda grad: grad * 0.5)

            value_loss += current_value_loss / self.config.num_unroll_steps
            reward_loss += current_reward_loss / self.config.num_unroll_steps
            policy_loss += current_policy_loss / self.config.num_unroll_steps

        loss = value_loss * self.config.vf_coef + reward_loss + policy_loss
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
        )

    # def update_lr(self):
    #     """
    #     Update learning rate
    #     """
    #     lr = self.config.lr * self.config.lr_decay_rate ** (
    #         self.training_step / self.config.lr_decay_steps
    #     )
    #     for param_group in self.optimizer.param_groups:
    #         param_group["lr"] = lr
