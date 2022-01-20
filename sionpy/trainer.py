import time
import numpy as np
import ray
import torch
from sionpy.buffer import ReplayBuffer
from sionpy.config import Config
import copy
from torch import Tensor
from sionpy.network import SionNetwork
from sionpy.shared_storage import SharedStorage
from torch.functional import F

from sionpy.transformation import dict_to_cpu, transform_to_logits


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

        self.model = SionNetwork(config)
        self.model.load_state_dict(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.config.lr, weight_decay=1e-4, momentum=0.9
        )

        if initial_checkpoint["optimizer_state"] is not None:
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
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

            self.update_lr()

            (total_loss, value_loss, reward_loss, policy_loss,) = self.update_weights(
                batch
            )

            if self.training_step % self.config.checkpoint_interval == 0:
                self.shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(dict_to_cpu(self.model.state_dict())),
                        "optimizer_state": copy.deepcopy(dict_to_cpu(self.optimizer.state_dict())),
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
            
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(self.shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.steps
                    and not ray.get(self.shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def initial_mse(
        self, value, policy_logits, target_value, target_policy,
    ):
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return value_loss, policy_loss

    def recurrent_mse(
        self, value, reward, policy_logits, target_value, target_reward, target_policy,
    ):
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return value_loss, reward_loss, policy_loss

    def update_weights(self, batch) -> Tensor:
        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            gradient_scale_batch,
        ) = batch

        device = next(self.model.parameters()).device
        observation_batch = torch.tensor(np.array(observation_batch)).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)

        target_value = transform_to_logits(target_value, self.config.support_size)
        target_reward = transform_to_logits(target_reward, self.config.support_size)

        value, reward, policy_logits, encoded_state = self.model.initial_inference(
            observation_batch
        )

        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            (
                value,
                reward,
                policy_logits,
                encoded_state,
            ) = self.model.recurrent_inference(encoded_state, action_batch[:, i])
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            encoded_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))

        value_loss, reward_loss, policy_loss = (0, 0, 0)
        value, reward, policy_logits = predictions[0]

        current_value_loss, current_policy_loss = self.initial_mse(
            value.squeeze(-1), policy_logits, target_value[:, 0], target_policy[:, 0],
        )

        value_loss += current_value_loss
        policy_loss += current_policy_loss

        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
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

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_reward_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

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

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
