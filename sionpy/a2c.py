from typing import Dict
from gym import Env
import numpy as np
import torch
from torch import Tensor
from torch.functional import F
from sionpy.network import ActorCriticModel
from sionpy.buffer import BufferDataset, Experience, ReplayBuffer
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import datetime


class A2C:
    def __init__(
        self,
        env: Env,
        stacked_observations: int = 8,
        lr: int = 0.1,
        epochs: int = 1000,
        steps: int = 1e6,
        batch_size: int = 1024,
        rollout_steps: int = 8192,
        max_episode: int = 2500,
        epsilon_gamma: float = 0.99,
        checkpoint_interval: int = 500,
        vf_coef: float = 0.5,
        hidden_nodes: int = 128,
        scheduler_gamma: float = 0.95,
        device: str = "cuda",
        log_dir: str = None,
    ):
        self.env = env
        self.device = torch.device(device)
        self.net = ActorCriticModel(
            env.observation_space.shape,
            env.action_space.n,
            stacked_observations,
            hidden_nodes=hidden_nodes,
        ).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.steps = steps
        self.epsilon_gamma = epsilon_gamma
        self.vf_coef = vf_coef
        self.max_episode = max_episode
        self.buffer = ReplayBuffer(stacked_observations, batch_size)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=scheduler_gamma
        )
        if log_dir is None:
            log_dir = os.path.join(
                "results",
                env.spec.id,
                datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
            )
        self.log_dir = log_dir
        self.checkpoint_interval = checkpoint_interval
        self.log = SummaryWriter(log_dir=log_dir)
        os.mkdir(os.path.join(log_dir, "checkpoint"))
        self.current_epoch = 0
        self.current_step = 0

        self.avg_ep_reward = 0.0
        self.avg_ep_len = 0.0
        self.avg_reward = 0.0

        self.checkpoint = {
            "epoch": 0,
            "step": 0,
            "lr": 0.0,
            "state_dict": None,
            "optimizer": None,
        }

        self.last_best_model = None
        self.last_checkpoint_model = None

    def collect_rollouts(self):
        ep_rewards = []
        ep_values = []
        epoch_rewards = []
        episode_step = 0

        self.buffer.clear()
        state = self.env.reset()
        self.buffer.states.append(state)
        self.buffer.actions.append(0)

        for step in range(self.rollout_steps):
            observations = self.buffer.get_stacked_observations(-1)
            with torch.no_grad():
                torch_observations = (
                    torch.tensor(observations).float().to(self.device).unsqueeze(0)
                )
                output = self.net.forward(torch_observations)
            action = output.actions.item()

            next_state, reward, done, _ = self.env.step(action)

            episode_step += 1

            self.buffer.add(
                Experience(
                    next_state,
                    observations,
                    action,
                    torch.tensor(reward, device=self.device),
                    1 - torch.tensor(done, device=self.device, dtype=torch.float),
                )
            )
            state = next_state

            ep_rewards.append(reward)
            ep_values.append(output.values.item())

            epoch_end = step == (self.rollout_steps - 1)
            terminal = len(ep_rewards) == self.max_episode

            if epoch_end or done or terminal:
                if (terminal or epoch_end) and not done:
                    steps_before_cutoff = episode_step
                else:
                    steps_before_cutoff = 0
                epoch_rewards.append(sum(ep_rewards))
                ep_rewards = []
                ep_values = []
                episode_step = 0
                state = self.env.reset()

        self.avg_reward = sum(epoch_rewards) / self.rollout_steps

        epoch_rewards = epoch_rewards
        if not done:
            epoch_rewards = epoch_rewards[:-1]

        total_epoch_reward = sum(epoch_rewards)
        nb_episodes = len(epoch_rewards)

        if nb_episodes != 0:
            self.avg_ep_reward = total_epoch_reward / nb_episodes
            self.avg_ep_len = (self.rollout_steps - steps_before_cutoff) / nb_episodes
        else:
            self.avg_ep_reward = 0.0
            self.avg_ep_len = 0.0

        observations = self.buffer.get_stacked_observations(-1)
        with torch.no_grad():
            observations = (
                torch.tensor(observations).float().to(self.device).unsqueeze(0)
            )
            output = self.net.forward(observations)

        self.buffer.compute_advantage_returns(output.values.item(), self.epsilon_gamma)

    def loss_function(self, batch) -> Tensor:
        (observations_target, actions_target, returns) = batch

        values, log_probs = self.net.evaluation_actions(
            observations_target, actions_target
        )

        # Se normalizan las recomenpesas y los advantages, para mejor apredizaje de la red
        with torch.no_grad():
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(advantages * log_probs).mean()

        value_loss = F.mse_loss(returns, values)

        loss = policy_loss + self.vf_coef * value_loss

        self.log.add_scalar("Loss/Total Loss", loss.item(), self.current_step)
        self.log.add_scalar("Loss/Policy Loss", policy_loss.item(), self.current_step)
        self.log.add_scalar("Loss/Value Loss", value_loss.item(), self.current_step)

        return loss

    def optimizer_step(self, loss: Tensor):
        """
        Se encarga de actualizar los pesos de la red mediante el optimizador.

        Args:
            loss (Tensor): Valor de la funcion de perdida para hacer backpropagation.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_dataloader(self) -> DataLoader:
        """
        Se obtiene toda la informacion del ambiente y es pasada a la clase DataLoader
        encargada de crear los batch de la info.

        Returns:
            DataLoader: Clase encargada de crear los batch, proviene de pytorch
        """
        self.collect_rollouts()
        dataset = BufferDataset(self.buffer)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: tuple(x_.to(self.device) for x_ in default_collate(x)),
        )

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def save_checkpoint(self, is_best: bool):
        if self.current_step % self.checkpoint_interval == 0 or is_best:
            self.checkpoint["epoch"] = self.current_epoch
            self.checkpoint["step"] = self.current_step
            self.checkpoint["lr"] = self.get_current_lr()
            self.checkpoint["state_dict"] = self.net.state_dict()
            self.checkpoint["optimizer"] = self.optimizer.state_dict()

        if self.current_step % self.checkpoint_interval == 0:
            if self.last_checkpoint_model is not None:
                os.remove(self.last_checkpoint_model)

            self.last_checkpoint_model = os.path.join(
                self.log_dir,
                "checkpoint",
                f"model_epoch_{self.current_epoch}_step_{self.current_step}.pth",
            )
            torch.save(
                self.checkpoint, self.last_checkpoint_model,
            )

        if is_best:
            if self.last_best_model is not None:
                os.remove(self.last_best_model)

            self.last_best_model = os.path.join(
                self.log_dir,
                "checkpoint",
                f"best_model_epoch_{self.current_epoch}_step_{self.current_step}.pth",
            )
            torch.save(
                self.checkpoint, self.last_best_model,
            )

    def check_end_condition(self):
        while self.current_epoch < self.epochs or self.current_step < self.steps:
            yield

    def train(self):
        best_avg_reward = 0.0
        pbar = tqdm(self.check_end_condition(), desc="Epoch")

        for _ in pbar:
            # Ponemos la red en modo evaluacion para optimizacion
            self.net.eval()

            # Cargamos el dataloader
            dataloader = self.train_dataloader()

            # Ponemos la red en modo training para ciertas cosas de optimizacion
            self.net.train()

            # Se iteran los batch y se actualiza el peso de la red
            total_loss = 0.0
            for batch in dataloader:
                loss = self.loss_function(batch)
                total_loss += loss.item()
                self.optimizer_step(loss)
                self.current_step += 1

            self.log.add_scalar(
                "Loss/Learning Rate", self.get_current_lr(), self.current_step - 1
            )
            self.log.add_scalar(
                "Reward/Avg Reward", self.avg_reward, self.current_step - 1
            )
            self.log.add_scalar(
                "Reward/Avg Episode Reward", self.avg_ep_reward, self.current_step - 1
            )
            self.log.add_scalar(
                "Reward/Avg Episode Length", self.avg_ep_len, self.current_step - 1
            )

            is_best = self.avg_reward > best_avg_reward
            best_avg_reward = max(self.avg_reward, best_avg_reward)

            self.save_checkpoint(is_best)

            # Despues de actualizar los pesos de la red, se actualizar el scheduler
            self.scheduler.step()
            self.current_epoch += 1
            pbar.set_postfix(
                {
                    "avg_reward": self.avg_reward,
                    "loss": total_loss,
                    "step": self.current_step,
                }
            )

        self.log.close()

