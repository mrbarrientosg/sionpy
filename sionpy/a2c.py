from typing import Dict
from gym import Env
import numpy as np
import torch
from torch import Tensor
from torch.functional import F
from sionpy.mcts import MCTS, Node
from sionpy.network import ActorCriticModel
from sionpy.buffer import (
    BufferDataset,
    Experience,
    GameHistory,
    ReplayBuffer,
    ReplaySample,
)
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import datetime
from argparse import ArgumentParser


class A2C:
    def __init__(
        self,
        env: Env,
        stacked_observations: int = 8,
        lr: int = 0.1,
        epochs: int = 1000,
        steps: int = 1e6,
        batch_size: int = 1024,
        encoding_size: int = 30,
        max_windows: int = 1e6,
        num_unroll_steps: int = 5,
        td_steps: int = 10,
        max_episodes: int = 10,
        max_episode_len: int = 2500,
        epsilon_gamma: float = 0.99,
        checkpoint_interval: int = 500,
        vf_coef: float = 0.5,
        hidden_nodes: int = 128,
        scheduler_gamma: float = 0.95,
        simulations: int = 30,
        device: str = "cuda",
        log_dir: str = None,
        **kwargs,
    ):
        self.env = env
        self.device = torch.device(device)
        self.net = ActorCriticModel(
            env.observation_space.shape,
            env.action_space.n,
            stacked_observations,
            encoding_size,
            hidden_nodes=hidden_nodes,
        ).to(self.device)
        self.stacked_observations = stacked_observations
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.steps = steps
        self.epsilon_gamma = epsilon_gamma
        self.vf_coef = vf_coef
        self.simulations = simulations
        self.action_space = list(range(env.action_space.n))
        self.max_episode_len = max_episode_len
        self.buffer = ReplayBuffer(
            max_windows, num_unroll_steps, td_steps, epsilon_gamma, self.action_space
        )
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

    def select_action(self, node: Node, temperature: float = 1.0):
        visit_counts = np.array([child.visit_count for child in node.children.values()])
        actions = [action for action in node.children.keys()]
        if temperature == 0.0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def visit_softmax_temperature_fn(self):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        if self.current_step < 500e3:
            return 1.0
        elif self.current_step < 750e3:
            return 0.5
        else:
            return 0.25

    def collect_rollouts(self):
        ep_rewards = []
        epoch_rewards = []

        for step in range(self.max_episodes):
            done = False
            observation = self.env.reset()
            game_history = GameHistory(self.stacked_observations)
            game_history.add(Experience(observation, 0, 0))

            while not done or len(game_history.actions) < self.max_episode_len:
                observations = game_history.get_stacked_observations(-1)
                root = MCTS(self.epsilon_gamma).run(
                    self.net,
                    self.simulations,
                    observations,
                    self.action_space,
                    self.device,
                )
                action = self.select_action(root, self.visit_softmax_temperature_fn())

                observation, reward, done, _ = self.env.step(action)

                ep_rewards.append(reward)

                game_history.add(Experience(observation, action, reward,))
                game_history.store_search_statistics(root, self.action_space)

            self.buffer.add(game_history)
            epoch_rewards.append(sum(ep_rewards))
            ep_rewards.clear()

        self.avg_reward = sum(epoch_rewards) / self.max_episodes
        # self.avg_ep_len = (self.max_episode_len - steps_before_cutoff) / nb_episodes

        # epoch_rewards = epoch_rewards
        # if not done:
        #     epoch_rewards = epoch_rewards[:-1]

        # total_epoch_reward = sum(epoch_rewards)
        # nb_episodes = len(epoch_rewards)

        # if nb_episodes != 0:
        #     self.avg_ep_reward = total_epoch_reward / nb_episodes
        #
        # else:
        #     self.avg_ep_reward = 0.0
        #     self.avg_ep_len = 0.0

        # observations = self.buffer.get_stacked_observations(-1)
        # with torch.no_grad():
        #     observations = (
        #         torch.tensor(observations).float().to(self.device).unsqueeze(0)
        #     )
        #     output = self.net.forward(observations)

        # self.buffer.compute_advantage_returns(output.values.item(), self.epsilon_gamma)

    def mse(
        self, value, reward, policy_logits, target_value, target_reward, target_policy,
    ):
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=-1)(value)).sum()
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=-1)(reward)).sum()
        policy_loss = (
            -target_policy * torch.nn.LogSoftmax(dim=-1)(policy_logits)
        ).sum()
        return value_loss, reward_loss, policy_loss

    def loss_function(self, batch) -> Tensor:

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
        ) = batch

        observation_batch = (
            torch.tensor(np.array(observation_batch)).float().to(self.device)
        )
        action_batch = torch.tensor(action_batch).long().to(self.device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(self.device)
        target_reward = torch.tensor(target_reward).float().to(self.device)
        target_policy = torch.tensor(target_policy).float().to(self.device)

        ioutput = self.net.initial_inference(observation_batch)

        predictions = [(ioutput.value, ioutput.reward, ioutput.logits)]

        for i in range(1, action_batch.shape[1]):
            routput = self.net.recurrent_inference(
                ioutput.encoded_state, action_batch[:, i]
            )
            predictions.append((routput.value, routput.reward, routput.logits))

        value_loss, reward_loss, policy_loss = (0, 0, 0)
        value, reward, policy_logits = predictions[0]

        current_value_loss, _, current_policy_loss = self.mse(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
        )

        value_loss += current_value_loss
        policy_loss += current_policy_loss

        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]

            (current_value_loss, current_reward_loss, current_policy_loss,) = self.mse(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
            )

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

        loss = value_loss * self.vf_coef + reward_loss + policy_loss
        loss = loss.mean()

        # Se normalizan las recomenpesas y los advantages, para mejor apredizaje de la red
        # with torch.no_grad():
        #     returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        #     advantages = returns - values
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # policy_loss = -(advantages * log_probs).mean()

        # value_loss = F.mse_loss(returns, values)

        # loss = policy_loss + self.vf_coef * value_loss

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
        dataset = BufferDataset(self.buffer.sample())
        return DataLoader(dataset, batch_size=self.batch_size)

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
            # dataloader = self.train_dataloader()
            self.collect_rollouts()

            # Ponemos la red en modo training para ciertas cosas de optimizacion
            self.net.train()

            # Se iteran los batch y se actualiza el peso de la red
            total_loss = 0.0
            for batch in self.buffer.batch(self.batch_size):
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

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("A2C")
        parser.add_argument("--stacked_observations", type=int, default=8)
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--epochs", type=int, default=1000)
        parser.add_argument("--steps", type=int, default=1e6)
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--max_episodes", type=int, default=8192)
        parser.add_argument("--max_episode_len", type=int, default=2500)
        parser.add_argument("--epsilon_gamma", type=float, default=0.99)
        parser.add_argument("--checkpoint_interval", type=int, default=500)
        parser.add_argument("--vf_coef", type=float, default=0.5)
        parser.add_argument("--hidden_nodes", type=int, default=128)
        parser.add_argument("--scheduler_gamma", type=float, default=0.95)
        parser.add_argument("--log_dir", type=str, default=None)
        parser.add_argument("--device", type=str, default="cuda")
        return parent_parser
