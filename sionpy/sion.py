import time
from typing import Callable, Dict
from gym import Env
import numpy as np
import torch
from torch import Tensor
from torch.functional import F
from sionpy.config import Config
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
import ray
from sionpy.self_play import SelfPlay

from sionpy.shared_storage import SharedStorage
from sionpy.trainer import Trainer


class Sion:
    def __init__(self, config: Config, make_env: Callable[[int], Env]):
        self.config = config
        self.make_env = make_env

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        ray.init(num_gpus=self.config.gpus, ignore_reinit_error=True)

        self.checkpoint = {
            "training_step": 0,
            "weights": None,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "episode_length": 0,
            "total_reward": 0,
            "mean_value": 0,
            "num_played_steps": 0,
            "num_played_games": 0,
            "optimizer_state": None,
            "terminate": False,
            "lr": 0,
        }

        self.replay_buffer = {}

        self.checkpoint["weights"] = ActorCriticModel(config).state_dict()

        self.self_play_workers = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self):
        self.shared_storage_worker = SharedStorage.remote(self.checkpoint, self.config,)
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = ReplayBuffer.remote(self.checkpoint, self.config)

        self.training_worker = Trainer.options(num_cpus=0, num_gpus=0,).remote(
            self.checkpoint,
            self.config,
            self.replay_buffer_worker,
            self.shared_storage_worker,
        )

        self.self_play_workers = [
            SelfPlay.options(num_cpus=0, num_gpus=0,).remote(
                self.make_env,
                self.checkpoint,
                self.replay_buffer_worker,
                self.shared_storage_worker,
                self.config,
                self.config.seed + seed,
            )
            for seed in range(self.config.num_workers)
        ]

        [
            self_play_worker.start_playing.remote()
            for self_play_worker in self.self_play_workers
        ]

        self.training_worker.train.remote()

        self.logging_loop(0)

    def logging_loop(self, num_gpus):
        """
            Keep track of the training performance.
            """
        # Launch the test worker to get performance metrics
        test_worker = SelfPlay.options(num_cpus=0, num_gpus=num_gpus,).remote(
            self.make_env,
            self.checkpoint,
            self.replay_buffer_worker,
            self.shared_storage_worker,
            self.config,
            self.config.seed + self.config.num_workers,
        )
        test_worker.start_playing.remote(True)

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.log_dir)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        # hp_table = [f"| {key} | {value} |" for key, value in config.__dict__.items()]
        # writer.add_text(
        #     "Hyperparameters",
        #     "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        # )
        # Save model representation
        # writer.add_text(
        #     "Model summary", self.summary,
        # )
        # Loop for updating the training performance
        counter = 0
        keys = [
            "total_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalar(
                    "1.Total_reward/1.Total_reward", info["total_reward"], counter,
                )
                writer.add_scalar(
                    "1.Total_reward/2.Mean_value", info["mean_value"], counter,
                )
                writer.add_scalar(
                    "1.Total_reward/3.Episode_length", info["episode_length"], counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self_played_games", info["num_played_games"], counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training_steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/5.Training_steps_per_self_played_step_ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    counter,
                )
                writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
                print(
                    f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                    end="\r",
                )
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            # self.checkpoint = ray.get(
            #     self.shared_storage_worker.get_checkpoint.remote()
            # )
        print("\nShutting down workers...")
