import copy
import importlib
import math
import os
import pickle
import time
from typing import Callable, Dict
from gym import Env
import numpy as np
import torch
from sionpy.config import Config
from sionpy.network import SionNetwork
from sionpy.buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import ray
from sionpy.self_play import SelfPlay
from sionpy.shared_storage import SharedStorage
from sionpy.trainer import Trainer
from sionpy.transformation import dict_to_cpu


class Sion:
    def __init__(self, config: Config, game: str):
        self.config = config
        self.game_module = importlib.import_module("games." + game + ".game")
        self.make_env = self.game_module.make_game

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
            "counter": 0,
        }

        self.replay_buffer = []

        model = SionNetwork(config)
        self.summary = str(model).replace("\n", " \n\n")
        self.checkpoint["weights"] = copy.deepcopy(dict_to_cpu(model.state_dict()))

        self.self_play_workers = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.checkpoint = torch.load(checkpoint_path)

        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                with open(replay_buffer_path, "rb") as f:
                    replay_buffer_infos = pickle.load(f)
                self.replay_buffer = replay_buffer_infos["buffer"]
                self.checkpoint["num_played_steps"] = replay_buffer_infos[
                    "num_played_steps"
                ]
                self.checkpoint["num_played_games"] = replay_buffer_infos[
                    "num_played_games"
                ]
            else:
                self.checkpoint["training_step"] = 0
                self.checkpoint["num_played_steps"] = 0
                self.checkpoint["num_played_games"] = 0

    def train(self):
        if 0 < self.config.gpus:
            num_gpus_per_worker = self.config.gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + 1 * self.config.selfplay_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        self.shared_storage_worker = SharedStorage.remote(self.checkpoint, self.config,)
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = ReplayBuffer.remote(
            self.checkpoint, self.replay_buffer, self.config
        )

        self.training_worker = Trainer.options(
            num_cpus=0, num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0
        ).remote(
            self.checkpoint,
            self.config,
            self.replay_buffer_worker,
            self.shared_storage_worker,
        )

        self.self_play_workers = [
            SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(
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

        self.logging_loop(num_gpus_per_worker if self.config.selfplay_on_gpu else 0)

    def logging_loop(self, gpus):
        test_worker = SelfPlay.options(num_cpus=0, num_gpus=gpus).remote(
            self.make_env,
            self.checkpoint,
            self.replay_buffer_worker,
            self.shared_storage_worker,
            self.config,
            self.config.seed + self.config.num_workers,
        )
        test_worker.start_playing.remote(True)

        writer = SummaryWriter(self.config.log_dir)

        print(
            "\nTraining...\nCorre el comando tensorboard --logdir ./results para ver el performance del training en tiempo real.\n"
        )

        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        writer.add_text(
            "Model summary", self.summary,
        )

        counter = self.checkpoint["counter"]
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

        self.shared_storage_worker.set_info.remote("counter", counter)
        self.shared_storage_worker.save_checkpoint.remote()

        self.terminate_workers()

        pickle.dump(
            {
                "buffer": self.replay_buffer,
                "num_played_games": self.checkpoint["num_played_games"],
                "num_played_steps": self.checkpoint["num_played_steps"],
            },
            open(os.path.join(self.config.log_dir, "replay_buffer.pkl"), "wb"),
        )

    def terminate_workers(self):
        self.shared_storage_worker.set_info.remote("terminate", True)
        self.checkpoint = ray.get(self.shared_storage_worker.get_checkpoint.remote())

        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(self, num_tests=1):
        self_play_worker = SelfPlay.remote(
            self.game_module.make_game_test,
            self.checkpoint,
            None,
            None,
            self.config,
            np.random.randint(10000),
        )
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(ray.get(self_play_worker.play_game.remote(0)))
        self_play_worker.close.remote()

        result = np.mean([sum(history.rewards) for history in results])

        return result
