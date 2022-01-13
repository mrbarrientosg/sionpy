import time
import gym
from torch.utils.tensorboard.writer import SummaryWriter
from sionpy.a2c import A2C
from sionpy.buffer import ReplayBuffer
from sionpy.config import Config
from sionpy.network import ActorCriticModel
from sionpy.self_play import SelfPlay
from sionpy.shared_storage import SharedStorage
from sionpy.trainer import Trainer
from sionpy.wrappers import Game
from argparse import ArgumentParser
import ray


def make_env():
    env = gym.make("SpaceInvaders-v0", full_action_space=False)
    return Game(env)


def logging_loop(
    num_gpus, checkpoint, replay_buffer_worker, shared_storage_worker, config
):
    """
        Keep track of the training performance.
        """
    # Launch the test worker to get performance metrics
    test_worker = SelfPlay.options(num_cpus=0, num_gpus=num_gpus,).remote(
        make_env, checkpoint, replay_buffer_worker, shared_storage_worker, config
    )
    test_worker.start_playing.remote(True)

    # Write everything in TensorBoard
    writer = SummaryWriter(config.log_dir)

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
    info = ray.get(shared_storage_worker.get_info.remote(keys))
    try:
        while info["training_step"] < config.steps:
            info = ray.get(shared_storage_worker.get_info.remote(keys))
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
                f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{config.steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                end="\r",
            )
            counter += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    terminate_workers(shared_storage_worker, replay_buffer_worker)


def terminate_workers(shared_storage_worker, replay_buffer_worker):
    """
    Softly terminate the running tasks and garbage collect the workers.
    """
    if shared_storage_worker:
        shared_storage_worker.set_info.remote("terminate", True)
        # checkpoint = ray.get(
        #     shared_storage_worker.get_checkpoint.remote()
        # )
    # if replay_buffer_worker:
    #     replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

    print("\nShutting down workers...")


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser = A2C.add_model_specific_args(parser)
    # args = parser.parse_args()
    config = Config("SpaceInvaders-v0")

    # dict_args = vars(args)
    # model = A2C(env, max_episodes=5, max_episode_len=300)
    # model.train()

    ray.init(num_gpus=1, ignore_reinit_error=True)

    checkpoint = {
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

    replay_buffer = {}

    checkpoint["weights"] = ActorCriticModel(config).state_dict()

    self_play_workers = None
    training_worker = None
    replay_buffer_worker = None
    shared_storage_worker = None

    shared_storage_worker = SharedStorage.remote(checkpoint, config,)

    shared_storage_worker.set_info.remote("terminate", False)

    replay_buffer_worker = ReplayBuffer.remote(checkpoint, config)

    training_worker = Trainer.options(num_cpus=0, num_gpus=0,).remote(
        checkpoint, config, replay_buffer_worker, shared_storage_worker
    )

    self_play_workers = [
        SelfPlay.options(num_cpus=0, num_gpus=0,).remote(
            make_env, checkpoint, replay_buffer_worker, shared_storage_worker, config
        )
        for seed in range(6)  # TODO: cambiar
    ]

    [self_play_worker.start_playing.remote() for self_play_worker in self_play_workers]

    training_worker.train.remote()

    logging_loop(0, checkpoint, replay_buffer_worker, shared_storage_worker, config)
