from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from sionpy.model import A2C
import gym
from sionpy.wrappers import Game
from argparse import ArgumentParser
import datetime

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = A2C.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="avg_reward", mode="max", save_last=False, verbose=True
    )

    env = gym.make("SpaceInvaders-v0", full_action_space=False)
    env = Game(env)

    dict_args = vars(args)
    model = A2C(
        env, env.observation_space.shape, list(range(env.action_space.n)), **dict_args
    )

    store_dir = f"results/"
    store_name = f"{env.spec.id}"
    version = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    logger = TensorBoardLogger(
        store_dir, store_name, version=version, default_hp_metric=False
    )

    trainer: Trainer = Trainer.from_argparse_args(
        args,
        max_steps=1e6,
        reload_dataloaders_every_n_epochs=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)
    # trainer.test(model, ckpt_path="best")

