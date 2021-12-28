from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler
from sionpy.model import A2C
import gym
from sionpy.wrappers import Game
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = A2C.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="avg_reward", mode="max", save_last=True, verbose=True
    )

    env = gym.make("SpaceInvaders-v0", frameskip=1)
    env = Game(env)

    dict_args = vars(args)
    model = A2C(env, **dict_args)

    store_dir = f"experiments/{model.__class__.__name__}/"
    store_name = f"SpaceInvaders"
    logger = TensorBoardLogger(store_dir, store_name, default_hp_metric=False)

    trainer: Trainer = Trainer.from_argparse_args(
        args, logger=logger, callbacks=[checkpoint_callback]
    )

    trainer.fit(model)

    env = gym.make("SpaceInvaders-v0", render_mode="human")
    env = Game(env)
    obs = env.reset()

    for i in range(5000):
        logprob, _ = model.forward(obs)
        action = model.agent(logprob)
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
    env.close()
