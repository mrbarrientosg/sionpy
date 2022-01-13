import gym
from sionpy.config import Config
from sionpy.sion import Sion
from sionpy.wrappers import Game
from argparse import ArgumentParser


def make_env(seed):
    env = gym.make("SpaceInvaders-v0", full_action_space=False)
    env.seed(seed)
    return Game(env)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Config.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    config = Config("SpaceInvaders-v0", **dict_args)

    sion = Sion(config, make_env)
    sion.train()
