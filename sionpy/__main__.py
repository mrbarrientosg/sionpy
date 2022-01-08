import gym
from sionpy.a2c import A2C
from sionpy.wrappers import Game
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = A2C.add_model_specific_args(parser)
    args = parser.parse_args()

    env = gym.make("SpaceInvaders-v0", full_action_space=False)
    env = Game(env)

    dict_args = vars(args)
    model = A2C(env, **dict_args)
    model.train()
