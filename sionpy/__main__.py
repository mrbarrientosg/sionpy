import datetime
import gym
from gym import Env
import numpy as np
from gym.spaces import Box
from sionpy.config import Config
from sionpy.sion import Sion
from sionpy.transformation import DATE
from sionpy.wrappers import Game
from argparse import ArgumentParser
import cv2


ID = "ALE/SpaceInvaders-v5"


class CartObservation(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation):
        return np.array([[observation]])


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def observation(self, observation):
        observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA)
        observation = np.asarray(observation, dtype="float32") / 255.0
        observation = np.moveaxis(observation, -1, 0)
        return observation


def make_env(seed):
    env = gym.make(ID, full_action_space=False)
    env.seed(seed)
    return ResizeObservation(env)


def make_env_test(seed):
    env = gym.make(ID, full_action_space=False)
    env.seed(seed)
    return gym.wrappers.RecordVideo(ResizeObservation(env), f"video/{ID}/{DATE}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Config.add_model_specific_args(parser)
    args = parser.parse_args()
    dict_args = vars(args)

    config = Config(ID, **dict_args)

    sion = Sion(config, make_env)
    sion.train()
    print(sion.test(make_env_test, 10))
