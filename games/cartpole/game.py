import os
import gym
import numpy as np
from sionpy.transformation import DATE


class CartObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation):
        return np.array([[observation]])


def make_game(id: str, logdir: str, seed: int):
    env = gym.make(id)
    env.seed(seed)
    return CartObservation(env)


def make_game_test(id: str, logdir: str, seed: int):
    return gym.wrappers.RecordVideo(make_game(id, logdir, seed), os.path.join(logdir, "video"))
