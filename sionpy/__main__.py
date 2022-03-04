from ast import arg
import datetime
import importlib
import os
import gym
from gym import Env
import numpy as np
from gym.spaces import Box
import torch
from sionpy.config import Config
from sionpy.network import SionNetwork
from sionpy.sion import Sion
from sionpy.wrappers import Game
from argparse import ArgumentParser
import toml


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--game",
        type=str,
        choices=os.listdir("./games"),
        help="Choose one game.",
        required=True,
    )
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    game_config = toml.load(os.path.join("games", args.game, "config.toml"))

    config = Config(**game_config)
    config.gpus = args.gpus
    config.num_workers = args.num_workers    

    sion = Sion(config, args.game)
    sion.load_model(
        replay_buffer_path="C:\\Users\\Matias\\Documents\\practica\\sionpy\\results\\SpaceInvaders\\exp2\\replay_buffer.pkl",
        checkpoint_path="C:\\Users\\Matias\\Documents\\practica\\sionpy\\results\\SpaceInvaders\\exp2\\model.checkpoint",
    )
    #sion.train()
    print(sion.test(num_tests=10))
