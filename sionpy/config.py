from argparse import ArgumentParser
import os
from typing import List, Tuple
import gym
import torch
from sionpy.transformation import DATE
from enum import IntEnum


class NetworkTopology(IntEnum):
    FULLY = 1
    RESNET = 2


class Config:
    def __init__(
        self,
        game_id: str = None,
        seed: int = 0,
        stacked_observations: int = 16,
        action_space: int = 0,
        observation_shape: Tuple[int, int, int] = (1, 1, 1),
        selfplay_on_gpu: bool = False,
        lr: float = 0.02,
        lr_decay_rate: float = 0.9,
        lr_decay_steps: int = 1000,
        pb_c_init: float = 1.25,
        pb_c_base: float = 19652,
        simulations: int = 50,
        root_dirichlet_alpha: float = 0.25,
        root_exploration_fraction: float = 0.25,
        steps: int = 1e4,
        batch_size: int = 128,
        encoding_size: int = 8,
        max_windows: int = 500,
        num_unroll_steps: int = 10,
        td_steps: int = 50,
        max_moves: int = 500,
        support_size: int = 10,
        epsilon_gamma: float = 0.997,
        checkpoint_interval: int = 10,
        vf_coef: float = 1.0,
        topology: NetworkTopology = NetworkTopology.FULLY,
        fc_representation_layers: List[int] = [],
        fc_dynamics_layers: List[int] = [16],
        fc_reward_layers: List[int] = [16],
        fc_policy_layers: List[int] = [16],
        fc_value_layers: List[int] = [16],
        ratio: int = None,
        num_workers: int = 1,
        gpus: int = 0,
        log_dir: str = None,
        **kwargs,
    ):
        self.game_id = game_id
        self.seed = seed
        self.stacked_observations = stacked_observations
        self.action_space = list(range(action_space))
        self.observation_shape = observation_shape
        self.selfplay_on_gpu = selfplay_on_gpu

        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base
        self.simulations = simulations
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction

        self.steps = steps
        self.batch_size = batch_size
        self.encoding_size = encoding_size
        self.max_windows = max_windows
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.max_moves = max_moves
        self.support_size = support_size

        self.epsilon_gamma = epsilon_gamma
        self.vf_coef = vf_coef

        self.checkpoint_interval = checkpoint_interval

        self.topology = topology
        self.fc_representation_layers = fc_representation_layers
        self.fc_dynamics_layers = fc_dynamics_layers
        self.fc_reward_layers = fc_reward_layers
        self.fc_policy_layers = fc_policy_layers
        self.fc_value_layers = fc_value_layers

        self.num_workers = num_workers
        self.gpus = gpus
        
        self.train_on_gpu = torch.cuda.is_available() 
        
        self.ratio = ratio

        if log_dir is None:
            log_dir = os.path.join("results", game_id, DATE,)
        self.log_dir = log_dir

    def visit_softmax_temperature_fn(self, trained_steps):
        if trained_steps < 0.5 * self.steps:
            return 1.0
        elif trained_steps < 0.75 * self.steps:
            return 0.5
        else:
            return 0.25
