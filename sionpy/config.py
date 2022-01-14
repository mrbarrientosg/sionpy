from argparse import ArgumentParser
import datetime
import os
import torch


class Config:
    def __init__(
        self,
        game_id: str,
        seed: int = 0,
        stacked_observations: int = 16,
        lr: float = 0.005,
        steps: int = 1e6,
        batch_size: int = 1024,
        encoding_size: int = 30,
        max_windows: int = 1e6,
        num_unroll_steps: int = 5,
        td_steps: int = 10,
        max_moves: int = 27000,
        epsilon_gamma: float = 0.997,
        checkpoint_interval: int = 500,
        vf_coef: float = 0.25,
        hidden_nodes: int = 64,
        support_size: int = 300,
        pb_c_base: float = 19652,
        pb_c_init: float = 1.25,
        simulations: int = 30,
        device: str = "cpu",
        gpus: int = 0,
        num_workers: int = 1,
        log_dir: str = None,
        **kwargs,
    ):
        self.lr = lr
        self.lr_decay_rate = 0.1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 350e3

        self.gpus = gpus
        self.seed = seed
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.observation_shape = (1, 1, 30)
        self.stacked_observations = stacked_observations
        self.batch_size = batch_size
        self.steps = steps
        self.epsilon_gamma = epsilon_gamma
        self.vf_coef = vf_coef
        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base
        self.simulations = simulations
        self.action_space = list(range(6))

        self.max_moves = max_moves
        self.encoding_size = encoding_size
        self.max_windows = max_windows
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.checkpoint_interval = checkpoint_interval
        self.hidden_nodes = hidden_nodes
        self.support_size = support_size

        if log_dir is None:
            log_dir = os.path.join(
                "results",
                game_id,
                datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
            )
        self.log_dir = log_dir

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("Sion")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--stacked_observations", type=int, default=16)
        parser.add_argument("--lr", type=float, default=0.005)
        parser.add_argument("--steps", type=int, default=1e6)
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--encoding_size", type=int, default=30)
        parser.add_argument("--max_windows", type=int, default=1e6)
        parser.add_argument("--num_unroll_steps", type=int, default=5)
        parser.add_argument("--td_steps", type=int, default=10)
        parser.add_argument("--max_moves", type=int, default=27000)
        parser.add_argument("--epsilon_gamma", type=float, default=0.997)
        parser.add_argument("--checkpoint_interval", type=int, default=500)
        parser.add_argument("--vf_coef", type=float, default=0.25)
        parser.add_argument("--hidden_nodes", type=int, default=64)
        parser.add_argument("--support_size", type=int, default=300)
        parser.add_argument("--pb_c_base", type=float, default=19652)
        parser.add_argument("--pb_c_init", type=float, default=1.25)
        parser.add_argument("--simulations", type=int, default=30)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--gpus", type=int, default=0)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--log_dir", type=str, default=None)
        return parent_parser
