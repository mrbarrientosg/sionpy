import datetime
import os
import torch


class Config:
    def __init__(
        self,
        game_id: str,
        stacked_observations: int = 8,
        lr: float = 0.005,
        epochs: int = 1000,
        steps: int = 1e6,
        batch_size: int = 1024,
        encoding_size: int = 30,
        max_windows: int = 1e6,
        num_unroll_steps: int = 5,
        td_steps: int = 10,
        max_episodes: int = 10,
        max_moves: int = 27000,
        epsilon_gamma: float = 0.99,
        checkpoint_interval: int = 500,
        vf_coef: float = 0.25,
        hidden_nodes: int = 128,
        scheduler_gamma: float = 0.95,
        pb_c_base: float = 19652,
        pb_c_init: float = 1.25,
        simulations: int = 30,
        device: str = "cuda",
        log_dir: str = None,
        **kwargs,
    ):
        self.lr = lr
        self.lr_decay_rate = 0.1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 350e3
        self.device = torch.device(device)
        self.observation_shape = (1, 1, 30)
        self.stacked_observations = stacked_observations
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.steps = steps
        self.epsilon_gamma = epsilon_gamma
        self.vf_coef = vf_coef
        self.pb_c_init = pb_c_init
        self.pb_c_base = pb_c_base
        self.simulations = simulations
        self.action_space = list(range(6))  # TODO: cambiar
        self.max_moves = max_moves
        self.encoding_size = encoding_size
        self.max_windows = max_windows
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.checkpoint_interval = checkpoint_interval
        self.hidden_nodes = hidden_nodes
        self.scheduler_gamma = scheduler_gamma

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

