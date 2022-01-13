from typing import Callable
import numpy as np
import ray
from gym import Env
import torch
from sionpy.buffer import Experience, GameHistory, ReplayBuffer
from sionpy.config import Config
from sionpy.mcts import MCTS, Node
from sionpy.network import ActorCriticModel
from sionpy.shared_storage import SharedStorage


@ray.remote
class SelfPlay:
    def __init__(
        self,
        make_env: Callable[[], Env],
        initial_checkpoint,
        replay_buffer: ReplayBuffer,
        shared_storage: SharedStorage,
        config: Config,
    ):
        self.replay_buffer = replay_buffer
        self.shared_storage = shared_storage
        self.config = config
        self.env = make_env()
        self.model = ActorCriticModel(config)
        self.model.load_state_dict(initial_checkpoint["weights"])
        self.model.to(config.device)
        self.model.eval()

    def start_playing(self, test_mode: bool = False):
        while ray.get(
            self.shared_storage.get_info.remote("training_step")
        ) < self.config.steps or not ray.get(
            self.shared_storage.get_info.remote("terminate")
        ):
            self.model.load_state_dict(ray.get(self.shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        ray.get(self.shared_storage.get_info.remote("training_step"))
                    )
                )

                self.replay_buffer.add.remote(game_history, self.shared_storage)
            else:
                game_history = self.play_game(0.0)

                self.shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.actions) - 1,
                        "total_reward": sum(game_history.rewards),
                        "mean_value": np.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )

        self.env.close()

    def play_game(self, temperature: float) -> GameHistory:
        game_history = GameHistory(self.config.stacked_observations)
        observation = self.env.reset()
        game_history.add(Experience(observation, 0, 0))

        done = False

        with torch.no_grad():
            while not done and len(game_history.actions) <= self.config.max_moves:
                stacked_observations = game_history.get_stacked_observations(-1)
                root, info = MCTS(self.config).run(
                    self.model,
                    self.config.simulations,
                    stacked_observations,
                    self.config.action_space,
                    self.config.device,
                )

                action = self.select_action(root, temperature)

                observation, reward, done, _ = self.env.step(action)

                game_history.store_search_statistics(root, self.config.action_space)
                game_history.add(Experience(observation, action, reward))

        return game_history

    def select_action(self, node: Node, temperature: float = 1.0):
        visit_counts = np.array([child.visit_count for child in node.children.values()])
        actions = [action for action in node.children.keys()]
        if temperature == 0.0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

