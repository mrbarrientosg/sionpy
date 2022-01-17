from collections import namedtuple
from typing import List
import numpy as np
import ray
from sionpy.config import Config
from sionpy.mcts import Node
from sionpy.shared_storage import SharedStorage

Experience = namedtuple("Experience", field_names=["observation", "action", "reward"],)


# class ReplaySample(NamedTuple):
#     observation_batch
#     action_batch
#     value_batch
#     reward_batch
#     policy_batch


ReplaySample = namedtuple(
    "ReplaySample",
    field_names=[
        "observation_batch",
        "action_batch",
        "value_batch",
        "reward_batch",
        "policy_batch",
        "gradient_scale_batch",
    ],
)


class GameHistory:
    def __init__(self, num_stacked_observations: int):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.num_stacked_observations = num_stacked_observations

    def add(self, exp: Experience):
        self.observations.append(exp.observation)
        self.actions.append(exp.action)
        self.rewards.append(exp.reward)

    def store_search_statistics(self, root: Node, action_space: List[int]):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )

        self.root_values.append(root.value())

    def get_stacked_observations(self, index: int) -> np.ndarray:
        index = index % len(self.observations)

        stacked_observations = self.observations[index].copy()
        for past_observation_index in reversed(
            range(index - self.num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = np.concatenate(
                    (
                        self.observations[past_observation_index],
                        [
                            np.ones_like(stacked_observations[0])
                            * self.actions[past_observation_index + 1]
                        ],
                    )
                )
            else:
                previous_observation = np.concatenate(
                    (
                        np.zeros_like(self.observations[index]),
                        [np.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = np.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


@ray.remote
class ReplayBuffer:
    def __init__(
        self, initial_checkpoint, config: Config,
    ):
        self.config = config
        self.game_histories: List[GameHistory] = []
        self.max_windows = config.max_windows
        self.num_unroll_steps = config.num_unroll_steps
        self.td_steps = config.td_steps
        self.gamma = config.epsilon_gamma
        self.action_space = config.action_space
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]

        np.random.seed(config.seed)

    def add(self, game_history: GameHistory, shared_storage: SharedStorage = None):
        if len(self.game_histories) > self.max_windows:
            self.game_histories.pop(0)

        self.game_histories.append(game_history)
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def sample(self, batch_size: int) -> ReplaySample:
        (
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [])

        games = self.sample_games(batch_size)
        game_pos = [(g, self.sample_position(g)) for g in games]

        for (game, game_idx) in game_pos:
            values, rewards, policies, actions = self.make_target(game, game_idx)

            observation_batch.append(game.get_stacked_observations(game_idx))
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [min(self.config.num_unroll_steps, len(game.actions) - game_idx,)]
                * len(actions)
            )

        return ReplaySample(
            observation_batch,
            action_batch,
            value_batch,
            reward_batch,
            policy_batch,
            gradient_scale_batch,
        )

    def make_target(self, game_history: GameHistory, state_index):
        target_values, target_rewards, target_policies, actions = [], [], [], []

        for current_index in range(
            state_index, state_index + self.num_unroll_steps + 1
        ):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                actions.append(game_history.actions[current_index])
                target_rewards.append(game_history.rewards[current_index])
                target_policies.append(game_history.child_visits[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(np.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions

    def sample_games(self, batch_size) -> GameHistory:
        return np.random.choice(self.game_histories, batch_size)

    def sample_position(self, game) -> int:
        return np.random.choice(len(game.root_values))

    def compute_target_value(self, game_history: GameHistory, index: int):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = game_history.root_values
            last_step_value = root_values[bootstrap_index]

            value = last_step_value * self.gamma ** self.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.rewards[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += reward * (self.gamma ** i)

        return value
