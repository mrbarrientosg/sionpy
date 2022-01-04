from typing import Callable, Iterator, List, Union
import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import IterableDataset


class GameHistory:
    def __init__(
        self,
        discount: float,
        num_unroll_steps: int,
        td_steps: int,
        action_space: List[int],
        num_stacked_observations: int,
    ):
        self.states = []
        self.actions = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        self.action_space = action_space
        self.num_stacked_observations = num_stacked_observations

    def add(self, state: np.ndarray, action: int, reward: Union[int, float]):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def store_search_stats(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in self.action_space
            ]
        )

        self.root_values.append(root.value())

    def get_stacked_observations(self, index):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.states)

        stacked_observations = self.states[index].copy()
        for past_observation_index in reversed(
            range(index - self.num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = np.concatenate(
                    (
                        self.states[past_observation_index],
                        [
                            np.ones_like(stacked_observations[0])
                            * self.actions[past_observation_index + 1]
                        ],
                    )
                )
            else:
                previous_observation = np.concatenate(
                    (
                        np.zeros_like(self.states[index]),
                        [np.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = np.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations

    def make_target(self, state_index: int):
        """Generate targets to learn from during the network training."""

        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.num_unroll_steps + 1
        ):
            bootstrap_index = current_index + self.td_steps
            if bootstrap_index < len(self.root_values):
                value = (
                    self.root_values[bootstrap_index] * self.discount ** self.td_steps
                )
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            if current_index < len(self.root_values):
                target_values.append(value)
                target_rewards.append(self.rewards[current_index])
                target_policies.append(self.child_visits[current_index])
                actions.append(self.actions[current_index])
            else:
                # States past the end of games are treated as absorbing states.
                target_values.append(0)
                target_rewards.append(0)
                target_policies.append(
                    [
                        1 / len(self.child_visits[0])
                        for _ in range(len(self.child_visits[0]))
                    ]
                )
                actions.append(np.random.choice(self.action_space))
        return target_values, target_rewards, target_policies, actions

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def __iter__(self) -> Iterator:
        return zip(self.states, self.actions, self.rewards)


class SimpleBatch:
    def __init__(self, batch, device):
        transposed_data = list(zip(*batch))
        self.observation_batch = (
            torch.tensor(np.array(transposed_data[0])).float().to(device)
        )
        self.action_batch = (
            torch.tensor(transposed_data[1]).long().to(device).unsqueeze(-1)
        )
        self.target_reward = torch.tensor(transposed_data[2]).float().to(device)
        self.target_value = torch.tensor(transposed_data[3]).float().to(device)
        self.target_policy = torch.tensor(transposed_data[4]).float().to(device)


class ReplayBuffer:
    def __init__(self, window_size: int):
        self.buffer: List[GameHistory] = []
        self.window_size = window_size
        self.epoch_rewards, self.epoch_mean_value, self.epoch_length = [], [], []

    def __len__(self):
        return len(self.buffer)

    def save_game(self, game_history):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game_history)

    def update_statistic(self, game_history: GameHistory):
        self.epoch_rewards.append(sum(game_history.rewards))
        self.epoch_mean_value.append(sum(game_history.root_values))
        self.epoch_length.append(len(game_history.actions) - 1)

    def reset_statistic(self):
        self.epoch_rewards.clear()
        self.epoch_mean_value.clear()
        self.epoch_length.clear()

    def get_statistic(self):
        return (
            np.mean(self.epoch_rewards),
            np.mean(self.epoch_mean_value),
            np.mean(self.epoch_length),
        )

    def sample_batch(self):
        (observation_batch, action_batch, reward_batch, value_batch, policy_batch,) = (
            [],
            [],
            [],
            [],
            [],
        )

        for game_history in self.buffer:
            game_pos = self.sample_pos(game_history)
            values, rewards, policies, actions = game_history.make_target(game_pos)

            observation_batch.append(game_history.get_stacked_observations(game_pos))
            action_batch.append(actions)
            reward_batch.append(rewards)
            value_batch.append(values)
            policy_batch.append(policies)

        return zip(
            observation_batch, action_batch, reward_batch, value_batch, policy_batch,
        )

    def sample_pos(self, game_history: GameHistory):
        return np.random.choice(len(game_history.actions))


# class ReplayDataset(IterableDataset):
#     def __init__(self, generate_batch: Callable[[bool], ReplayBuffer], test=False):
#         self.test = test
#         self.generate_batch = generate_batch

#     def __iter__(self) -> Iterator:
#         return self.generate_batch(self.test).sample_batch()
