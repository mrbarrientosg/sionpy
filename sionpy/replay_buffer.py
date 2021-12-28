from typing import Callable, Iterator, Union
import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import IterableDataset


class ExperienceSourceDataset(IterableDataset):
    """Basic experience source dataset.
    Takes a generate_batch function that returns an iterator. The logic for the experience source and how the batch is
    generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch()
        return iterator


class ReplayBuffer:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def add(
        self, state: np.ndarray, action: int, reward: Union[int, float], mask: bool
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)

    def compute_returns(self, last_value: Tensor, gamma: float) -> Tensor:
        g = last_value
        returns = []

        for r, d in zip(self.rewards[::-1], self.masks[::-1]):
            g = r + gamma * g * (1 - d)
            returns.append(g)

        # reverse list and stop the gradients
        returns = torch.tensor(returns[::-1])

        return returns

    def get_stacked_observations(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.states)

        stacked_observations = self.states[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
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

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.masks.clear()
