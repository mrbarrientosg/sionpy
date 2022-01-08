from collections import namedtuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

Experience = namedtuple(
    "Experience", field_names=["state", "observation", "action", "reward", "mask"],
)


class BufferDataset(Dataset):
    def __init__(self, data: "ReplayBuffer"):
        self.data = data

    def __len__(self):
        return len(self.data.rewards)

    def __getitem__(self, idx):
        return (
            self.data.observations[idx],
            self.data.actions[idx],
            self.data.returns[idx],
        )


class ReplayBuffer:
    def __init__(self, num_stacked_observations: int, batch_size: int):
        self.states = []
        self.observations = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.returns = []
        self.num_stacked_observations = num_stacked_observations
        self.batch_size = batch_size

    def add(self, exp: Experience):
        self.observations.append(exp.observation)
        self.states.append(exp.state)
        self.actions.append(exp.action)
        self.rewards.append(exp.reward)
        self.masks.append(exp.mask)

    def get_stacked_observations(self, index: int) -> np.ndarray:
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

    def compute_advantage_returns(self, last_value: Tensor, gamma: float = 0.95):
        g = last_value
        self.returns.clear()

        for step in reversed(range(len(self.rewards))):
            g = self.rewards[step] + gamma * g * self.masks[step]
            self.returns.insert(0, g)

    def batch(self):
        actions = self.actions[1:]
        size = len(actions)

        for idx in range(0, size, self.batch_size):
            yield (
                self.log_probs[idx : min(idx + self.batch_size, size)],
                self.values[idx : min(idx + self.batch_size, size)],
                self.returns[idx : min(idx + self.batch_size, size)],
            )

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.masks.clear()
        self.states.clear()
        self.returns.clear()
