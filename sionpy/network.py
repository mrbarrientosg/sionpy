from typing import List, NamedTuple, Tuple
from torch import nn
from torch import Tensor
import torch
from sionpy.config import Config, NetworkTopology
from abc import ABC, abstractmethod


class NetworkOutput(NamedTuple):
    value: Tensor
    reward: Tensor
    logits: Tensor
    hidden_state: Tensor


class AbstractNetwork(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.full_support_size = 0

    @abstractmethod
    def representation(self, observation: Tensor) -> Tensor:
        pass

    @abstractmethod
    def prediction(self, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def dynamics(self, hidden_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    def initial_inference(self, observation: Tensor) -> NetworkOutput:
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)

        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return NetworkOutput(value, reward, policy_logits, hidden_state)

    def recurrent_inference(
        self, hidden_state: Tensor, action: Tensor
    ) -> NetworkOutput:
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return NetworkOutput(value, reward, policy_logits, next_hidden_state)

    def normalize_state(self, encoded_state: Tensor) -> Tensor:
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized


class SionNetwork:
    def __new__(cls, config: Config) -> AbstractNetwork:
        if config.topology == NetworkTopology.FULLY:
            return SionFullyConnectedNetwork(config)
        else:
            raise NotImplementedError()


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


class SionFullyConnectedNetwork(AbstractNetwork):
    def __init__(self, config: Config):
        super().__init__()
        self.action_space_size = len(config.action_space)
        self.full_support_size = 2 * config.support_size + 1

        self.representation_network = mlp(
            config.observation_shape[0]
            * config.observation_shape[1]
            * config.observation_shape[2]
            * (config.stacked_observations + 1)
            + config.stacked_observations
            * config.observation_shape[1]
            * config.observation_shape[2],
            config.fc_representation_layers,
            config.encoding_size,
            activation=nn.Tanh,
        )

        self.dynamics_state_network = mlp(
            config.encoding_size + self.action_space_size,
            config.fc_dynamics_layers,
            config.encoding_size,
            activation=nn.Tanh,
        )

        self.dynamics_reward_network = mlp(
            config.encoding_size + self.action_space_size,
            config.fc_reward_layers,
            self.full_support_size,
            activation=nn.LeakyReLU,
        )

        self.prediction_policy_network = mlp(
            config.encoding_size,
            config.fc_policy_layers,
            self.action_space_size,
            activation=nn.LeakyReLU,
        )

        self.prediction_value_network = mlp(
            config.encoding_size,
            config.fc_value_layers,
            self.full_support_size,
            activation=nn.LeakyReLU,
        )

    def prediction(self, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        policy_logits = self.prediction_policy_network.forward(hidden_state)
        value = self.prediction_value_network.forward(hidden_state)
        return policy_logits, value

    def representation(self, observation: Tensor) -> Tensor:
        return self.normalize_state(
            self.representation_network.forward(
                observation.view(observation.shape[0], -1)
            )
        )

    def dynamics(self, hidden_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((hidden_state, action_one_hot), dim=1)

        next_hidden_state = self.dynamics_state_network.forward(x)

        reward = self.dynamics_reward_network.forward(x)

        return self.normalize_state(next_hidden_state), reward
