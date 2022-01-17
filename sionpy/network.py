from collections import namedtuple
from turtle import forward
from typing import NamedTuple, Tuple
from torch import nn
from torch import Tensor
import torch
from torch.functional import F
from torch.distributions import Categorical

from sionpy.config import Config


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


class ActorCriticOutput(NamedTuple):
    value: Tensor
    reward: Tensor
    logits: Tensor
    encoded_state: Tensor


class ActorModel(nn.Module):
    def __init__(self, hidden_nodes: int, action_dim: int):
        super(ActorModel, self).__init__()
        self.net = nn.Linear(hidden_nodes, action_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x)


class CriticModel(nn.Module):
    def __init__(self, hidden_nodes: int, support_size: int):
        super(CriticModel, self).__init__()
        self.net = nn.Linear(hidden_nodes, support_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x)


class ActorCriticModel(nn.Module):
    def __init__(
        self, config: Config,
    ):
        super(ActorCriticModel, self).__init__()
        self.action_dim = len(config.action_space)
        self.full_support_size = 2 * config.support_size + 1

        self.representation = mlp(
            config.observation_shape[0]
            * config.observation_shape[1]
            * config.observation_shape[2]
            * (config.stacked_observations + 1)
            + config.stacked_observations
            * config.observation_shape[1]
            * config.observation_shape[2],
            [],
            config.encoding_size,
        )

        self.dynamic_state = mlp(
            config.encoding_size + self.action_dim,
            [config.hidden_nodes],
            config.encoding_size,
        )

        self.dynamic_reward = mlp(
            config.encoding_size, [config.hidden_nodes], self.full_support_size
        )
        self.actor = mlp(config.encoding_size, [], self.action_dim)
        self.critic = mlp(
            config.encoding_size, [], self.full_support_size
        )

    def initial_inference(self, observation: Tensor) -> ActorCriticOutput:
        encoded_state = self.representation.forward(
            observation.view(observation.shape[0], -1)
        )
        encoded_state_normalized = self.normalize_state(encoded_state)
        policy = self.actor.forward(encoded_state_normalized)
        value = self.critic.forward(encoded_state_normalized)

        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return ActorCriticOutput(value, reward, policy, encoded_state_normalized)

    def recurrent_inference(
        self, encoded_state: Tensor, action: Tensor
    ) -> ActorCriticOutput:

        action_one_hot = (
            torch.zeros((action.shape[0], self.action_dim)).to(action.device).float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat([encoded_state, action_one_hot], dim=1)

        next_enconded_state = self.dynamic_state.forward(x)
        reward = self.dynamic_reward.forward(next_enconded_state)

        encoded_state_normalized = self.normalize_state(next_enconded_state)

        policy = self.actor.forward(encoded_state_normalized)
        value = self.critic.forward(encoded_state_normalized)

        return ActorCriticOutput(value, reward, policy, encoded_state_normalized)

    def normalize_state(self, encoded_state: Tensor) -> Tensor:
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized
