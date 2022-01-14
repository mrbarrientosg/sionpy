from collections import namedtuple
from turtle import forward
from typing import NamedTuple, Tuple
from torch import nn
from torch import Tensor
import torch
from torch.functional import F
from torch.distributions import Categorical

from sionpy.config import Config

# ActorCriticOutput = namedtuple(
#     "ActorCriticOutput", field_names=["value", "reward", "probs", "encoded_state"]
# )


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
    def __init__(self, hidden_nodes: int):
        super(CriticModel, self).__init__()
        self.net = nn.Linear(hidden_nodes, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x)


class ActorCriticModel(nn.Module):
    def __init__(
        self, config: Config,
    ):
        super(ActorCriticModel, self).__init__()
        self.action_dim = len(config.action_space)

        self.representation = nn.Sequential(
            nn.Linear(
                config.observation_shape[0]
                * config.observation_shape[1]
                * config.observation_shape[2]
                * (config.stacked_observations + 1)
                + config.stacked_observations
                * config.observation_shape[1]
                * config.observation_shape[2],
                config.hidden_nodes,
            ),
            nn.ReLU(),
            nn.Linear(config.hidden_nodes, config.encoding_size),
            nn.Tanh(),
        )

        self.dynamic_state = nn.Sequential(
            nn.Linear(config.encoding_size + self.action_dim, config.hidden_nodes),
            nn.ReLU(),
            nn.Linear(config.hidden_nodes, config.encoding_size),
            nn.Tanh(),
        )

        self.dynamic_reward = nn.Linear(config.encoding_size, 1)
        self.actor = ActorModel(config.encoding_size, self.action_dim)
        self.critic = CriticModel(config.encoding_size)

    def initial_inference(self, observation: Tensor) -> ActorCriticOutput:
        observations = observation.view(observation.shape[0], -1).float()
        encoded_state = self.representation.forward(observations)
        policy = self.actor.forward(encoded_state)
        value = self.critic.forward(encoded_state)

        reward = (
            torch.tensor([0.0]).float().repeat(len(observation)).to(observation.device)
        )

        return ActorCriticOutput(value, reward, policy, encoded_state)

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

        policy = self.actor.forward(next_enconded_state)
        value = self.critic.forward(next_enconded_state)

        return ActorCriticOutput(value, reward, policy, next_enconded_state)

