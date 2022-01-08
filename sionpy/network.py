from collections import namedtuple
from typing import Tuple
from torch import nn
from torch import Tensor
from torch.functional import F
from torch.distributions import Categorical

ActorCriticOutput = namedtuple(
    "ActorCriticOutput", field_names=["actions", "values", "log_probs"]
)


class ActorModel(nn.Module):
    def __init__(self, hidden_nodes: int, action_dim: int):
        super(ActorModel, self).__init__()
        self.net = nn.Linear(hidden_nodes, action_dim)

    def action_distribution(self, x: Tensor) -> Categorical:
        x = self.net.forward(x)
        return Categorical(logits=x)

    def log_probs(self, distribution: Categorical, actions: Tensor) -> Tensor:
        return distribution.log_prob(actions)


class CriticModel(nn.Module):
    def __init__(self, hidden_nodes: int):
        super(CriticModel, self).__init__()
        self.net = nn.Linear(hidden_nodes, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.net.forward(x).squeeze(1)


class ActorCriticModel(nn.Module):
    def __init__(
        self,
        observation_shape: tuple,
        action_dim: int,
        stacked_observations: int,
        hidden_nodes: int = 64,
    ):
        super(ActorCriticModel, self).__init__()

        self.body = nn.Sequential(
            nn.Linear(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                hidden_nodes,
            ),
            nn.ReLU(),
        )

        self.actor = ActorModel(hidden_nodes, action_dim)
        self.critic = CriticModel(hidden_nodes)

    def forward(self, observations: Tensor) -> ActorCriticOutput:
        observations = observations.view(observations.shape[0], -1).float()
        x = self.body.forward(observations)
        x = self.normalize_encoded_state(x)
        policy = self.actor.action_distribution(x)
        actions = policy.sample()
        log_probs = self.actor.log_probs(policy, actions)
        values = self.critic.forward(x)
        return ActorCriticOutput(actions, values, log_probs)

    def evaluation_actions(
        self, observations: Tensor, actions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        observations = observations.view(observations.shape[0], -1).float()
        x = self.body.forward(observations)
        x = self.normalize_encoded_state(x)
        policy = self.actor.action_distribution(x)
        log_probs = self.actor.log_probs(policy, actions)
        values = self.critic.forward(x)
        return values, log_probs

    def normalize_encoded_state(self, encoded_state: Tensor):
        mean, std = encoded_state.mean(), encoded_state.std()
        encoded_state_normalized = encoded_state - mean / (std + 1e-8)
        return encoded_state_normalized
