from typing import Tuple
from gym.spaces.space import Space
from torch import nn, Tensor
import torch
from torch.functional import F
from torch.nn.modules.activation import ELU
from torch.nn.modules.linear import Identity


class MlpNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: Tuple,
        action_dim: int,
        stacked_observations: int,
        encoding_size: int,
        hidden_nodes: int = 32,
    ):
        super(MlpNetwork, self).__init__()

        self.representation = nn.Sequential(
            nn.Linear(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                hidden_nodes,
            ),
            nn.ELU(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.ELU(),
            nn.Linear(hidden_nodes, encoding_size),
            nn.Identity()
        )
        self.actor_head = nn.Linear(encoding_size, action_dim)
        self.critic_head = nn.Linear(encoding_size, 1)
        
    def prediction(self, encoded_state: Tensor):
        policy_logits = self.actor_head(encoded_state)
        value = self.critic_head(encoded_state)
        return policy_logits, value
        
    def initial_inference(self, x: Tensor):
        x = x.view(x.shape[0], -1).float()
        encoded_state = self.representation(x)
        policy_logits, value = self.prediction(encoded_state)
        logprobs = F.log_softmax(policy_logits, dim=-1)
        return logprobs, value, encoded_state
