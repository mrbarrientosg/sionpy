from typing import Tuple
from torch import nn, Tensor
from torch.functional import F


class MlpNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_nodes: int = 128,
        activation_fn: nn.Module = nn.Tanh,
    ):
        super(MlpNetwork, self).__init__()

        self.activation = activation_fn
        self.fc1 = nn.Linear(input_dim, hidden_nodes)
        self.actor_head = nn.Linear(hidden_nodes, action_dim)
        self.critic_head = nn.Linear(hidden_nodes, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.activation(self.fc1(x))

        probs = F.log_softmax(self.actor_head(x), dim=-1)
        value = self.critic_head(x)
        return probs, value
