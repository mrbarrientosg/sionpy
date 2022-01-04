from typing import Tuple
from gym.spaces.space import Space
from torch import nn, Tensor
import torch
from torch.functional import F
from torch.nn.modules.activation import ELU
from torch.nn.modules.linear import Identity


class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: Tuple,
        stacked_observations: int,
        encoding_size: int,
        hidden_nodes: int = 32,
        activation_fn: nn.Module = nn.ELU,
    ):
        """
        Input: Recibe el stack de frames más las acciones aplicadas en cada una con un one hot encoding.
        Output: Retorna un Tensor de tamaño fijo prediciendo el estado siguiente.

        Args:
            observation_shape (Tuple): Tupla que contiene el tamaño de la observacion que nos entrega el ambiente.
            stacked_observations (int): Cantidad de frames que se van a stackear.
            encoding_size (int): El tamaño del output que nos va a entregar la red de siguiente estado.
            hidden_nodes (int, optional): Tamaño de la cantidad de nodos ocultos, por defecto 32.
            activation_fn (nn.Module, optional): Funcion de activación, por defecto nn.ELU.
        """
        super(RepresentationNetwork, self).__init__()
        self.activation = activation_fn
        self.linear1 = nn.Linear(
            observation_shape[0]
            * observation_shape[1]
            * observation_shape[2]
            * (stacked_observations + 1)
            + stacked_observations * observation_shape[1] * observation_shape[2],
            hidden_nodes,
        )
        self.linear2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear3 = nn.Linear(hidden_nodes, encoding_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation()(self.linear1(x))
        x = self.activation()(self.linear2(x))
        x = self.linear3(x)
        return x


class DynamicNetwork(nn.Module):
    def __init__(
        self,
        encoding_size: int,
        action_dim: int,
        hidden_nodes: int = 32,
        activation_fn: nn.Module = nn.ELU,
    ):
        """
        Input: Recibe el estado creado por la red :class:`RepresentationNetwork`.
        Output: Crea el siguiente estado y la recompensa de este.

        Args:
            encoding_size (int): Tamaño del estado creado por la red :class:`RepresentationNetwork`.
            action_dim (int): Tamaño de las acciones.
            hidden_nodes (int, optional): Tamaño de la cantidad de nodos ocultos, por defecto 32.
            activation_fn (nn.Module, optional): Funcion de activación, por defecto nn.ELU.
        """
        super(DynamicNetwork, self).__init__()

        self.action_dim = action_dim

        self.encoded_state_network = nn.Sequential(
            nn.Linear(encoding_size + action_dim, hidden_nodes),
            activation_fn(),
            nn.Linear(hidden_nodes, encoding_size),
        )

        self.reward_network = torch.nn.Sequential(
            nn.Linear(encoding_size, hidden_nodes), nn.ELU(), nn.Linear(hidden_nodes, 1)
        )

    def forward(self, encoded_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_dim)).to(action.device).float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.encoded_state_network(x)

        reward = self.reward_network(next_encoded_state)

        return next_encoded_state, reward


class SionNetwork(nn.Module):
    def __init__(
        self,
        observation_shape: Tuple,
        action_dim: int,
        stacked_observations: int,
        encoding_size: int,
        hidden_nodes: int = 32,
    ):
        super(SionNetwork, self).__init__()

        self.representation_network = RepresentationNetwork(
            observation_shape, stacked_observations, encoding_size, hidden_nodes
        )
        self.dynamic_network = DynamicNetwork(encoding_size, action_dim, hidden_nodes)

        self.policy_network = nn.Linear(encoding_size, action_dim)
        self.value_network = nn.Linear(encoding_size, 1)

    def prediction(self, encoded_state: Tensor) -> Tuple[Tensor, Tensor]:
        policy_logits = self.policy_network(encoded_state)
        value = self.value_network(encoded_state)
        return policy_logits, value

    def recurrent_inference(
        self, encoded_state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Funcion encargada de generar el siguiente estado y recompensa dentro
        del arbol de busqueda.

        Args:
            encoded_state (Tensor): Estado creado por la red dynamic.
            action (Tensor): La accion que fue aplicada en el estado.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: policy, value, next_encoded_state, reward
        """
        next_encoded_state, reward = self.dynamic_network(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return policy_logits, value, next_encoded_state, reward

    def initial_inference(self, observations: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Funcion que hace la primera inferencia de el stack de frames, esta nos creo el primer
        estado del nodo raiz del arbol.

        Args:
            observations (Tensor): Stack de frames de las observaciones del ambiente.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: policy, value, encoded_state
        """
        observations = observations.view(observations.shape[0], -1).float()
        encoded_state = self.representation_network(observations)
        policy_logits, value = self.prediction(encoded_state)
        
        reward = torch.log(
            (
                torch.zeros(1, 1)
                .scatter(1, torch.tensor([[1 // 2]]).long(), 1.0)
                .repeat(len(observations), 1)
                .to(observations.device)
            )
        )
        
        return policy_logits, value, encoded_state, reward
