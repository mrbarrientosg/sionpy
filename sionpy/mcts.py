import numpy as np
import torch
import math
from sionpy.config import Config
from sionpy.network import ActorCriticModel


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float("inf")
        self.minimum = max_value_bound if max_value_bound else float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.reward = 0
        self.hidden_state = None
        self.children = {}
        self.value_sum = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, reward, logits, hidden_state):
        self.reward = reward
        self.hidden_state = hidden_state

        policy_v = torch.softmax(
            torch.tensor([logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_v[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)


class MCTS:
    def __init__(self, config: Config):
        self.config = config

    def run(
        self, model: ActorCriticModel, simulations: int, observation, actions, device
    ):
        min_max_stats = MinMaxStats()
        root = Node(0)

        observation = torch.tensor(observation).float().to(device).unsqueeze(0)

        output = model.initial_inference(observation)
        root_predicted_value = output.value.item()

        root.expand(actions, output.reward.item(), output.logits, output.encoded_state)

        max_tree_depth = 0
        for _ in range(simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

            parent = search_path[-2]
            action = torch.tensor([[action]]).long().to(parent.hidden_state.device)
            output = model.recurrent_inference(parent.hidden_state, action)

            node.expand(
                actions, output.reward.item(), output.logits, output.encoded_state
            )

            self.backpropagate(search_path, output.value.item(), min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        return (
            root,
            {
                "max_tree_depth": max_tree_depth,
                "root_predicted_value": root_predicted_value,
            },
        )

    def select_child(self, node: Node, min_max_stats: MinMaxStats):
        _, action, child = max(
            (self.ucb_score(node, child, min_max_stats), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def backpropagate(self, search_path, value, min_max_stats: MinMaxStats):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.epsilon_gamma * node.value())

            value = node.reward + self.config.epsilon_gamma * value

    def ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats):
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())
        return prior_score + value_score
