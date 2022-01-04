import numpy as np
import torch
import math

from sionpy.network import SionNetwork


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
    def __init__(self, discount: float):
        self.discount = discount

    def run(self, model: SionNetwork, simulations: int, observation, actions, device):
        root = Node(0)

        observation = torch.tensor(observation).float().to(device).unsqueeze(0)

        logits, value, hidden_state, reward = model.initial_inference(observation)

        root.expand(actions, reward.item(), logits, hidden_state)

        for _ in range(simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            parent = search_path[-2]
            action = torch.tensor([[action]]).to(parent.hidden_state.device)
            logits, value, hidden_state, reward = model.recurrent_inference(
                parent.hidden_state, action
            )

            node.expand(actions, reward.item(), logits, hidden_state)

            self.backpropagate(search_path, value.item())

        return root

    def select_child(self, node):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child) for action, child in node.children.items()
        )

        action = np.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child) == max_ucb
            ]
        )

        return action, node.children[action]

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

            value = node.reward + self.discount * value

    def ucb_score(self, parent: Node, child: Node):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        prior_score = (
            child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        )
        if child.visit_count > 0:
            # The value of the child is from the perspective of the opposing player
            value_score = -child.value()
        else:
            value_score = 0

        return value_score + prior_score
