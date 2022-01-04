from gym import Env
import numpy as np
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.utils.data.dataset import IterableDataset
from sionpy.mcts import MCTS, Node
from sionpy.network import SionNetwork
from sionpy.replay_buffer import GameHistory, ReplayBuffer


class ExperienceSourceDataset(IterableDataset):
    def __init__(
        self,
        env: Env,
        net: SionNetwork,
        buffer: ReplayBuffer,
        hparams: AttributeDict,
        device: str,
        test: bool = False,
    ):
        self.env = env
        self.net = net
        self.buffer = buffer
        self.hparams = hparams
        self.test = test
        self.device = device

    # def __len__(self):
    #     return len(self.buffer)

    def select_action(self, node: Node, temperature: float = 1.0):
        visit_counts = np.array([child.visit_count for child in node.children.values()])
        actions = [action for action in node.children.keys()]
        if temperature == 0.0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def __iter__(self):
        self.buffer.reset_statistic()

        for _ in range(self.hparams.max_episodes):
            observation = self.env.reset()
            game_history = GameHistory(
                self.hparams.discount,
                self.hparams.num_unroll_steps,
                self.hparams.td_steps,
                self.hparams.action_space,
                self.hparams.stacked_observations,
            )
            game_history.states.append(observation)
            game_history.actions.append(0)
            game_history.rewards.append(0)
            done = False
            while not done and len(game_history.actions) <= self.hparams.max_moves:
                observation = game_history.get_stacked_observations(-1)

                root = MCTS(self.hparams.discount).run(
                    self.net,
                    self.hparams.simulations,
                    observation,
                    self.hparams.action_space,
                    self.device,
                )

                action = self.select_action(root, 0.0 if self.test else 1.0)

                observation, reward, done, _ = self.env.step(action)

                game_history.add(observation, action, reward)
                game_history.store_search_stats(root)

                if self.test:
                    self.env.render()

            self.buffer.update_statistic(game_history)
            self.buffer.save_game(game_history)

        # self.dict_log["avg_reward"] = np.mean(epoch_rewards)
        # self.dict_log["avg_mean_value"] = np.mean(epoch_mean_value)
        # self.dict_log["avg_length"] = np.mean(epoch_length)

        return self.buffer.sample_batch()
