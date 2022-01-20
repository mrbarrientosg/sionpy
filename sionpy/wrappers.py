import gym
from gym import spaces
from gym.core import Env
from gym.wrappers.frame_stack import FrameStack
import numpy as np
from gym.spaces import Box
import cv2


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=env.observation_space.dtype
        )
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.

    :param env: the environment
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return max(min(reward, 1), -1)


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    :param env: the environment to wrap
    :param noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            reward -= 100
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class RelativeObservation(gym.ObservationWrapper):
    def __init__(self, env, objects):
        super().__init__(env)
        self.objects = objects

    def observation(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        objects = self.find_objects(gray)

        if len(objects["agent"]) == 0:
            return np.zeros((1, 1, 30))

        for obj in objects:
            if obj == "agent":
                continue

            for item in objects[obj]:
                item[0] -= objects["agent"][0][0]
                item[1] = objects["agent"][0][1] - item[1]

        # take second element for sort
        def abs_x(elem):
            return abs(elem[0]) + 0.01 * elem[1]

        def second(elem):
            return abs(elem[1])

        objects["enemy"].sort(key=abs_x)
        objects["bullet"].sort(key=second)

        state = np.zeros((3, 10))
        state[0][0] = objects["agent"][0][0] / 160
        state[1][0] = objects["agent"][0][1] / 210

        i = 1
        for bull in objects["bullet"]:
            state[0][i] = bull[0] / 160
            state[1][i] = bull[1] / 210
            state[2][i] = np.sqrt(state[0][i] ** 2 + state[1][i] ** 2)
            i += 1
            if i > 2:
                break

        i = 5
        for enemy in objects["enemy"]:
            state[0][i] = enemy[0] / 160
            state[1][i] = enemy[1] / 210
            state[2][i] = np.sqrt(state[0][i] ** 2 + state[1][i] ** 2)
            i += 1
            if i >= 10:
                break

        state = state.flatten()
        state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state, axis=0)
        return state

    def find_objects(self, observation, threshold=0.95):
        obj_centroids = {}

        for obj in self.objects:
            if obj[0] not in obj_centroids:
                obj_centroids[obj[0]] = []

            res = cv2.matchTemplate(observation, obj[1], cv2.TM_CCOEFF)
            loc = np.argwhere(res >= threshold)

            w, h = obj[1].shape[::-1]

            for o in loc:
                y, x = o
                centroid = (x + (w / 2), y + (h / 2))
                obj_centroids[obj[0]].append(list(centroid))

        return obj_centroids


class Game(gym.Wrapper):
    def __init__(self, env: Env, skip_frames: int = 5):
        super().__init__(env)
        # self.env = NoopResetEnv(self.env, noop_max=30)
        # self.env = MaxAndSkipEnv(self.env, skip=skip_frames)
        # self.env = EpisodicLifeEnv(self.env)
        # self.env = ClipRewardEnv(self.env)
        self.env = RelativeObservation(
            self.env,
            [
                ("agent", cv2.imread(f"./assets/space_invaders/agent.png", 0)),
                ("bullet", cv2.imread(f"./assets/space_invaders/bullet2.png", 0)),
            ]
            + [
                ("enemy", cv2.imread(f"./assets/space_invaders/enemy_{i}.png", 0))
                for i in range(0, 6)
            ]
            + [
                ("enemy", cv2.imread(f"./assets/space_invaders/enemy_{i}{i}.png", 0))
                for i in range(0, 6)
            ],
        )

