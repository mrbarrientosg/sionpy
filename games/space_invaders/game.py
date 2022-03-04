import os
import cv2
import gym
import numpy as np


class NoopResetEnv(gym.Wrapper):
    """
    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/env/atari_wrappers.py
    """

    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    """
    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/env/atari_wrappers.py
    """

    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
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


class EpisodicLifeEnv(gym.Wrapper):
    """
    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/env/atari_wrappers.py
    """

    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    """
    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/env/atari_wrappers.py
    """

    def __init__(self, env):
        """Take action on reset.
        For environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


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


# class Game(gym.Wrapper):
#     def __init__(self, env: gym.Env):
#         super().__init__(env)
#         self.env = NoopResetEnv(self.env, noop_max=130)
#         self.env = FireResetEnv(self.env)
#         self.env = RelativeObservation(
#             self.env,
#             [
#                 ("agent", cv2.imread(f"./assets/space_invaders/agent.png", 0)),
#                 ("bullet", cv2.imread(f"./assets/space_invaders/bullet2.png", 0)),
#             ]
#             + [
#                 ("enemy", cv2.imread(f"./assets/space_invaders/enemy_{i}.png", 0))
#                 for i in range(0, 6)
#             ]
#             + [
#                 ("enemy", cv2.imread(f"./assets/space_invaders/enemy_{i}{i}.png", 0))
#                 for i in range(0, 6)
#             ],
#         )


def make_game(id: str, logdir: str, seed: int):
    env = gym.make(id, frameskip=1, repeat_action_probability = 0.0, full_action_space=False)
    env = NoopResetEnv(env, noop_max=130)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = MaxAndSkipEnv(env, skip=5)
    env = RelativeObservation(
        env,
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
    env.seed(seed)

    return env


def make_game_test(id: str, logdir: str, seed: int):
    return gym.wrappers.RecordVideo(
        make_game(id, logdir, seed), os.path.join(logdir, "video")
    )


if __name__ == "__main__":
    env = make_game("ALE/SpaceInvaders-v5", logdir="", seed=0)
    print(env.action_space.n)
