import os
import cv2
import gym
import numpy as np
from sionpy.transformation import DATE


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
            # self.was_real_done = True
            done = True
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
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # self.env = EpisodicLifeEnv(self.env)
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


def make_game(id: str, logdir: str, seed: int):
    env = gym.make(id)
    env.seed(seed)
    return Game(env)


def make_game_test(id: str, logdir: str, seed: int):
    return gym.wrappers.RecordVideo(make_game(id, logdir, seed), os.path.join(logdir, "video"))
