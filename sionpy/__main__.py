import gym
from sionpy.a2c import A2C
from sionpy.wrappers import Game

if __name__ == "__main__":
    env = gym.make("SpaceInvaders-v0", full_action_space=False)
    env = Game(env)

    model = A2C(env)
    model.train()
