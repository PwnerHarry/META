import numpy as np
import gym
import gym.utils.seeding

class RingWorldEnv(gym.Env):
    def __init__(self, N, unit=1):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(N)
        # self.episode_length, self.episode_time = N * 1e2, 0
        self.seed()
        self.unit = unit
        P = {}
        for s in range(self.observation_space.n):
            small_dict = {}
            for a in [0, 1]:
                increment = -1 if a == 0 else 1
                if s == 0 or s == self.observation_space.n - 1:
                    entry = [(1.0, s, 0, True)]
                elif s + increment == self.observation_space.n - 1:
                    entry = [(1.0, self.observation_space.n - 1, self.unit, True)]
                elif s + increment == 0:
                    entry = [(1.0, 0, -self.unit, True)]
                else:
                    entry = [(1.0, s + increment, 0, False)]
                small_dict[a] = entry
            P[s] = small_dict
        self.unwrapped.P = P
        self.unwrapped.reward_range = (-1, 1)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        increment = -1 if action == 0 else 1
        # self.episode_time += 1
        self.state = (self.state + increment) % self.observation_space.n
        # flag = self.episode_time >= self.episode_length
        if self.state == 0:
            return self.state, -self.unit, True, {}
        elif self.state == self.observation_space.n - 1:
            return self.state, self.unit, True, {}
        return self.state, 0, False, {}

    def reset(self):
        self.state = int(self.observation_space.n / 2)
        self.episode_time = 0
        return self.state