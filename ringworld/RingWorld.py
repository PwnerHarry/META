import gym, gym.spaces
from gym.envs.toy_text import discrete
from gym.utils import seeding

class RingWorldEnv(discrete.DiscreteEnv):
    def __init__(self):
        N = 11
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(N)
        P = {}
        for s in range(self.observation_space.n):
            small_dict = {}
            for a in [0, 1]:
                increment = -1 if a == 0 else 1
                if s == 0 or s == self.observation_space.n - 1:
                    entry = [(1.0, s, 0, True)]
                elif s + increment == self.observation_space.n - 1:
                    entry = [(1.0, self.observation_space.n - 1, 1, True)]
                elif s + increment == 0:
                    entry = [(1.0, 0, -1, True)]
                else:
                    entry = [(1.0, s + increment, 0, False)]
                small_dict[a] = entry
            P[s] = small_dict
        self.unwrapped.P = P
        self.unwrapped.reward_range = (-1, 1)
    def step(self, action):
        increment = -1 if action == 0 else 1
        self.state = (self.state + increment) % self.observation_space.n
        if self.state == 0:
            return self.state, -1, True, {}
        elif self.state == self.observation_space.n - 1:
            return self.state, 1, True, {}
        return self.state, 0, False, {}
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self):
        self.state = int(self.observation_space.n / 2)
        return self.state