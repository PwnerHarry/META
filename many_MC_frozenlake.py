from utils import *
from methods import *
from joblib import Parallel, delayed
from MC import *
from true_online_GTD import *
import numpy.matlib
from frozen_lake import FrozenLakeEnv
import random

unit = 1
env = FrozenLakeEnv(None, '4x4', True, unit)
N = env.observation_space.n

mc_episodes = int(1e3)
gamma = lambda x: 0.95
target_policy = np.matlib.repmat(np.array([0.2, 0.3, 0.3, 0.2]).reshape(1, 4), env.observation_space.n, 1)

while True:
    filename = 'frozenlake_ground_truths_uniform_%d_%d.npz' % (mc_episodes, random.randint(0, 1e6))
    E, V, return_counts = MC(env, mc_episodes, target_policy, target_policy, None, gamma)
    stationary_dist = return_counts / np.sum(return_counts)
    true_expectation, true_variance = E[-1], V[-1]
    np.savez(filename, true_expectation = true_expectation, true_variance = true_variance, stationary_dist = stationary_dist)
