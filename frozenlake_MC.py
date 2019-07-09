from utils import *
from joblib import Parallel, delayed
from MC import *
import numpy.matlib

env = gym.make('FrozenLake-v0'); env.reset()
N = env.observation_space.n
mc_episodes = int(5e8)
runtimes, gamma = 16, lambda x: 0.95

target_policy = np.matlib.repmat(np.ones((1, env.action_space.n)) / env.action_space.n, env.observation_space.n, 1)
# behavior_policy = np.matlib.repmat(np.ones((1, env.action_space.n)) / env.action_space.n, env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
filename = 'frozen_lake_ground_truths_uniform.npz'
try:
    loaded = np.load(filename)
    true_expectation, true_variance, stationary_dist = loaded['true_expectation'], loaded['true_variance'], loaded['stationary_dist']
except FileNotFoundError:
    E, V, return_counts = MC(env, mc_episodes, target_policy, target_policy, None, gamma)
    stationary_dist = return_counts / np.sum(return_counts)
    true_expectation, true_variance = E[-1], V[-1]
    np.savez(filename, true_expectation = true_expectation, true_variance = true_variance, stationary_dist = stationary_dist)
pass