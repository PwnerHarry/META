from utils import *
from joblib import Parallel, delayed
from MC import *
import numpy.matlib

env = gym.make('FrozenLake-v0'); env.reset()
N = env.observation_space.n
mc_episodes = int(1e9)
gamma = lambda x: 0.95

target_policy = np.matlib.repmat(np.array([0.2, 0.3, 0.3, 0.2]).reshape(1, 4), env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
filename = 'frozenlake_truths_heuristic_%g.npz' % mc_episodes
try:
    loaded = np.load(filename)
    true_expectation, true_variance, stationary_dist = loaded['true_expectation'], loaded['true_variance'], loaded['stationary_dist']
except FileNotFoundError:
    true_expectation, true_variance, return_counts = MC(env, mc_episodes, target_policy, target_policy, gamma)
    stationary_dist = return_counts / np.sum(return_counts)
    np.savez(filename, true_expectation=true_expectation, true_variance=true_variance, stationary_dist=stationary_dist)
pass