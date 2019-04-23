from utils import *
from methods import *
from joblib import Parallel, delayed
from MC import *
import numpy.matlib
from frozen_lake import FrozenLakeEnv
import random

def rollout_MC_per_run(env, runtime, runtimes, episodes, target, gamma, Lambda, alpha, beta):
    print('rolling out %d of %d for MC' % (runtime + 1, runtimes))
    expected_return_trace, variance_of_return_trace, return_counts = MC(env, episodes, target, target, None, gamma)
    stationary_dist = return_counts / np.sum(return_counts)
    return (expected_return_trace[-1], variance_of_return_trace[-1], stationary_dist)

def rollout_MC(env, target, gamma = lambda x: 0.95, runtimes=20, episodes=100000):
    results = Parallel(n_jobs = -1)(delayed(rollout_MC_per_run)(env, runtime, runtimes, episodes, target, gamma, None, None, None) for runtime in range(runtimes))
    expectation_returns = [entry[0] for entry in results]
    variance_returns = [entry[1] for entry in results]
    state_dists = [entry[2] for entry in results]
    return np.concatenate(expectation_returns, axis=0), np.concatenate(variance_returns, axis=0), np.concatenate(state_dists, axis=0)

unit = 1
env = FrozenLakeEnv(None, '4x4', True, unit)
N = env.observation_space.n

episodes = int(1e7)
gamma = lambda x: 0.95
target_policy = np.matlib.repmat(np.array([0.2, 0.3, 0.3, 0.2]).reshape(1, 4), env.observation_space.n, 1)

while True:
    filename = 'frozenlake_truths_uniform_%d_%d.npz' % (episodes, random.randint(0, 1e6))
    true_expectation, true_variance, stationary_dist = rollout_MC(env, target_policy, gamma = gamma, runtimes=8, episodes=episodes)
    np.savez(filename, true_expectation = true_expectation, true_variance = true_variance, stationary_dist = stationary_dist)
