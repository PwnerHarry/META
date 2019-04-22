from utils import *
from methods import *
from joblib import Parallel, delayed
from MC import *
from true_online_GTD import *
import numpy.matlib
from frozen_lake import FrozenLakeEnv

unit = 1
env = FrozenLakeEnv(None, '4x4', True, unit)
N = env.observation_space.n

runtimes = 9
mc_episodes = int(1e7)
gamma = lambda x: 0.95
runtime = 0
target_policy = np.matlib.repmat(np.ones((1, env.action_space.n)) / env.action_space.n, env.observation_space.n, 1)
true_expectations = np.zeros((runtimes, env.action_space.n ** 2))
true_variances = np.zeros((runtimes, env.action_space.n ** 2))
stationary_dists = np.zeros((runtimes, env.action_space.n ** 2))

while runtime < runtimes:
    filename = 'frozen_lake_ground_truths_uniform_%d_%d.npz' % (mc_episodes, runtime)
    loaded = np.load(filename)
    true_expectations[runtime, :] = loaded['true_expectation']
    true_variances[runtime, :] = loaded['true_variance']
    stationary_dists[runtime, :] = loaded['stationary_dist']
    runtime += 1
filename = 'frozen_lake_ground_truths_uniform.npz'
np.savez(filename, true_expectation = np.mean(true_expectations, axis=0), true_variance = np.mean(true_variances, axis=0), stationary_dist = np.mean(stationary_dists, axis=0))
