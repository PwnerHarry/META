from utils import *
from methods import *
from joblib import Parallel, delayed
from MC import *
from true_online_GTD import *
import numpy.matlib, os
from frozen_lake import FrozenLakeEnv

unit = 1
env = FrozenLakeEnv(None, '4x4', True, unit)
N = env.observation_space.n

runtimes = 10
mc_episodes = int(1e7)
gamma = lambda x: 0.95
runtime = 0
target_policy = np.matlib.repmat(np.ones((1, env.action_space.n)) / env.action_space.n, env.observation_space.n, 1)
true_expectations = np.zeros((runtimes, env.action_space.n ** 2))
true_variances = np.zeros((runtimes, env.action_space.n ** 2))
stationary_dists = np.zeros((runtimes, env.action_space.n ** 2))

cumulative_expectation = np.zeros((1, env.action_space.n ** 2))
cumulative_variance = np.zeros((1, env.action_space.n ** 2))
cumulative_distribution = np.zeros((1, env.action_space.n ** 2))
count = 0

directory = 'frozenlake'
filelist = os.listdir(directory)
for filename in filelist:
    if filename.find('%d' % mc_episodes) == -1: continue
    loaded = np.load(directory + '/' + filename)
    cumulative_expectation += loaded['true_expectation'].reshape(1, -1)
    cumulative_variance += loaded['true_variance'].reshape(1, -1)
    cumulative_distribution += loaded['stationary_dist'].reshape(1, -1)
    count += 1

filename = 'frozenlake_truths_4x4.npz'
np.savez(filename, true_expectation = cumulative_expectation / count, true_variance = cumulative_variance / count, stationary_dist = cumulative_distribution / count)