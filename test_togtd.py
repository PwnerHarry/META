import matplotlib as mpl
mpl.use('Agg')
from utils import *
from methods import *
from greedy import *
from mta import *
import numpy as np
import numpy.matlib as npm
import warnings
import scipy.io

# experiment Preparation
N = 11; env, runtimes, episodes, gamma = RingWorldEnv(N, unit = 1000), int(80), int(1e3), lambda x: 0.95
alpha, beta = 0.05, 0.05
target_policy = npm.repmat(np.array([0.05, 0.95]).reshape(1, -1), env.observation_space.n, 1)
behavior_policy = npm.repmat(np.array([0.05, 0.95]).reshape(1, -1), env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
true_expectation, true_variance, stationary_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)

BASELINE_LAMBDAS = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 1]
things_to_save = {}

for baseline_lambda in BASELINE_LAMBDAS:
    Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = baseline_lambda * np.ones(N))
    results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)
    exec("things_to_save[\'togtd_%g_results\'] = results.copy()" % (baseline_lambda * 1e5))
    
filename = 'togtd_N_%s_behavior_%g_target_%g_episodes_%g' % (N, behavior_policy[0, 0], target_policy[0, 0], episodes)
scipy.io.savemat(filename, things_to_save)
pass