from utils import *
from methods import *
from greedy import *
import numpy as np
import numpy.matlib as npm
import scipy.io

# experiment Preparation
N = 25; env, runtimes, episodes, gamma = RingWorldEnv(N), int(80), int(1e4), lambda x: 0.95
alpha, beta = 0.05, 0.05
target_policy = npm.repmat(np.array([0.25, 0.75]).reshape(1, -1), env.observation_space.n, 1)
behavior_policy = npm.repmat(np.array([0.25, 0.75]).reshape(1, -1), env.observation_space.n, 1)
# get ground truth expectation, variance and stationary distribution
true_expectation, true_variance, stationary_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)

error_E, error_V, error_value, lambda_trace = eval_greedy(env, true_expectation, true_variance, stationary_dist, behavior_policy, target_policy, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 1.0 * np.ones(N))
off_togtd_10_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0.8 * np.ones(N))
off_togtd_08_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0.6 * np.ones(N))
off_togtd_06_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0.4 * np.ones(N))
off_togtd_04_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0.2 * np.ones(N))
off_togtd_02_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0 * np.ones(N))
off_togtd_00_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

things_to_save = {}

things_to_save['error_E'] = error_E
things_to_save['error_V'] = error_V
things_to_save['error_value'] = error_value
things_to_save['lambda_trace'] = lambda_trace
things_to_save['off_togtd_00_results'] = off_togtd_00_results
things_to_save['off_togtd_02_results'] = off_togtd_02_results
things_to_save['off_togtd_04_results'] = off_togtd_04_results
things_to_save['off_togtd_06_results'] = off_togtd_06_results
things_to_save['off_togtd_08_results'] = off_togtd_08_results
things_to_save['off_togtd_10_results'] = off_togtd_10_results

filename = 'greedy_N_%s_behavior_%g_target_%g_episodes_%g' % (N, behavior_policy[0, 0], target_policy[0, 0], episodes)
scipy.io.savemat(filename, things_to_save)
pass