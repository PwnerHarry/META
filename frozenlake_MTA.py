from utils import *
from methods import *
from joblib import Parallel, delayed
from MC import *
from true_online_GTD import *
import numpy.matlib
from frozen_lake import FrozenLakeEnv
import scipy.io
from greedy import *
from mta import *

unit = 1.0
kappa = 0.1
# experiment Preparation
env = FrozenLakeEnv(None, '4x4', True, unit)
N = env.observation_space.n
mc_episodes = int(1e7)
runtimes, episodes, gamma = 8, int(1e5), lambda x: 0.95

target_policy = np.matlib.repmat(np.ones((1, env.action_space.n)) / env.action_space.n, env.observation_space.n, 1)
behavior_policy = np.matlib.repmat(np.ones((1, env.action_space.n)) / env.action_space.n, env.observation_space.n, 1)
alpha, beta = 0.05, 0.05

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

true_expectation = true_expectation * unit
true_variance = true_variance * (unit ** 2)

# BASELINE_LAMBDAS = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875]
# things_to_save = {}
# for baseline_lambda in BASELINE_LAMBDAS:
#     Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = baseline_lambda * np.ones(N))
#     results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)
#     exec("things_to_save[\'togtd_%g_results\'] = results.copy()" % (baseline_lambda * 1e5))

_, error_L_exp, error_L_var, error_value, lambda_trace_mta = eval_MTA(env, true_expectation, true_variance, stationary_dist, behavior_policy, target_policy, kappa = kappa, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)
_, error_var_greedy, direct_greedy_results, lambda_trace_greedy = eval_greedy(env, true_expectation, true_variance, stationary_dist, behavior_policy, target_policy, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)
 
things_to_save = {}

things_to_save['error_value'] = error_value
things_to_save['error_L_exp'] = error_L_exp
things_to_save['error_L_var'] = error_L_var
things_to_save['error_var_greedy'] = error_var_greedy
things_to_save['lambda_trace_mta'] = lambda_trace_mta
things_to_save['lambda_trace_greedy'] = lambda_trace_greedy
things_to_save['direct_greedy_results'] = direct_greedy_results


filename = 'frozenlake_N_%s_behavior_%g_target_%g_episodes_%g' % (N, behavior_policy[0, 0], target_policy[0, 0], episodes)
scipy.io.savemat(filename, things_to_save)
pass