import matplotlib as mpl
mpl.use('Agg')
from utils import *
from methods import *
from greedy import *
from mta import *
import numpy as np
import numpy.matlib as npm
import warnings, argparse, scipy.io

parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', type=float, default=0.05, help='')
parser.add_argument('--beta', type=float, default=0.05, help='')
parser.add_argument('--kappa', type=float, default=0.1, help='')
parser.add_argument('--episodes', type=int, default=1000, help='')
parser.add_argument('--runtimes', type=int, default=160, help='')
parser.add_argument('--N', type=int, default=11, help='')
parser.add_argument('--target', type=float, default=0.4, help='')
parser.add_argument('--behavior', type=float, default=0.4, help='')
args = parser.parse_args()

# experiment Preparation
N = args.N; env, runtimes, episodes, gamma = RingWorldEnv(args.N, unit = 1), args.runtimes, args.episodes, lambda x: 0.95
alpha, beta, kappa = args.alpha, args.beta, args.kappa
target_policy = npm.repmat(np.array([args.target, 1 - args.target]).reshape(1, -1), env.observation_space.n, 1)
behavior_policy = npm.repmat(np.array([args.behavior, 1 - args.behavior]).reshape(1, -1), env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
true_expectation, true_variance, stationary_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)

error_value, lambda_trace_mta, error_L_var  = eval_MTA(env, true_expectation, true_variance, stationary_dist, behavior_policy, target_policy, kappa = kappa, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

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

_, error_var_greedy, direct_greedy_results, lambda_trace_greedy = eval_greedy(env, true_expectation, true_variance, stationary_dist, behavior_policy, target_policy, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

things_to_save = {}

things_to_save['error_value'] = error_value
things_to_save['error_L_var'] = error_L_var
things_to_save['error_var_greedy'] = error_var_greedy
things_to_save['lambda_trace_mta'] = lambda_trace_mta
things_to_save['lambda_trace_greedy'] = lambda_trace_greedy
things_to_save['off_togtd_00_results'] = off_togtd_00_results
things_to_save['off_togtd_02_results'] = off_togtd_02_results
things_to_save['off_togtd_04_results'] = off_togtd_04_results
things_to_save['off_togtd_06_results'] = off_togtd_06_results
things_to_save['off_togtd_08_results'] = off_togtd_08_results
things_to_save['off_togtd_10_results'] = off_togtd_10_results
things_to_save['direct_greedy_results'] = direct_greedy_results

filename = 'ringworld_N_%s_behavior_%g_target_%g_episodes_%g' % (N, behavior_policy[0, 0], target_policy[0, 0], episodes)
scipy.io.savemat(filename, things_to_save)

pass