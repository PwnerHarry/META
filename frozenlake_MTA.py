import matplotlib as mpl
mpl.use('Agg')
from utils import *
from methods import *
from greedy import *
from mta import *
import numpy as np
import warnings, argparse, scipy.io
from joblib import Parallel, delayed
from MC import *
from true_online_GTD import *
import numpy.matlib
from frozen_lake import FrozenLakeEnv

parser = argparse.ArgumentParser(description='')
parser.add_argument('--N', type=int, default=4, help='')
parser.add_argument('--alpha', type=float, default=0.05, help='')
parser.add_argument('--beta', type=float, default=0.05, help='')
parser.add_argument('--kappa', type=float, default=0.1, help='')
parser.add_argument('--episodes', type=int, default=1000, help='')
parser.add_argument('--runtimes', type=int, default=160, help='')
parser.add_argument('--off_policy', type=int, default=0, help='')
args = parser.parse_args()

unit = 1.0
# experiment Preparation
env = FrozenLakeEnv(None, '%dx%d' % (args.N, args.N), True, unit)
runtimes, episodes, gamma = args.runtimes, args.episodes, lambda x: 0.95

target_policy = np.matlib.repmat(np.array([0.2, 0.3, 0.3, 0.2]).reshape(1, 4), env.observation_space.n, 1)
if args.off_policy == 0:
    behavior_policy = target_policy
else:
    behavior_policy = np.matlib.repmat(np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 4), env.observation_space.n, 1)
alpha, beta, kappa = args.alpha, args.beta, args.kappa

# get ground truth expectation, variance and stationary distribution
filename = 'frozen_lake_ground_truths_uniform.npz'
loaded = np.load(filename)
true_expectation, true_variance, stationary_dist = loaded['true_expectation'], loaded['true_variance'], loaded['stationary_dist']
true_expectation = true_expectation * unit
true_variance = true_variance * (unit ** 2)

BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
things_to_save = {}
for baseline_lambda in BASELINE_LAMBDAS:
    Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = baseline_lambda * np.ones(args.N))
    results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)
    exec("things_to_save[\'togtd_%g_results\'] = results.copy()" % (baseline_lambda * 1e5))

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

filename = 'frozenlake_N_%s_behavior_%g_target_%g_episodes_%g' % (args.N, behavior_policy[0, 0], target_policy[0, 0], episodes)
scipy.io.savemat(filename, things_to_save)
pass