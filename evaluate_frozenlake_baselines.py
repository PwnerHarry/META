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
parser.add_argument('--episodes', type=int, default=int(1e7), help='')
parser.add_argument('--runtimes', type=int, default=16, help='')
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
alpha, beta = args.alpha, args.beta

# get ground truth expectation, variance and stationary distribution
filename = 'frozenlake_truths_%dx%d.npz' % (args.N, args.N)
loaded = np.load(filename)
true_expectation, true_variance, stationary_dist = loaded['true_expectation'], loaded['true_variance'], loaded['stationary_dist']
true_expectation = true_expectation * unit
true_variance = true_variance * (unit ** 2)
evaluation = lambda estimate, stat_type: evaluate_estimate(estimate, true_expectation, true_variance, stationary_dist, stat_type)
N = args.N ** 2
things_to_save = {}

BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
for baseline_lambda in BASELINE_LAMBDAS:
    Lambda = LAMBDA(env, baseline_lambda, approximator = 'constant')
    results = eval_togtd(env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes, evaluate=evaluate)
    exec("things_to_save[\'error_value_togtd_%g_mean\'] = np.nanmean(results, axis=0)" % (baseline_lambda * 100)) # no dots in variable names for MATLAB
    exec("things_to_save[\'error_value_togtd_%g_std\'] = np.nanstd(results, axis=0)" % (baseline_lambda * 100))

filename = 'frozenlake_baselines_N_%s_behavior_%g_target_%g_episodes_%g_kappa_%g' % (N, behavior_policy[0, 0], target_policy[0, 0], episodes, kappa)
scipy.io.savemat(filename, things_to_save)
pass