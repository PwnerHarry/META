from utils import *
from methods import *
from greedy import *
from mta import *
import numpy as np
import numpy.matlib as npm
import warnings, argparse, scipy.io
from true_online_GTD import *
from true_online_TD import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', type=float, default=0.05, help='')
parser.add_argument('--beta', type=float, default=0.05, help='')
parser.add_argument('--episodes', type=int, default=10000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--N', type=int, default=11, help='')
parser.add_argument('--target', type=float, default=0.05, help='')
parser.add_argument('--behavior', type=float, default=0.05, help='')
parser.add_argument('--learner_type', type=str, default='togtd', help='')
args = parser.parse_args()

# experiment Preparation
N = args.N; env, runtimes, episodes, gamma = RingWorldEnv(args.N, unit = 1), args.runtimes, args.episodes, lambda x: 0.95
alpha, beta = args.alpha, args.beta
target_policy = npm.repmat(np.array([args.target, 1 - args.target]).reshape(1, -1), env.observation_space.n, 1)
behavior_policy = npm.repmat(np.array([args.behavior, 1 - args.behavior]).reshape(1, -1), env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
true_expectation, true_variance, stationary_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)
evaluate = lambda estimate, stat_type: evaluate_estimate(estimate, true_expectation, true_variance, stationary_dist, stat_type)
things_to_save = {}

BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
for baseline_lambda in BASELINE_LAMBDAS:
    Lambda = LAMBDA(env, baseline_lambda, approximator = 'constant')
    if args.learner_type == 'totd':
        results = eval_totd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=alpha, runtimes=runtimes, episodes=episodes, evaluate=evaluate)
    elif args.learner_type == 'togtd':
        results = eval_togtd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes, evaluate=evaluate)
    exec("things_to_save[\'error_value_%s_%g_mean\'] = np.nanmean(results, axis=0)" % (args.learner_type, baseline_lambda * 100)) # no dots in variable names for MATLAB
    exec("things_to_save[\'error_value_%s_%g_std\'] = np.nanstd(results, axis=0)" % (args.learner_type, baseline_lambda * 100))

filename = 'ringworld_%s_baselines_N_%d_behavior_%g_target_%g_episodes_%g' % (args.learner_type, N, behavior_policy[0, 0], target_policy[0, 0], episodes)
scipy.io.savemat(filename, things_to_save)
pass