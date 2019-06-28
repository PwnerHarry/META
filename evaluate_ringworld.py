import numpy as np
import numpy.matlib as npm
import warnings, argparse, scipy.io
from utils import *
from methods import *
from greedy import *
from mta import *
from true_online_GTD import *
from true_online_TD import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', type=float, default=0.05, help='')
parser.add_argument('--beta', type=float, default=0.05, help='')
parser.add_argument('--kappa', type=float, default=0.01, help='')
parser.add_argument('--episodes', type=int, default=10000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--N', type=int, default=11, help='')
parser.add_argument('--target', type=float, default=0.05, help='')
parser.add_argument('--behavior', type=float, default=0.05, help='')
parser.add_argument('--learner_type', type=str, default='totd', help='')
args = parser.parse_args()

# experiment Preparation
N = args.N; env, runtimes, episodes, gamma = RingWorldEnv(args.N, unit = 1), args.runtimes, args.episodes, lambda x: 0.95
alpha, beta, kappa = args.alpha, args.beta, args.kappa
target_policy = npm.repmat(np.array([args.target, 1 - args.target]).reshape(1, -1), env.observation_space.n, 1)
behavior_policy = npm.repmat(np.array([args.behavior, 1 - args.behavior]).reshape(1, -1), env.observation_space.n, 1)
encoder = lambda s: onehot(s, env.observation_space.n)

# get ground truth expectation, variance and stationary distribution
true_expectation, true_variance, stationary_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)
evaluate = lambda estimate, stat_type: evaluate_estimate(estimate, true_expectation, true_variance, stationary_dist, stat_type, encoder)
things_to_save = {}

# BASELINES
BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
for baseline_lambda in BASELINE_LAMBDAS:
    Lambda = LAMBDA(env, baseline_lambda, approximator = 'constant')
    if args.learner_type == 'totd':
        results = eval_totd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=alpha, runtimes=runtimes, episodes=episodes, evaluate=evaluate, encoder=encoder)
    elif args.learner_type == 'togtd':
        results = eval_togtd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes, evaluate=evaluate, encoder=encoder)
    exec("things_to_save[\'error_value_%s_%g_mean\'] = np.nanmean(results, axis=0)" % (args.learner_type, baseline_lambda * 100)) # no dots in variable names for MATLAB
    exec("things_to_save[\'error_value_%s_%g_std\'] = np.nanstd(results, axis=0)" % (args.learner_type, baseline_lambda * 100))

# LAMBDA-GREEDY
error_value_greedy = eval_greedy(env, behavior_policy, target_policy, gamma=gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
things_to_save['error_value_greedy_mean'], things_to_save['error_value_greedy_std'] = np.nanmean(error_value_greedy, axis=0), np.nanstd(error_value_greedy, axis=0)

# MTA
error_value_mta = eval_MTA(env, behavior_policy, target_policy, kappa=kappa, gamma=gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
things_to_save['error_value_mta_mean'], things_to_save['error_value_mta_std'] = np.nanmean(error_value_mta, axis=0), np.nanstd(error_value_mta, axis=0)

filename = 'ringworld_%s_behavior_%g_target_%g_episodes_%g' % (args.learner_type, behavior_policy[0, 0], target_policy[0, 0], episodes)
scipy.io.savemat(filename, things_to_save)
pass