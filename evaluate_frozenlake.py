import numpy as np
import warnings, argparse, scipy.io, numpy.matlib, gym
from joblib import Parallel, delayed
from utils import *
from greedy import *
from mta import *
from MC import *
from true_online_TD import *
from true_online_GTD import *


parser = argparse.ArgumentParser(description='')
parser.add_argument('--N', type=int, default=4, help='')
parser.add_argument('--alpha', type=float, default=0.001, help='')
parser.add_argument('--beta', type=float, default=0.001, help='')
parser.add_argument('--kappa', type=float, default=0.001, help='')
parser.add_argument('--episodes', type=int, default=10000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--off_policy', type=int, default=0, help='')
parser.add_argument('--learner_type', type=str, default='togtd', help='')
args = parser.parse_args()

# experiment Preparation
# env = FrozenLakeEnv(None, '%dx%d' % (args.N, args.N), True)
env = gym.make('FrozenLake-v0'); env.reset()
runtimes, episodes, gamma = args.runtimes, args.episodes, lambda x: 0.95

target_policy = np.matlib.repmat(np.array([0.2, 0.3, 0.3, 0.2]).reshape(1, 4), env.observation_space.n, 1)
if args.off_policy == 0:
    behavior_policy = target_policy
else:
    behavior_policy = np.matlib.repmat(np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 4), env.observation_space.n, 1)
alpha, beta, kappa = args.alpha, args.beta, args.kappa

# get ground truth expectation, variance and stationary distribution
# filename = 'frozenlake_truths_%dx%d.npz' % (args.N, args.N)
# loaded = np.load(filename)
# true_expectation, true_variance, stationary_dist = loaded['true_expectation'], loaded['true_variance'], loaded['stationary_dist']
# # TODO: check if it is MDP????
# dp_expectation, dp_variance, dp_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)

# print('difference of two expectations: %.2e' % (numpy.linalg.norm(true_expectation - dp_expectation, 2)))
# print('difference of two variances: %.2e' % (numpy.linalg.norm(true_variance - dp_variance, 2)))
# print('difference of two stat_dists: %.2e' % (numpy.linalg.norm(stationary_dist - dp_dist, 2)))

true_expectation, true_variance, stationary_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)

evaluate = lambda estimate, stat_type: evaluate_estimate(estimate, true_expectation, true_variance, stationary_dist, stat_type, encoder)
N = args.N ** 2
things_to_save = {}
encoder = lambda s: index2plane(s, args.N)
# encoder = lambda s: onehot(s, N)

# BASELINES
BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
for baseline_lambda in BASELINE_LAMBDAS:
    Lambda = LAMBDA(env, baseline_lambda, approximator='constant')
    if args.learner_type == 'togtd':
        results = eval_togtd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes, evaluate=evaluate, encoder=encoder)
    elif args.learner_type == 'totd':
        results = eval_totd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=alpha, runtimes=runtimes, episodes=episodes, evaluate=evaluate, encoder=encoder)
    exec("things_to_save[\'error_value_%s_%g_mean\'] = np.nanmean(results, axis=0)" % (args.learner_type, baseline_lambda * 100)) # no dots in variable names for MATLAB
    exec("things_to_save[\'error_value_%s_%g_std\'] = np.nanstd(results, axis=0)" % (args.learner_type, baseline_lambda * 100))

# LAMBDA-GREEDY
error_value_greedy = eval_greedy(env, behavior_policy, target_policy, gamma=gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
things_to_save['error_value_greedy_mean'], things_to_save['error_value_greedy_std'] = np.nanmean(error_value_greedy, axis=0), np.nanstd(error_value_greedy, axis=0)

# MTA
error_value_mta = eval_MTA(env, behavior_policy, target_policy, kappa=kappa, gamma=gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
things_to_save['error_value_mta_mean'], things_to_save['error_value_mta_std'] = np.nanmean(error_value_mta, axis=0), np.nanstd(error_value_mta, axis=0)

filename = 'frozenlake_%s_behavior_%g_target_%g_alpha_%g_beta_%g_kappa_%g_episodes_%g' % (args.learner_type, behavior_policy[0, 0], target_policy[0, 0], alpha, beta, kappa, episodes)
scipy.io.savemat(filename, things_to_save)
pass