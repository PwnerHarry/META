import warnings, argparse, scipy.io, numpy.matlib, gym, numpy as np
from joblib import Parallel, delayed
from utils import *
from greedy import *
from mta import *
from MC import *
from TOTD import *
from TOGTD import *


parser = argparse.ArgumentParser(description='')
parser.add_argument('--N', type=int, default=4, help='')
parser.add_argument('--alpha', type=float, default=0.001, help='')
parser.add_argument('--beta', type=float, default=0.001, help='')
parser.add_argument('--gamma', type=float, default=0.95, help='')
parser.add_argument('--kappa', type=float, default=0.001, help='')
parser.add_argument('--episodes', type=int, default=10000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--off_policy', type=int, default=0, help='')
parser.add_argument('--learner_type', type=str, default='togtd', help='')
parser.add_argument('--evaluate_baselines', type=int, default=1, help='')
parser.add_argument('--evaluate_greedy', type=int, default=1, help='')
parser.add_argument('--evaluate_MTA', type=int, default=1, help='')
args = parser.parse_args()

# experiment Preparation
env = gym.make('FrozenLake-v0'); env.reset()
gamma, encoder = lambda x: args.gamma, lambda s: index2plane(s, args.N)

target_policy = np.matlib.repmat(np.array([0.2, 0.3, 0.3, 0.2]).reshape(1, 4), env.observation_space.n, 1)
if args.off_policy == 0:
    behavior_policy = target_policy
else:
    behavior_policy = np.matlib.repmat(np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 4), env.observation_space.n, 1)

true_expectation, true_variance, stationary_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)
evaluate = lambda estimate, stat_type: evaluate_estimate(estimate, true_expectation, true_variance, stationary_dist, stat_type, encoder)


things_to_save = {}

# BASELINES
if args.evaluate_baselines:
    BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for baseline_lambda in BASELINE_LAMBDAS:
        Lambda = LAMBDA(env, baseline_lambda, approximator='constant')
        # if args.learner_type == 'togtd':
        results = eval_togtd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder)
        # elif args.learner_type == 'totd':
        #     results = eval_totd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=args.alpha, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder)
        exec("things_to_save[\'error_value_%s_%g_mean\'] = np.nanmean(results, axis=0)" % (args.learner_type, baseline_lambda * 100)) # no dots in variable names for MATLAB
        exec("things_to_save[\'error_value_%s_%g_std\'] = np.nanstd(results, axis=0)" % (args.learner_type, baseline_lambda * 100))

# LAMBDA-GREEDY
if args.evaluate_greedy:
    error_value_greedy = eval_greedy(env, behavior_policy, target_policy, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
    things_to_save['error_value_greedy_mean'], things_to_save['error_value_greedy_std'] = np.nanmean(error_value_greedy, axis=0), np.nanstd(error_value_greedy, axis=0)

# MTA
if args.evaluate_MTA:
    error_value_mta = eval_MTA(env, behavior_policy, target_policy, kappa=args.kappa, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
    things_to_save['error_value_mta_mean'], things_to_save['error_value_mta_std'] = np.nanmean(error_value_mta, axis=0), np.nanstd(error_value_mta, axis=0)

filename = 'frozenlake_behavior_%g_target_%g_a_%g_b_%g_k_%g_e_%g_r_%d' % (behavior_policy[0, 0], target_policy[0, 0], args.alpha, args.beta, args.kappa, args.episodes, args.runtimes)
scipy.io.savemat(filename, things_to_save)
pass