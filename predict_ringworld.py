import time, argparse, warnings
import scipy.io
import numpy as np
import numpy.matlib as npm
from greedy import *
from mta import *
from TOGTD import *
from TOTD import *
from utils import *
import ringworld

parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', type=float, default=0.01, help='')
parser.add_argument('--beta', type=float, default=0, help='does not mean anyting if using TOTD!')
parser.add_argument('--kappa', type=float, default=0.1, help='')
parser.add_argument('--gamma', type=float, default=0.95, help='')
parser.add_argument('--steps', type=int, default=10000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--target', type=float, default=0.35, help='')
parser.add_argument('--behavior', type=float, default=0.4, help='')
parser.add_argument('--learner_type', type=str, default='totd', help='')
parser.add_argument('--evaluate_others', type=int, default=1, help='')
parser.add_argument('--evaluate_MTA', type=int, default=1, help='')
args = parser.parse_args()

if args.beta == 0:
    args.beta = args.alpha

# experiment Preparation
env_name, gamma, encoder = 'RingWorld-v0', lambda x: args.gamma, lambda s: onehot(s, env.observation_space.n)
env = gym.make(env_name)
target_policy, behavior_policy = npm.repmat(np.array([args.target, 1 - args.target]).reshape(1, -1), env.observation_space.n, 1), npm.repmat(np.array([args.behavior, 1 - args.behavior]).reshape(1, -1), env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
start_dist = np.zeros(env.observation_space.n)
start_dist[int(env.observation_space.n / 2)] = 1.0
DP_expectation, DP_variance, DP_stat_dist, index_terminal = iterative_policy_evaluation(env, target_policy, gamma=gamma, start_dist=start_dist)
DP_stat_dist_sqrt = np.sqrt(DP_stat_dist)
DP_stat_dist_sqrt[index_terminal] = 0
evaluate = lambda estimate, stat_type: evaluate_estimate(estimate, DP_expectation, DP_variance, DP_stat_dist_sqrt, stat_type, get_state_set_matrix(env, encoder))

things_to_save = {}
time_start = time.time()

if args.evaluate_others:
    things_to_save = {}
    time_start = time.time()
    # BASELINES
    BASELINE_LAMBDAS = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    for baseline_lambda in BASELINE_LAMBDAS:
        Lambda = LAMBDA(env, baseline_lambda, approximator='constant')
        if args.learner_type == 'totd':
            results = eval_totd(env_name, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=args.alpha, runtimes=args.runtimes, steps=args.steps, evaluate=evaluate, encoder=encoder)
        elif args.learner_type == 'togtd':
            results = eval_togtd(env_name, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, steps=args.steps, evaluate=evaluate, encoder=encoder)
        exec("things_to_save[\'error_value_%s_%g_mean\'] = np.nanmean(results, axis=0)" % (args.learner_type, baseline_lambda * 1000)) # no dots in variable names for MATLAB
        exec("things_to_save[\'error_value_%s_%g_std\'] = np.nanstd(results, axis=0)" % (args.learner_type, baseline_lambda * 1000))
    # LAMBDA-GREEDY
    error_value_greedy = eval_greedy(env_name, behavior_policy, target_policy, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, steps=args.steps, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
    things_to_save['error_value_greedy_mean'], things_to_save['error_value_greedy_std'] = np.nanmean(error_value_greedy, axis=0), np.nanstd(error_value_greedy, axis=0)
    time_finish = time.time()
    print('time elapsed: %gs' % (time_finish - time_start))
    # SAVE
    filename = 'ringworld_behavior_%g_target_%g_' % (behavior_policy[0, 0], target_policy[0, 0])
    if args.learner_type == "togtd":
        filename = filename + 'a_%g_b_%g_' % (args.alpha, args.beta)
    else:
        filename = filename + 'a_%g_' % (args.alpha)
    filename = filename + 'e_%g_r_%d.mat' % (args.steps, args.runtimes)
    scipy.io.savemat(filename, things_to_save)

# MTA
if args.evaluate_MTA:
    things_to_save = {}
    time_start = time.time()
    error_value_mta = eval_MTA(env_name, behavior_policy, target_policy, kappa=args.kappa, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, steps=args.steps, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
    things_to_save['error_value_mta_mean'], things_to_save['error_value_mta_std'] = np.nanmean(error_value_mta, axis=0), np.nanstd(error_value_mta, axis=0)
    time_finish = time.time()
    print('time elapsed: %gs' % (time_finish - time_start))
    # SAVE
    filename = 'ringworld_behavior_%g_target_%g_' % (behavior_policy[0, 0], target_policy[0, 0])
    if args.learner_type == "togtd":
        filename = filename + 'a_%g_b_%g_' % (args.alpha, args.beta)
    else:
        filename = filename + 'a_%g_' % (args.alpha)
    filename = filename + 'k_%g_e_%g_r_%d.mat' % (args.kappa, args.steps, args.runtimes)
    scipy.io.savemat(filename, things_to_save)