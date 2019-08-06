import time, warnings, argparse, scipy.io, numpy.matlib, gym, numpy as np
from utils import *
from greedy import *
from mta import *
from MC import *
from TOTD import *
from TOGTD import *
import frozenlake

parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', type=float, default=0.001, help='')
parser.add_argument('--beta', type=float, default=0, help='')
parser.add_argument('--gamma', type=float, default=0.95, help='')
parser.add_argument('--kappa', type=float, default=0.001, help='')
parser.add_argument('--episodes', type=int, default=1000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--off_policy', type=int, default=1, help='')
parser.add_argument('--learner_type', type=str, default='togtd', help='')
parser.add_argument('--evaluate_others', type=int, default=1, help='')
parser.add_argument('--evaluate_MTA', type=int, default=1, help='')
args = parser.parse_args()
if args.beta == 0:
    args.beta = args.alpha

# Experiment Preparation
env_name, gamma, encoder = 'FrozenLake-v1', lambda x: args.gamma, lambda s: tilecoding4x4(s)
env = gym.make(env_name)
target = np.matlib.repmat(np.array([0.2, 0.3, 0.3, 0.2]).reshape(1, 4), env.observation_space.n, 1)
if args.off_policy:
    behavior = np.matlib.repmat(np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 4), env.observation_space.n, 1)
else:
    behavior = np.copy(target)

start_dist = np.zeros(env.observation_space.n); start_dist[0] = 1.0
DP_expectation, DP_variance, DP_stat_dist = iterative_policy_evaluation(env, target, gamma=gamma, start_dist=start_dist)
evaluate = lambda estimate, stat_type: evaluate_estimate(estimate, DP_expectation, DP_variance, DP_stat_dist, stat_type, get_state_set_matrix(env, encoder))

things_to_save = {}
time_start = time.time()

if args.evaluate_others:
    # BASELINES
    BASELINE_LAMBDAS = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    for baseline_lambda in BASELINE_LAMBDAS:
        Lambda = LAMBDA(env, baseline_lambda, approximator='constant')
        results = eval_togtd(env_name, behavior, target, Lambda, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder)
        exec("things_to_save[\'error_value_%s_%g_mean\'] = np.nanmean(results, axis=0)" % (args.learner_type, baseline_lambda * 1000)) # no dots in variable names for MATLAB
        exec("things_to_save[\'error_value_%s_%g_std\'] = np.nanstd(results, axis=0)" % (args.learner_type, baseline_lambda * 1000))
    # LAMBDA-GREEDY
    error_value_greedy = eval_greedy(env_name, behavior, target, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
    things_to_save['error_value_greedy_mean'], things_to_save['error_value_greedy_std'] = np.nanmean(error_value_greedy, axis=0), np.nanstd(error_value_greedy, axis=0)

# MTA
if args.evaluate_MTA:
    error_value_mta = eval_MTA(env_name, behavior, target, kappa=args.kappa, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
    things_to_save['error_value_mta_mean'], things_to_save['error_value_mta_std'] = np.nanmean(error_value_mta, axis=0), np.nanstd(error_value_mta, axis=0)

time_finish = time.time()
print('time elapsed: %gs' % (time_finish - time_start))

# SAVE
if args.evaluate_MTA:
    if args.off_policy:
        filename = 'frozenlake_off_a_%g_b_%g_k_%g_e_%g_r_%d.mat' % (args.alpha, args.beta, args.kappa, args.episodes, args.runtimes)
    else:
        filename = 'frozenlake_on_a_%g_k_%g_e_%g_r_%d.mat' % (args.alpha, args.kappa, args.episodes, args.runtimes)
else:
    if args.off_policy:
        filename = 'frozenlake_off_a_%g_b_%g_e_%g_r_%d.mat' % (args.alpha, args.beta, args.episodes, args.runtimes)
    else:
        filename = 'frozenlake_on_a_%g_e_%g_r_%d.mat' % (args.alpha, args.episodes, args.runtimes)
scipy.io.savemat(filename, things_to_save)