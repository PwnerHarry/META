import time, warnings, argparse, scipy.io, numpy.matlib, gym, numpy as np
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
parser.add_argument('--episodes', type=int, default=100000, help='')
parser.add_argument('--runtimes', type=int, default=15, help='')
parser.add_argument('--off_policy', type=int, default=0, help='')
parser.add_argument('--learner_type', type=str, default='togtd', help='')
parser.add_argument('--evaluate_baselines', type=int, default=1, help='')
parser.add_argument('--evaluate_greedy', type=int, default=1, help='')
parser.add_argument('--evaluate_MTA', type=int, default=1, help='')
args = parser.parse_args()

# Experiment Preparation
env = gym.make('FrozenLake-v0')
gamma, encoder = lambda x: args.gamma, lambda s: tilecoding4x4(s)
target = np.matlib.repmat(np.array([0.2, 0.3, 0.3, 0.2]).reshape(1, 4), env.observation_space.n, 1)
if args.off_policy:
    behavior = np.matlib.repmat(np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 4), env.observation_space.n, 1)
else:
    behavior = np.copy(target)

start_dist = np.zeros(env.observation_space.n); start_dist[0] = 1.0
DP_expectation, DP_variance, DP_stat_dist = iterative_policy_evaluation(env, target, gamma=gamma, start_dist=start_dist)
try:
    filename = 'frozenlake_truths_heuristic_1e+09.npz'
    loaded = np.load(filename)
    MC_expectation, MC_variance, MC_stat_dist = loaded['true_expectation'], loaded['true_variance'], loaded['stationary_dist']
    print('difference between expectations: %.2e' % (np.linalg.norm(DP_expectation.reshape(-1) - MC_expectation.reshape(-1), 2)))
    print('difference between variances: %.2e' % (np.linalg.norm(DP_variance.reshape(-1) - MC_variance.reshape(-1), 2)))
    print('difference between stationary distributions: %.2e' % (np.linalg.norm(DP_stat_dist.reshape(-1) - MC_stat_dist.reshape(-1), 2)))
    evaluate = lambda estimate, stat_type: evaluate_estimate(estimate, MC_expectation, MC_variance, MC_stat_dist, stat_type, get_state_set_matrix(env, encoder))
except FileNotFoundError:
    print('MC simulation results not loaded!')
    evaluate = lambda estimate, stat_type: evaluate_estimate(estimate, DP_expectation, DP_variance, DP_stat_dist, stat_type, get_state_set_matrix(env, encoder))

things_to_save = {}
time_start = time.time()
# BASELINES
if args.evaluate_baselines:
    BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for baseline_lambda in BASELINE_LAMBDAS:
        Lambda = LAMBDA(env, baseline_lambda, approximator='constant')
        results = eval_togtd(env, behavior, target, Lambda, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder)
        exec("things_to_save[\'error_value_%s_%g_mean\'] = np.nanmean(results, axis=0)" % (args.learner_type, baseline_lambda * 100)) # no dots in variable names for MATLAB
        exec("things_to_save[\'error_value_%s_%g_std\'] = np.nanstd(results, axis=0)" % (args.learner_type, baseline_lambda * 100))

# LAMBDA-GREEDY
if args.evaluate_greedy:
    error_value_greedy = eval_greedy(env, behavior, target, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
    things_to_save['error_value_greedy_mean'], things_to_save['error_value_greedy_std'] = np.nanmean(error_value_greedy, axis=0), np.nanstd(error_value_greedy, axis=0)

# MTA
if args.evaluate_MTA:
    error_value_mta = eval_MTA(env, behavior, target, kappa=args.kappa, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
    things_to_save['error_value_mta_mean'], things_to_save['error_value_mta_std'] = np.nanmean(error_value_mta, axis=0), np.nanstd(error_value_mta, axis=0)

time_finish = time.time()
print('time elapsed: %gs' % (time_finish - time_start))

# SAVE
if args.evaluate_MTA:
    if args.off_policy:
        filename = 'frozenlake_off_a_%g_b_%g_k_%g_e_%g_r_%d' % (args.alpha, args.beta, args.kappa, args.episodes, args.runtimes)
    else:
        filename = 'frozenlake_on_a_%g_k_%g_e_%g_r_%d' % (args.alpha, args.kappa, args.episodes, args.runtimes)
else:
    if args.off_policy:
        filename = 'frozenlake_off_a_%g_b_%g_e_%g_r_%d' % (args.alpha, args.beta, args.episodes, args.runtimes)
    else:
        filename = 'frozenlake_on_a_%g_e_%g_r_%d' % (args.alpha, args.episodes, args.runtimes)
scipy.io.savemat(filename, things_to_save)