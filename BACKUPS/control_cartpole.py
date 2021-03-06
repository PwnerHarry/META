import time, warnings, argparse, scipy.io, numpy.matlib, gym, numpy as np
from utils import *
from greedy import *
from mta import *
from AC import *
from TOTD import *
from TOGTD import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', type=float, default=0.01, help='')
parser.add_argument('--beta', type=float, default=0, help='')
parser.add_argument('--eta', type=float, default=0, help='')
parser.add_argument('--gamma', type=float, default=0.99, help='')
parser.add_argument('--kappa', type=float, default=0.0001, help='')
parser.add_argument('--episodes', type=int, default=1000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--learner_type', type=str, default='togtd', help='')
parser.add_argument('--evaluate_baselines', type=int, default=1, help='')
parser.add_argument('--evaluate_greedy', type=int, default=1, help='')
parser.add_argument('--evaluate_MTA', type=int, default=1, help='')
args = parser.parse_args()
if args.beta == 0:
    args.beta = 10.0 * args.alpha
if args.eta == 0:
    args.eta = 0.5 * args.alpha
# Experiment Preparation
env_name = 'CartPole-v1'
env, gamma = gym.make(env_name), lambda x: args.gamma
encoder = lambda x: x # tilecoding does not work!

things_to_save = {}
time_start = time.time()

# BASELINES
if args.evaluate_baselines:
    BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for baseline_lambda in BASELINE_LAMBDAS:
        results = eval_AC(env_name, critic_type='baseline', learner_type=args.learner_type, gamma=gamma, alpha=args.alpha, beta=args.beta, eta=args.eta, runtimes=args.runtimes, episodes=args.episodes, encoder=encoder, constant_lambda=baseline_lambda, kappa=args.kappa)
        exec("things_to_save[\'return_baseline_%g_mean\'] = np.nanmean(results, axis=0)" % (baseline_lambda * 100)) # no dots in variable names for MATLAB
        exec("things_to_save[\'return_baseline_%g_std\'] = np.nanstd(results, axis=0)" % (baseline_lambda * 100))

# LAMBDA-GREEDY
if args.evaluate_greedy:
    results = eval_AC(env_name, critic_type='greedy', learner_type=args.learner_type, gamma=gamma, alpha=args.alpha, beta=args.beta, eta=args.eta, runtimes=args.runtimes, episodes=args.episodes, encoder=encoder, constant_lambda=0, kappa=args.kappa)
    things_to_save['return_greedy_mean'], things_to_save['return_greedy_std'] = np.nanmean(results, axis=0), np.nanstd(results, axis=0)

# MTA
if args.evaluate_MTA:
    results = eval_AC(env_name, critic_type='MTA', learner_type=args.learner_type, gamma=gamma, alpha=args.alpha, beta=args.beta, eta=args.eta, runtimes=args.runtimes, episodes=args.episodes, encoder=encoder, constant_lambda=0, kappa=args.kappa)
    things_to_save['return_MTA_mean'], things_to_save['return_MTA_std'] = np.nanmean(results, axis=0), np.nanstd(results, axis=0)

time_finish = time.time()
print('time elapsed: %gs' % (time_finish - time_start))

# SAVE
if args.evaluate_MTA:
    filename = 'cartpole_a_%g_b_%g_y_%g_k_%g_e_%g_r_%d.mat' % (args.alpha, args.beta, args.eta, args.kappa, args.episodes, args.runtimes)
else:
    filename = 'cartpole_a_%g_b_%g_y_%g_e_%g_r_%d.mat' % (args.alpha, args.beta, args.eta, args.episodes, args.runtimes)
scipy.io.savemat(filename, things_to_save)