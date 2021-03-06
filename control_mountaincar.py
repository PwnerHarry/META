import time, warnings, argparse, scipy.io, numpy.matlib, gym, numpy as np
from utils import *
from greedy import *
from mta import *
from AC import *
from TOTD import *
from TOGTD import *
import mountaincar

parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', type=float, default=0.01, help='')
parser.add_argument('--beta', type=float, default=0, help='')
parser.add_argument('--eta', type=float, default=0, help='')
parser.add_argument('--gamma', type=float, default=1, help='')
parser.add_argument('--kappa', type=float, default=1e-5, help='')
parser.add_argument('--steps', type=int, default=20000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--learner_type', type=str, default='togtd', help='')
parser.add_argument('--evaluate_others', type=int, default=1, help='')
parser.add_argument('--evaluate_MTA', type=int, default=1, help='')
parser.add_argument('--parametric_lambda', type=int, default=1, help='')
args = parser.parse_args()

if args.eta == 0:
    args.eta = args.alpha

if args.beta == 0:
    args.beta = args.alpha

# Experiment Preparation
env_name = 'MountainCar-v1'
env, gamma = gym.make(env_name), lambda x: args.gamma
encoder = lambda x: tile_encoding(x, env.observation_space.shape[0], env.observation_space.low, env.observation_space.high, 8, 8)
if args.parametric_lambda:
    encoder_lambda = encoder
else:
    encoder_lambda = lambda x: state_aggregation_2d(x, env.observation_space.low, env.observation_space.high, 64)

# BASELINES & GREEDY
if args.evaluate_others:
    things_to_save = {}
    time_start = time.time()
    # BASELINES
    BASELINE_LAMBDAS = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 1]
    for baseline_lambda in BASELINE_LAMBDAS:
        results = eval_AC(env_name, critic_type='baseline', learner_type=args.learner_type, gamma=gamma, alpha=args.alpha, beta=args.beta, eta=args.eta, runtimes=args.runtimes, steps=args.steps, encoder=encoder, encoder_lambda=encoder_lambda, constant_lambda=baseline_lambda, kappa=args.kappa)
        exec("things_to_save[\'return_baseline_%g_mean\'] = np.nanmean(results, axis=0)" % (baseline_lambda * 1000)) # no dots in variable names for MATLAB
        exec("things_to_save[\'return_baseline_%g_std\'] = np.nanstd(results, axis=0)" % (baseline_lambda * 1000))
    # LAMBDA-GREEDY
    results = eval_AC(env_name, critic_type='greedy', learner_type=args.learner_type, gamma=gamma, alpha=args.alpha, beta=args.beta, eta=args.eta, runtimes=args.runtimes, steps=args.steps, encoder=encoder, encoder_lambda=encoder_lambda, constant_lambda=0, kappa=args.kappa)
    things_to_save['return_greedy_mean'], things_to_save['return_greedy_std'] = np.nanmean(results, axis=0), np.nanstd(results, axis=0)
    time_finish = time.time()
    print('time elapsed: %gs' % (time_finish - time_start))
    # SAVE
    filename = 'mountaincar_'
    if args.learner_type == 'togtd':
        filename = filename + 'a_%g_b_%g_y_%g_' % (args.alpha, args.beta, args.eta)
    else:
        filename = filename + 'a_%g_y_%g_' % (args.alpha, args.eta)
    filename = filename + 'e_%g_r_%d.mat' % (args.steps, args.runtimes)
    scipy.io.savemat(filename, things_to_save)

# MTA
if args.evaluate_MTA:
    things_to_save = {}
    time_start = time.time()
    results = eval_AC(env_name, critic_type='MTA', learner_type=args.learner_type, gamma=gamma, alpha=args.alpha, beta=args.beta, eta=args.eta, runtimes=args.runtimes, steps=args.steps, encoder=encoder, encoder_lambda=encoder_lambda, constant_lambda=0, kappa=args.kappa)
    if args.parametric_lambda:
        things_to_save['return_MTA_mean'], things_to_save['return_MTA_std'] = np.nanmean(results, axis=0), np.nanstd(results, axis=0)
    else:
        things_to_save['return_MTA_nonparam_mean'], things_to_save['return_MTA_nonparam_std'] = np.nanmean(results, axis=0), np.nanstd(results, axis=0)
    time_finish = time.time()
    print('time elapsed: %gs' % (time_finish - time_start))
    # SAVE
    filename = 'mountaincar_'
    if not args.parametric_lambda:
        filename = filename + 'nonparam_'
    if args.learner_type == 'togtd':
        filename = filename + 'a_%g_b_%g_y_%g_' % (args.alpha, args.beta, args.eta)
    else:
        filename = filename + 'a_%g_y_%g_' % (args.alpha, args.eta)
    filename = filename + 'k_%g_e_%g_r_%d.mat' % (args.kappa, args.steps, args.runtimes)
    scipy.io.savemat(filename, things_to_save)