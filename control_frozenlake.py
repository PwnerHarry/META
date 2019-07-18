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
parser.add_argument('--eta', type=float, default=0.01, help='')
parser.add_argument('--gamma', type=float, default=1, help='')
parser.add_argument('--kappa', type=float, default=0.001, help='')
parser.add_argument('--episodes', type=int, default=100000, help='')
parser.add_argument('--runtimes', type=int, default=16, help='')
parser.add_argument('--learner_type', type=str, default='togtd', help='')
parser.add_argument('--critic_type', type=str, default='baseline', help='')
parser.add_argument('--constant_lambda', type=float, default=1.0, help='')
parser.add_argument('--evaluate_baselines', type=int, default=1, help='')
parser.add_argument('--evaluate_greedy', type=int, default=1, help='')
parser.add_argument('--evaluate_MTA', type=int, default=1, help='')
args = parser.parse_args()
if args.beta == 0:
    args.beta = 0.01 * args.alpha

# Experiment Preparation
env = gym.make('FrozenLake-v0')
gamma, encoder = lambda x: args.gamma, lambda s: tilecoding4x4(s)

things_to_save = {}
time_start = time.time()

returns = eval_AC(env, critic_type=args.critic_type, learner_type=args.learner_type, gamma=gamma, alpha=args.alpha, beta=args.beta, eta=args.eta, runtimes=args.runtimes, episodes=args.episodes, encoder=encoder, constant_lambda=args.constant_lambda, kappa=args.kappa)
things_to_save['return_mean'], things_to_save['return_std'] = np.nanmean(returns, axis=0), np.nanstd(returns, axis=0)

# BASELINES
# if args.evaluate_baselines:
#     BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
#     for baseline_lambda in BASELINE_LAMBDAS:
#         Lambda = LAMBDA(env, baseline_lambda, approximator='constant')
#         results = eval_togtd(env, behavior, target, Lambda, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder)
#         exec("things_to_save[\'error_value_%s_%g_mean\'] = np.nanmean(results, axis=0)" % (args.learner_type, baseline_lambda * 100)) # no dots in variable names for MATLAB
#         exec("things_to_save[\'error_value_%s_%g_std\'] = np.nanstd(results, axis=0)" % (args.learner_type, baseline_lambda * 100))

# LAMBDA-GREEDY
# if args.evaluate_greedy:
#     error_value_greedy = eval_greedy(env, behavior, target, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
#     things_to_save['error_value_greedy_mean'], things_to_save['error_value_greedy_std'] = np.nanmean(error_value_greedy, axis=0), np.nanstd(error_value_greedy, axis=0)

# MTA
# if args.evaluate_MTA:
#     error_value_mta = eval_MTA(env, behavior, target, kappa=args.kappa, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder, learner_type=args.learner_type)
#     things_to_save['error_value_mta_mean'], things_to_save['error_value_mta_std'] = np.nanmean(error_value_mta, axis=0), np.nanstd(error_value_mta, axis=0)

time_finish = time.time()
print('time elapsed: %gs' % (time_finish - time_start))

# SAVE
if args.evaluate_MTA:
    filename = 'frozenlake_AC_a_%g_b_%g_y_%g_k_%g_e_%g_r_%d.mat' % (args.alpha, args.beta, args.eta, args.kappa, args.episodes, args.runtimes)
else:
    filename = 'frozenlake_AC_a_%g_b_%g_y_%g_e_%g_r_%d.mat' % (args.alpha, args.beta, args.eta, args.episodes, args.runtimes)
scipy.io.savemat(filename, things_to_save)