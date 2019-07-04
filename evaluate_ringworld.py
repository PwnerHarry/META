import time, argparse, warnings, scipy.io, numpy as np, numpy.matlib as npm
from greedy import *
from mta import *
from TOGTD import *
from TOTD import *
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--alpha', type=float, default=0.0001, help='')
parser.add_argument('--beta', type=float, default=0.00001, help='')
parser.add_argument('--kappa', type=float, default=0.01, help='')
parser.add_argument('--gamma', type=float, default=0.95, help='')
parser.add_argument('--episodes', type=int, default=10000, help='')
parser.add_argument('--runtimes', type=int, default=8, help='')
parser.add_argument('--N', type=int, default=11, help='')
parser.add_argument('--target', type=float, default=0.05, help='')
parser.add_argument('--behavior', type=float, default=0.05, help='')
parser.add_argument('--learner_type', type=str, default='totd', help='')
parser.add_argument('--evaluate_baselines', type=int, default=1, help='')
parser.add_argument('--evaluate_greedy', type=int, default=1, help='')
parser.add_argument('--evaluate_MTA', type=int, default=1, help='')
args = parser.parse_args()

# experiment Preparation
env, gamma, encoder = RingWorldEnv(args.N), lambda x: args.gamma, lambda s: onehot(s, env.observation_space.n)
target_policy, behavior_policy = npm.repmat(np.array([args.target, 1 - args.target]).reshape(1, -1), env.observation_space.n, 1), npm.repmat(np.array([args.behavior, 1 - args.behavior]).reshape(1, -1), env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
true_expectation, true_variance, stationary_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)
evaluate = lambda estimate, stat_type: evaluate_estimate(estimate, true_expectation, true_variance, stationary_dist, stat_type, get_state_set_matrix(env, encoder))
things_to_save = {}

time_start = time.time()
# BASELINES
if args.evaluate_baselines:
    BASELINE_LAMBDAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for baseline_lambda in BASELINE_LAMBDAS:
        Lambda = LAMBDA(env, baseline_lambda, approximator='constant')
        if args.learner_type == 'totd':
            results = eval_totd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=args.alpha, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder)
        elif args.learner_type == 'togtd':
            results = eval_togtd(env, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=args.alpha, beta=args.beta, runtimes=args.runtimes, episodes=args.episodes, evaluate=evaluate, encoder=encoder)
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
time_finish = time.time()
print('time elapsed: %gs' % (time_finish - time_start))

# SAVE
if args.learner_type == "togtd":
    filename = 'ringworld_%s_behavior_%g_target_%g_a_%g_b_%g_k_%g_e_%g_r_%d' % (args.learner_type, behavior_policy[0, 0], target_policy[0, 0], args.alpha, args.beta, args.kappa, args.episodes, args.runtimes)
elif args.learner_type == "totd":
    filename = 'ringworld_%s_behavior_%g_target_%g_a_%g_k_%g_e_%g_r_%d' % (args.learner_type, behavior_policy[0, 0], target_policy[0, 0], args.alpha, args.kappa, args.episodes, args.runtimes)
scipy.io.savemat(filename, things_to_save)
pass