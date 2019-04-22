import matplotlib as mpl
mpl.use('Agg')
from utils import *
from methods import *
from greedy import *
from mta import *
import numpy as np
import numpy.matlib as npm
import warnings
import scipy.io

# experiment Preparation
N = 11; env, runtimes, episodes, gamma = RingWorldEnv(N, unit = 1), int(80), int(1e3), lambda x: 0.95
alpha, beta = 0.05, 0.01
target_policy = npm.repmat(np.array([0.5, 0.5]).reshape(1, -1), env.observation_space.n, 1)
behavior_policy = npm.repmat(np.array([0.5, 0.5]).reshape(1, -1), env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
true_expectation, true_variance, stationary_dist = iterative_policy_evaluation(env, target_policy, gamma=gamma)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 1.0 * np.ones(N))
off_togtd_10_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0.8 * np.ones(N))
off_togtd_08_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0.6 * np.ones(N))
off_togtd_06_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0.4 * np.ones(N))
off_togtd_04_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0.2 * np.ones(N))
off_togtd_02_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = 0 * np.ones(N))
off_togtd_00_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

_, error_var_greedy, direct_greedy_results, lambda_trace_greedy = eval_greedy(env, true_expectation, true_variance, stationary_dist, behavior_policy, target_policy, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)
warnings.filterwarnings('error')
_, error_L_exp, error_L_var, error_value, lambda_trace_mta = eval_MTA(env, true_expectation, true_variance, stationary_dist, behavior_policy, target_policy, gamma = gamma, alpha=alpha, beta=beta, runtimes=runtimes, episodes=episodes)

# plt.ioff()
# fig = plt.figure()
# plot_results(error_value, label='MTA')
# plot_results(direct_greedy_results, label='direct greedy')
# plot_results(off_togtd_10_results, label='GTD(1)')
# plot_results(off_togtd_08_results, label='GTD(0.8)')
# plot_results(off_togtd_06_results, label='GTD(0.6)')
# plot_results(off_togtd_04_results, label='GTD(0.4)')
# plot_results(off_togtd_02_results, label='GTD(0.2)')
# plot_results(off_togtd_00_results, label='GTD(0)')
# plt.xscale('log'); plt.yscale('log')
# plt.savefig('error.pdf')
# plt.close(fig)

# fig = plt.figure()
# plot_results(lambda_trace, label='Lambda(5)')
# plt.savefig('lambda.pdf')
# plt.close(fig)

# fig = plt.figure()
# plot_results(error_L_var, label='error_L_var')
# plt.xscale('log'); plt.yscale('log')
# plt.savefig('variance_Lambda_return.pdf')
# plt.close(fig)

things_to_save = {}

things_to_save['error_value'] = error_value
things_to_save['error_L_exp'] = error_L_exp
things_to_save['error_L_var'] = error_L_var
things_to_save['error_var_greedy'] = error_var_greedy
things_to_save['lambda_trace_mta'] = lambda_trace_mta
things_to_save['lambda_trace_greedy'] = lambda_trace_greedy
things_to_save['off_togtd_00_results'] = off_togtd_00_results
things_to_save['off_togtd_02_results'] = off_togtd_02_results
things_to_save['off_togtd_04_results'] = off_togtd_04_results
things_to_save['off_togtd_06_results'] = off_togtd_06_results
things_to_save['off_togtd_08_results'] = off_togtd_08_results
things_to_save['off_togtd_10_results'] = off_togtd_10_results
things_to_save['direct_greedy_results'] = direct_greedy_results

filename = 'mta_N_%s_behavior_%g_target_%g_episodes_%g' % (N, behavior_policy[0, 0], target_policy[0, 0], episodes)
scipy.io.savemat(filename, things_to_save)

pass