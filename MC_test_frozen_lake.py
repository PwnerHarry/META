from utils import *
from methods import *
from joblib import Parallel, delayed
from MC import *
from true_online_GTD import *
import numpy.matlib
from frozen_lake import FrozenLakeEnv


unit = 1.0

# experiment Preparation
env = FrozenLakeEnv(None, '4x4', True, unit)
N = env.observation_space.n
mc_episodes = int(1e7)
runtimes, episodes, gamma = 8, 1000, lambda x: 0.95

target_policy = np.matlib.repmat(np.ones((1, env.action_space.n)) / env.action_space.n, env.observation_space.n, 1)
# behavior_policy = np.matlib.repmat(np.ones((1, env.action_space.n)) / env.action_space.n, env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
filename = 'frozen_lake_ground_truths_uniform_%d.npz' % mc_episodes
try:
    loaded = np.load(filename)
    true_expectation, true_variance, stationary_dist = loaded['true_expectation'], loaded['true_variance'], loaded['stationary_dist']
except FileNotFoundError:
    E, V, return_counts = MC(env, mc_episodes, target_policy, target_policy, None, gamma)
    stationary_dist = return_counts / np.sum(return_counts)
    true_expectation, true_variance = E[-1], V[-1]
    np.savez(filename, true_expectation = true_expectation, true_variance = true_variance, stationary_dist = stationary_dist)

true_expectation = true_expectation * unit
true_variance = true_variance * (unit ** 2)

# test both on-policy and off-policy
# Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = np.ones(N))
# off_togtd_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, behavior_policy, target_policy, Lambda, gamma = lambda x: 0.95, alpha=0.05, beta=0.05, runtimes=runtimes, episodes=episodes)

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = np.ones(N)*0.0)
on_togtd_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, target_policy, target_policy, Lambda, gamma = lambda x: 0.95, alpha=0.05, beta=0.05, runtimes=runtimes, episodes=episodes)
plot_results(on_togtd_results, label='on-policy true online GTD(0)')

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = np.ones(N)*0.5)
on_togtd_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, target_policy, target_policy, Lambda, gamma = lambda x: 0.95, alpha=0.05, beta=0.05, runtimes=runtimes, episodes=episodes)
plot_results(on_togtd_results, label='on-policy true online GTD(.5)')

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = np.ones(N))
on_togtd_results = eval_method(true_online_gtd, env, true_expectation, stationary_dist, target_policy, target_policy, Lambda, gamma = lambda x: 0.95, alpha=0.05, beta=0.05, runtimes=runtimes, episodes=episodes)

# plot_results(off_togtd_results, label='off-policy true online GTD(1)')
plot_results(on_togtd_results, label='on-policy true online GTD(1)')
plt.yscale('log'); plt.xscale('symlog')
plt.show()
pass