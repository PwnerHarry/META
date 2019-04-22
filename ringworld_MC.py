from utils import *
from methods import *
from joblib import Parallel, delayed
from MC import *
import numpy.matlib

# experiment Preparation
N = 11; env, runtimes, episodes, gamma = RingWorldEnv(N), int(50), int(2500), lambda x: 0.95
target_policy = np.matlib.repmat(np.array([0.5, 0.5]).reshape(1, -1), env.observation_space.n, 1)
behavior_policy = np.matlib.repmat(np.array([0.5, 0.5]).reshape(1, -1), env.observation_space.n, 1)

# get ground truth expectation, variance and stationary distribution
filename = 'ground_truths_0.5_0.5_%d.npz' % N
try:
    loaded = np.load(filename)
    true_expectation, true_variance, stationary_dist = loaded['true_expectation'], loaded['true_variance'], loaded['stationary_dist']
except FileNotFoundError:
    E, V, return_counts = monte_carlo(env, decide, decide, lambda x: onehot(x, N), N, int(1e6), gamma = 0.99)
    stationary_dist = return_counts / np.sum(return_counts)
    true_expectation, true_variance = E[-1], V[-1]
    np.savez(filename, true_expectation = true_expectation, true_variance = true_variance, stationary_dist = stationary_dist)


j, v, s = iterative_policy_evaluation(env, target_policy, gamma=gamma)
print('Iterative policy evaluation')

Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = np.ones(N))
mc_j, mc_v, mc_counts = MC(env, 10000, target_policy, target_policy, Lambda, gamma)

# test both on-policy and off-policy
Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = np.ones(N))
off_mc_results, off_mc_var_results = eval_method_with_variance(MC, env, true_expectation, true_variance, stationary_dist, behavior_policy, target_policy, Lambda, gamma=gamma, alpha=0.05, beta=0.05, runtimes=runtimes, episodes=episodes)
Lambda = LAMBDA(env, lambda_type = 'constant', initial_value = np.ones(N))
on_mc_results, on_mc_var_results = eval_method_with_variance(MC, env, true_expectation, true_variance, stationary_dist, target_policy, target_policy, Lambda, gamma=gamma, alpha=0.05, beta=0.05, runtimes=runtimes, episodes=episodes)

plot_results(off_mc_results, label='off-policy MC')
plot_results(on_mc_results, label='on-policy MC')
plt.yscale('log')

plt.figure()
plot_results(off_mc_var_results, label='off-policy MC variance')
plot_results(on_mc_var_results, label='on-policy MC variance')
plt.yscale('log')
plt.show()
pass