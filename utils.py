import gym, numpy as np
from matplotlib import pyplot as plt
from RingWorld import RingWorldEnv
from joblib import Parallel, delayed

def mse(x, target, weight):
    diff = target - x.reshape(np.shape(target))
    return np.linalg.norm(np.multiply(diff, weight.reshape(np.shape(target))), 2) ** 2

def eval_method_with_variance_per_run(method, env, truth, var_truth, stat_dist, runtime, runtimes, episodes, target, behavior, gamma, Lambda, alpha, beta):
    result, var_result = np.zeros((1, episodes)), np.zeros((1, episodes))
    print('running %d of %d for %s' % (runtime + 1, runtimes, method.__name__))
    exp_trace, var_trace, dist_trace = method(env, episodes, target, behavior, Lambda, gamma = gamma, alpha = alpha, beta = beta)
    dist_trace = dist_trace / np.sum(dist_trace)
    for j in range(len(exp_trace)):
        result[0, j] = mse(exp_trace[j], truth, stat_dist)
        var_result[0, j] = mse(var_trace[j], var_truth, stat_dist)
    return (result, var_result)

def evaluate_estimate(estimate, expectation, variance, distribution, stat_type):
    if stat_type == 'expectation':
        return mse(estimate, expectation, distribution)
    elif stat_type == 'variance':
        return mse(estimate, variance, distribution)

def eval_method_with_variance(method, env, truth, var_truth, stat_dist, behavior, target, Lambda, gamma = lambda x: 0.95, alpha = 0.05, beta = 0.0001, runtimes=20, episodes=100000):
    results = Parallel(n_jobs = -1)(delayed(eval_method_with_variance_per_run)(method, env, truth, var_truth, stat_dist, runtime, runtimes, episodes, target, behavior, gamma, Lambda, alpha, beta) for runtime in range(runtimes))
    E = [entry[0] for entry in results]; V = [entry[1] for entry in results]
    E, V = np.concatenate(E, axis=0), np.concatenate(V, axis=0)
    return E, V

def plot_results(results, label=None):
    mean, std = np.nanmean(results, axis=0), np.nanstd(results, axis=0)
    plt.plot(mean, label = label)
    plt.fill_between(range(0, mean.shape[0]), mean - std, mean + std, alpha=.2)
    plt.legend()

def importance_sampling_ratio(target_policy, behavior_policy, s, a):
    return target_policy[s, a] / behavior_policy[s, a]

def decide(state_id, policy_matrix):
    dist = policy_matrix[state_id, :]
    action_id = np.random.choice(range(len(dist)), p = dist)
    return action_id

def onehot(observation, N):
    x = np.zeros(N)
    x[observation] = 1.0
    return x

def init_ring_world(N, p = [0.05, 0.95], seed=None):
    env = RingWorldEnv(N)
    np.random.seed(seed)
    env.seed(seed)
    target_policy = np.matlib.repmat(np.array([0.05, 0.95]).reshape(1, -1), env.observation_space.n, 1)
    behavior_policy = np.matlib.repmat(np.array([0.25, 0.75]).reshape(1, -1), env.observation_space.n, 1)
    return env, target_policy, behavior_policy

def gtd_step(r_next, gamma_next, gamma_curr, x_next, x_curr, w_curr, lambda_next, lambda_curr, rho_curr, e_prev, h_curr, alpha_curr, alpha_h_curr):
    delta_curr = r_next + gamma_next * np.dot(x_next, w_curr) - np.dot(x_curr, w_curr)
    e_curr = rho_curr * (gamma_curr * lambda_curr * e_prev + x_curr)
    w_next = w_curr + alpha_curr * (delta_curr * e_curr - gamma_next * (1 - lambda_next) * np.dot(h_curr, e_curr) * x_next)
    h_next = h_curr + alpha_h_curr * (delta_curr * e_curr - np.dot(x_curr, h_curr) * x_curr)
    return w_next, e_curr, h_next

class GTD_LEARNER():
    def __init__(self, env):
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.w_curr, self.w_prev = np.zeros(self.observation_space.n), np.zeros(self.observation_space.n)
        self.h_curr, self.h_prev = np.zeros(self.observation_space.n), np.zeros(self.observation_space.n)
        self.refresh()

    def learn(self, R_next, gamma_next, gamma_curr, x_next, x_curr, lambda_next, lambda_curr, rho_curr, alpha_curr, beta_curr):
        self.rho_curr = rho_curr
        self.w_next, self.e_grad_curr, self.h_next = gtd_step(R_next, gamma_next, gamma_curr, x_next, x_curr, self.w_curr, lambda_next, lambda_curr, rho_curr, self.e_grad_prev, self.h_curr, alpha_curr, beta_curr)
        pass
        
    def next(self):
        self.w_curr, self.w_prev = np.copy(self.w_next), np.copy(self.w_curr)
        self.e_grad_prev = np.copy(self.e_grad_curr)
        self.h_curr, self.h_prev = np.copy(self.h_next), np.copy(self.h_curr)
        self.rho_prev = np.copy(self.rho_curr)
        del self.w_next, self.e_grad_curr, self.h_next, self.rho_curr

    def refresh(self):
        self.e_grad_curr, self.e_grad_prev = np.zeros(self.observation_space.n), np.zeros(self.observation_space.n)
        self.rho_prev = 1

class LAMBDA():
    def __init__(self, env, lambda_type = 'constant', initial_value = 1):
        self.n = env.observation_space.n
        self.type = lambda_type
        if self.type == 'constant':
            self.w = initial_value
        else:
            if type(initial_value) is np.array or type(initial_value) is np.ndarray:
                self.w = initial_value.reshape(-1)
            else:
                self.w = np.ones(self.n)
    
    def value(self, x):
        if self.type == 'constant':
            if type(self.w) is float:
                l = self.w
            elif (type(self.w) is np.ndarray or type(self.w) is np.array) and np.size(self.w) > 1:
                l = np.dot(self.w, x.reshape(-1))
        else:
            l = np.dot(x.reshape(-1), self.w) # self.sigmoid(np.dot(x.reshape(-1), self.w))
        if l > 1:
            print('lambda value greater than 1')
            return 1
        elif l < 0:
            print('lambda value less than 0')
            return 0
        return l

    def gradient(self, x):
        return x.reshape(-1)# self.sigmoid(np.dot(x.reshape(-1), self.w), derivative=True) * x.reshape(-1)

    def gradient_descent(self, x, step_length):
        gradient = self.gradient(x)
        value_after = np.dot(x.reshape(-1), (self.w - step_length * gradient))
        if value_after > 1:
            pass # print('overflow of lambda rejected')
        elif value_after < 0:
            pass # print('underflow of lambda rejected')
        else:
            self.w -= step_length * gradient

    @staticmethod
    def sigmoid(x, derivative=False):
        sigm = 1. / (1. + np.exp(-x))
        if derivative:
            return sigm * (1. - sigm)
        return sigm
pass