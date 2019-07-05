import gym, gym.utils.seeding, numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

def softmax(x):
    # a numerically stable softmax!
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def jacobian_softmax(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.matmul(s, s.T) # $J = I - \bm{s}\bm{s}^{T}$

class LAMBDA():# state-based parametric lambda
    def __init__(self, env, initial_value, approximator = 'constant'):
        self.n = env.observation_space.n
        self.approximator = approximator
        if approximator == 'constant':
            self.w = initial_value
        elif approximator == 'linear':
            self.w = initial_value.reshape(-1)
        elif approximator == 'tabular':
            self.w = initial_value.reshape(-1)
        elif approximator == 'NN':
            pass # Neural Network approximator to be implemented using PyTorch

    def value(self, x):
        if self.approximator == 'constant':
            l = self.w
        elif self.approximator == 'tabular':
            l = self.w[x]
        elif self.approximator == 'linear':
            l = np.dot(x.reshape(-1), self.w)
        elif self.approximator == 'NN':
            pass # not implemented
        if l > 1:
            # print('lambda value greater than 1, truncated to 1')
            return 1
        elif l < 0:
            # print('lambda value less than 0, truncated to 0')
            return 0
        return l

    def gradient(self, x):
        if self.approximator == 'linear':
            return x.reshape(-1)

    def gradient_descent(self, x, step_length):
        gradient = self.gradient(x)
        value_after = np.dot(x.reshape(-1), (self.w - step_length * gradient))
        if value_after >= 0 and value_after <= 1:
            self.w -= step_length * gradient

class RingWorldEnv(gym.Env):
    def __init__(self, N):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(N)
        self.seed()
        P = {}
        for s in range(self.observation_space.n):
            small_dict = {}
            for a in [0, 1]:
                increment = -1 if a == 0 else 1
                if s == 0 or s == self.observation_space.n - 1:
                    entry = [(1.0, s, 0, True)]
                elif s + increment == self.observation_space.n - 1:
                    entry = [(1.0, self.observation_space.n - 1, 1, True)]
                elif s + increment == 0:
                    entry = [(1.0, 0, -1, True)]
                else:
                    entry = [(1.0, s + increment, 0, False)]
                small_dict[a] = entry
            P[s] = small_dict
        self.unwrapped.P = P
        self.unwrapped.reward_range = (-1, 1)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        increment = -1 if action == 0 else 1
        self.state = (self.state + increment) % self.observation_space.n
        if self.state == 0:
            return self.state, -1, True, {}
        elif self.state == self.observation_space.n - 1:
            return self.state, 1, True, {}
        return self.state, 0, False, {}

    def reset(self):
        self.state = int(self.observation_space.n / 2)
        self.episode_time = 0
        return self.state

def mse(estimate, target, weight):
    diff = target - estimate.reshape(np.shape(target))
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

def evaluate_estimate(weight, expectation, variance, distribution, stat_type, state_set_matrix):
    # place the state representations row by row in the state_set_matrix
    estimate = np.dot(state_set_matrix, weight).reshape(-1)
    if stat_type == 'expectation':
        return mse(estimate, expectation, distribution)
    elif stat_type == 'variance':
        return mse(estimate, variance, distribution)

def get_state_set_matrix(env, encoder):
    state_set_matrix = np.zeros((env.observation_space.n, np.size(encoder(0))))
    for s in range(env.observation_space.n):
        state_set_matrix[s, :] = encoder(s).reshape(1, -1)
    return state_set_matrix

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

# ENCODING METHODS
def onehot(observation, N):
    x = np.zeros(N)
    x[observation] = 1
    return x

def index2plane(s, n):
    feature = np.zeros(2 * n)
    feature[s // n] = 1; feature[n + s % n] = 1
    return feature

# def gtd_step(r_next, gamma_next, gamma_curr, x_next, x_curr, w_curr, lambda_next, lambda_curr, rho_curr, e_prev, h_curr, alpha_curr, alpha_h_curr):
#     delta_curr = r_next + gamma_next * np.dot(x_next, w_curr) - np.dot(x_curr, w_curr)
#     e_curr = rho_curr * (gamma_curr * lambda_curr * e_prev + x_curr)
#     w_next = w_curr + alpha_curr * (delta_curr * e_curr - gamma_next * (1 - lambda_next) * np.dot(h_curr, e_curr) * x_next)
#     h_next = h_curr + alpha_h_curr * (delta_curr * e_curr - np.dot(x_curr, h_curr) * x_curr)
#     return w_next, e_curr, h_next

# class GTD_LEARNER():
#     def __init__(self, env):
#         self.observation_space, self.action_space = env.observation_space, env.action_space
#         self.w_curr, self.w_prev = np.zeros(self.observation_space.n), np.zeros(self.observation_space.n)
#         self.h_curr, self.h_prev = np.zeros(self.observation_space.n), np.zeros(self.observation_space.n)
#         self.refresh()

#     def learn(self, R_next, gamma_next, gamma_curr, x_next, x_curr, lambda_next, lambda_curr, rho_curr, alpha_curr, beta_curr):
#         self.rho_curr = rho_curr
#         self.w_next, self.e_grad_curr, self.h_next = gtd_step(R_next, gamma_next, gamma_curr, x_next, x_curr, self.w_curr, lambda_next, lambda_curr, rho_curr, self.e_grad_prev, self.h_curr, alpha_curr, beta_curr)
#         pass
        
#     def next(self):
#         self.w_curr, self.w_prev = np.copy(self.w_next), np.copy(self.w_curr)
#         self.e_grad_prev = np.copy(self.e_grad_curr)
#         self.h_curr, self.h_prev = np.copy(self.h_next), np.copy(self.h_curr)
#         self.rho_prev = np.copy(self.rho_curr)
#         del self.w_next, self.e_grad_curr, self.h_next, self.rho_curr

#     def refresh(self):
#         self.e_grad_curr, self.e_grad_prev = np.zeros(self.observation_space.n), np.zeros(self.observation_space.n)
#         self.rho_prev = 1

def dynamic_programming(env, policy, gamma = lambda x: 0.95):
    TABLE = env.unwrapped.P # (s, (a, (p, s', reward, done)), ..., )
    # p(s, a, s') and r(s, a)
    P, R = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n)), np.zeros((env.observation_space.n, env.action_space.n))
    # terminal states
    terminal_states = []
    for s in range(env.observation_space.n)[1: -1]:
        for a in range(env.action_space.n):
            RELATED = TABLE[s][a]
            for entry in RELATED:
                if entry[-1] == True:
                    terminal_states.append(entry[1])
    for s in terminal_states:
        for a in range(env.action_space.n):
            P[s, a, s], R[s, a] = 1, 0
    # non-terminal states
    for s in list(set(range(env.observation_space.n)) - set(terminal_states)):
        for a in range(env.action_space.n):
            RELATED = TABLE[s][a]
            for entry in RELATED:
                R[s, a] += entry[0] * entry[2]
                P[s, a, entry[1]] = entry[0]
    r_pi = np.zeros((env.observation_space.n, 1))
    P_pi = np.zeros((env.observation_space.n, env.observation_space.n))
    for s in range(env.observation_space.n):
        r_pi[s] = np.dot(policy[s, :], R[s, :])
        for s_prime in range(env.observation_space.n):
            P_pi[s, s_prime] = np.dot(policy[s, :], P[s, :, s_prime])
    if not islambda(gamma):
        gamma = lambda x: gamma
    # for generalized \Gamma setting, one gamma for one state (or observation or feature)
    GAMMA = np.zeros((env.observation_space.n, env.observation_space.n))
    for i in range(env.observation_space.n):
        GAMMA[i, i] = gamma(i)

    expectation = np.linalg.solve(np.eye(env.observation_space.n) - np.matmul(P_pi, GAMMA), r_pi)
    return expectation, P_pi

def iterative_policy_evaluation(env, policy, gamma = lambda x: 0.95, start_dist = None):
    if not start_dist:
        # For legacy reasons, if start dist is not specified always start in the middle state.
        start_dist = np.zeros(env.observation_space.n)
        start_dist[int(env.observation_space.n / 2)] = 1.0

    TABLE = env.unwrapped.P # (s, (a, (p, s', reward, done)), ..., )
    # p(s, a, s') and r(s, a, s')
    P = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    R = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    # terminal states
    terminal_states = []
    for s in range(env.observation_space.n)[1: -1]:
        for a in range(env.action_space.n):
            RELATED = TABLE[s][a]
            for entry in RELATED:
                if entry[-1] == True:
                    terminal_states.append(entry[1])
    for s in terminal_states:
        for a in range(env.action_space.n):
            P[s, a, s] = 1
    # non-terminal states
    for s in list(set(range(env.observation_space.n)) - set(terminal_states)):
        for a in range(env.action_space.n):
            RELATED = TABLE[s][a]
            for entry in RELATED:
                R[s, a, entry[1]] = entry[2]
                P[s, a, entry[1]] = entry[0]
    
    theta = 1e-10
    delta = theta
    j = np.zeros(env.observation_space.n)
    while delta >= theta:
        delta = 0.0
        for s in range(env.observation_space.n):
            old_value = j[s]
            new_value = 0.0
            for s_prime in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    new_value += policy[s, a] * P[s, a, s_prime] * (R[s, a, s_prime] + gamma(s_prime) * j[s_prime])
            delta = max(delta, np.abs(new_value - old_value))
            j[s] = new_value
    
    theta = 1e-10
    delta = theta
    v = np.zeros(env.observation_space.n)
    while delta >= theta:
        delta = 0.0
        for s in range(env.observation_space.n):
            old_value = v[s]
            r_hat = 0.0
            r = 0.0
            j_hat = 0.0
            v_hat = 0.0
            # new_value = - j[s] ** 2
            for s_prime in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    tp = policy[s, a] * P[s, a, s_prime]
                    # new_value += tp * ((R[s, a, s_prime] + gamma(s_prime) * j[s_prime]) ** 2 + (gamma(s_prime) ** 2) * v[s_prime])
                    r_hat += tp * (R[s, a, s_prime] ** 2)
                    j_hat += tp * (R[s, a, s_prime] * gamma(s_prime) * j[s_prime])
                    v_hat += tp * (gamma(s_prime) ** 2) * v[s_prime]
            new_value = r_hat + 2 * j_hat + v_hat
            delta = max(delta, np.abs(new_value - old_value))
            v[s] = new_value

    P_pi = np.zeros((env.observation_space.n, env.observation_space.n))
    for s in range(env.observation_space.n):
        for s_prime in range(env.observation_space.n):
            P_pi[s, s_prime] = np.dot(policy[s, :], P[s, :, s_prime])

    return j, (v - np.square(j)), state_distribution(P_pi, start_dist)

def state_distribution(P, start_dist):
    """
    P:          stochastic matrix of transition
    start_dist: distribution of the starting state
    """
    n = np.shape(P)[0]
    state_dist = np.zeros((1, n))
    absorb_states = []
    for i in range(n):
        if P[i, i] == 1:
            absorb_states.append(i)
    start_dist = start_dist.reshape((1, n))
    state_dist += start_dist
    state_dist[0, absorb_states] = 0
    next_dict = np.sum(np.matmul(start_dist, P), axis = 0).reshape((1, n))
    next_dict_norm = np.linalg.norm(next_dict.reshape(-1), 1)
    while next_dict_norm > 1e-14:
        state_dist += next_dict
        next_dict[0, absorb_states] = 0
        next_dict = np.sum(np.matmul(next_dict, P), axis = 0).reshape((1, n))
        next_dict_norm = np.linalg.norm(next_dict.reshape(-1), 1)
    state_dist = state_dist.reshape(-1)
    state_dist = state_dist / np.sum(state_dist)
    return state_dist

pass