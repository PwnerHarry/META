import numpy as np
from utils import *
import math



def islambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

def monte_carlo(env, episodes, target_policy, behavior_policy, gamma = lambda x: 0.95):
    nx = env.observation_space.n
    E, V, return_sums, return_counts, return_square_sums = [], [], np.zeros(nx), np.zeros(nx), np.zeros(nx)
    expectation, variance = np.zeros(nx), np.zeros(nx)
    for epi in range(episodes):
        print('Monte Carlo: episode %d / %d' % (epi + 1, episodes))
        state = env.reset()
        episode = []
        done = False
        while not done:
            action = behavior(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        states_in_episode = set([x[0] for x in episode])
        for state in states_in_episode:
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            G = sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occurence_idx:])])
            old_expectation = expectation[state]
            return_sums[state] += G
            return_counts[state] += 1.0
            new_expectation = return_sums[state] / return_counts[state]
            return_square_sums[state] += (G - old_expectation) * (G - new_expectation)
            expectation[state] = new_expectation
            variance[state] = return_square_sums[state] / return_counts[state]
        E.append(np.copy(expectation)); V.append(np.copy(variance))
    return E, V, return_counts

def gtd(env, behavior, target, nx, n_episodes, Lambda, gamma = lambda x: 0.95, alpha = 0.05, beta = 0.025):
    learner, w_trace = GTD_LEARNER(env), []
    for _ in range(n_episodes):
        lambda_next, lambda_curr = 1.0, 1.0
        observation, done = env.reset(), False
        x_curr = observation_to_phi(observation)
        learner.refresh()
        while not done:
            action = behavior(observation)
            rho_curr = rho(x_curr, action)
            observation, r_next, done, _ = env.step(action)
            x_next = observation_to_phi(observation)
            learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, lambda_next, lambda_curr, rho_curr, alpha, beta)
            learner.next()
            x_curr = x_next
        w_trace.append(np.copy(learner.w_curr))
    return w_trace, []



def greedy(env, behavior, target, observation_to_phi, nx, n_episodes, rho = lambda x, a: 1, gamma = lambda x: 0.95, alpha = 0.005, beta = 0.0025):
    first_moment_learner, second_moment_learner, value_learner = GTD_LEARNER(env), GTD_LEARNER(env), GTD_LEARNER(env)
    first_moment_learner.w_prev, first_moment_learner.w_curr = 100 * np.ones(env.observation_space.n), 100 * np.ones(env.observation_space.n)
    second_moment_learner.w_prev, second_moment_learner.w_curr = 0.01 * np.ones(env.observation_space.n), 0.01 * np.zeros(env.observation_space.n)
    w_trace, lambda_trace = [], []
    for _ in range(n_episodes):
        lambda_curr = 1.0
        observation, done = env.reset(), False
        start_observation = observation
        x_curr = observation_to_phi(start_observation)
        value_learner.refresh()
        first_moment_learner.refresh()
        second_moment_learner.refresh()
        while not done:
            if observation == start_observation:
                lambda_trace.append(lambda_curr)
            action = behavior(observation)
            rho_curr = rho(x_curr, action)
            observation, r_next, done, _ = env.step(action)
            x_next = observation_to_phi(observation)
            first_moment_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha, beta)
            second_moment_learner.learn((rho_curr * r_next) ** 2 + 2 * rho_curr ** 2 * gamma(x_next) * r_next * np.dot(x_next, first_moment_learner.w_curr), (rho_curr * gamma(x_next)) ** 2, gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, alpha, beta)
            errsq = (np.dot(x_next, first_moment_learner.w_next) - np.dot(x_next, value_learner.w_curr)) ** 2
            varg = max(0, np.dot(x_next, second_moment_learner.w_next) - np.dot(x_next, first_moment_learner.w_next) ** 2)
            lambda_next = errsq / (errsq + varg)
            if math.isnan(lambda_next):
                lambda_next = lambda_curr
            # print("errsq %.2e, varg %.2e, lambda_next %.2e" % (errsq, varg, lambda_next))
            value_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, lambda_next, lambda_curr, rho_curr, alpha, beta)
            first_moment_learner.next()
            second_moment_learner.next()
            value_learner.next()
            x_curr, lambda_curr = x_next, lambda_next
        w_trace.append(np.copy(value_learner.w_curr))
    return w_trace, lambda_trace

# TODO: to be simplified
def true_online_greedy(env, behavior, target, observation_to_x, nx, n_episodes, rho = lambda x, a: 1, gamma = lambda x: 0.95, alpha = 0.05, beta = 0.025):
    W = []
    lambda_curr = 1
    w_curr, h_curr = np.zeros(nx), np.zeros(nx)
    r_max = env.unwrapped.reward_range[-1]
    w_err, w_sq = (r_max / (1 - 0.95)) * np.ones(nx), np.zeros(nx)
    w_prev, w_curr = np.zeros(nx), np.zeros(nx)
    for _ in range(n_episodes):
        e_prev, e_grad_prev, e_h_prev = np.zeros(nx), np.zeros(nx), np.zeros(nx)
        e_bar_prev, z_bar_prev = np.zeros(nx), np.zeros(nx)

        lambda_curr, rho_prev = 1.0, 1.0

        observation = env.reset()
        x_curr = observation_to_x(observation)
        gamma_bar_curr = gamma(x_curr) ** 2

        done = False
        while not done:
            action = behavior(observation)
            observation, r_next, done, _ = env.step(action)
            x_next = observation_to_x(observation)
            rho_curr = rho(x_curr, action)
            # not finished: greedy step is not currently using true online GTD
            lambda_next, w_err, w_sq, e_bar_curr, z_bar_curr, gamma_bar_next = greedy_step(w_err, w_sq, w_curr, x_curr, x_next, r_next, rho_curr,
                                                                                  e_bar_prev, z_bar_prev, gamma(x_next), gamma(x_curr), gamma_bar_curr, lambda_curr, alpha_curr)
            # learn the value function using true online GTD
            w_next, e_curr, e_grad_curr, e_h_curr, h_next = true_online_gtd_step(r_next, gamma(x_next), gamma(x_curr),
                                                                                     x_next, x_curr, w_curr, w_prev,
                                                                                     lambda_curr, lambda_curr,
                                                                                     rho_curr, rho_prev,
                                                                                     e_prev, e_grad_prev, e_h_prev,
                                                                                     h_curr,
                                                                                     alpha, beta)
            
            x_curr, w_curr, w_prev = x_next, w_next, w_curr
            e_prev, e_grad_prev, e_h_prev = e_curr, e_grad_curr, e_h_curr
            h_curr = h_next
            rho_prev = rho_curr
            e_bar_prev, z_bar_prev = e_bar_curr, z_bar_curr
            lambda_curr = lambda_next
            gamma_bar_curr = gamma_bar_next
        W.append(w_curr)   
    return W, []

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