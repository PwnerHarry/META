import gym, numpy as np
from utils import *
from TOGTD import *
from TOTD import *

def greedy(env, episodes, target, behavior, evaluate, Lambda, encoder, learner_type, gamma = lambda x: 0.95, alpha = 0.05, beta = 0.05):
    D = encoder(0).size
    if learner_type == 'togtd':
        LEARNER = TOGTD_LEARNER
    elif learner_type == 'totd':
        LEARNER = TOTD_LEARNER
    MC_exp_learner, MC_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D)
    MC_var_learner.w_prev, MC_var_learner.w_curr = np.zeros(D), np.zeros(D)
    value_trace = np.empty((episodes, 1)); value_trace[:] = np.nan
    for episode in range(episodes):
        o_curr, done = env.reset(), False
        x_curr = encoder(o_curr)
        value_learner.refresh(); MC_exp_learner.refresh(); MC_var_learner.refresh()
        value_trace[episode, 0] = evaluate(value_learner.w_curr, 'expectation')
        while not done:
            action = decide(o_curr, behavior)
            rho_curr = importance_sampling_ratio(target, behavior, o_curr, action)
            o_next, R_next, done, _ = env.step(action)
            x_next = encoder(o_next)
            if learner_type == 'togtd':
                MC_exp_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha, beta)
            elif learner_type == 'totd':
                MC_exp_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha)
            delta_curr = R_next + float(not done) * gamma(x_next) * np.dot(x_next, value_learner.w_curr) - np.dot(x_curr, value_learner.w_curr)
            r_bar_next = delta_curr ** 2
            gamma_bar_next = (rho_curr * gamma(x_next)) ** 2
            if learner_type == 'togtd':
                MC_var_learner.learn(r_bar_next, gamma_bar_next, 1, x_next, x_curr, 1, 1, 1, alpha, beta)
            elif learner_type == 'totd':
                MC_var_learner.learn(r_bar_next, gamma_bar_next, 1, x_next, x_curr, 1, 1, 1, alpha)
            errsq = (np.dot(x_next, MC_exp_learner.w_next) - np.dot(x_next, value_learner.w_curr)) ** 2
            varg = max(0, np.dot(x_next, MC_var_learner.w_next))
            if errsq + varg > 0:
                Lambda.w[o_next] = errsq / (errsq + varg)
            else:
                Lambda.w[o_next] = 1
            if learner_type == 'togtd':
                value_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.w[o_next], Lambda.w[o_curr], rho_curr, alpha, beta)
            elif learner_type == 'totd':
                value_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.w[o_next], Lambda.w[o_curr], rho_curr, alpha)
            MC_exp_learner.next(); MC_var_learner.next(); value_learner.next()
            o_curr, x_curr = o_next, x_next
    return value_trace

def eval_greedy_per_run(env, runtime, runtimes, episodes, target, behavior, encoder, gamma, Lambda, alpha, beta, evaluate, learner_type):
    print('running %d of %d for greedy' % (runtime + 1, runtimes))
    value_trace = greedy(env, episodes, target, behavior, evaluate=evaluate, Lambda=Lambda, encoder=encoder,gamma=gamma, alpha=alpha, beta=beta, learner_type=learner_type)
    return (value_trace, None)

def eval_greedy(env, behavior, target, evaluate, gamma, alpha, beta, runtimes, episodes, encoder, learner_type='togtd'):
    LAMBDAS = []
    for runtime in range(runtimes):
        LAMBDAS.append(LAMBDA(env, approximator='tabular', initial_value=np.ones(env.observation_space.n)))
    results = Parallel(n_jobs = -1)(delayed(eval_greedy_per_run)(env, runtime, runtimes, episodes, target, behavior, encoder, gamma, LAMBDAS[runtime], alpha, beta, evaluate, learner_type=learner_type) for runtime in range(runtimes))
    value_traces = [entry[0] for entry in results]
    return np.concatenate(value_traces, axis = 1).T