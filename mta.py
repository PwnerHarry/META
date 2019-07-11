import gym, numpy as np
from utils import *
from TOGTD import *
from TOTD import *

def MTA(env, episodes, target, behavior, evaluate, Lambda, encoder, learner_type = 'togtd', gamma = lambda x: 0.95, alpha = 0.05, beta = 0.05, kappa = 0.01):
    D = encoder(0).size
    value_trace = np.empty((episodes, 1)); value_trace[:] = np.nan
    if learner_type == 'togtd':
        MC_exp_learner, L_exp_learner, L_var_learner, value_learner = TOGTD_LEARNER(env, D), TOGTD_LEARNER(env, D), TOGTD_LEARNER(env, D), TOGTD_LEARNER(env, D)
    elif learner_type == 'totd':
        MC_exp_learner, L_exp_learner, L_var_learner, value_learner = TOTD_LEARNER(env, D), TOTD_LEARNER(env, D), TOTD_LEARNER(env, D), TOTD_LEARNER(env, D)
    for episode in range(episodes):
        o_curr, done = env.reset(), False
        x_curr = encoder(o_curr)
        log_rho_accu = 0 # use log accumulation of importance sampling ratio to increase stability
        MC_exp_learner.refresh(); L_exp_learner.refresh(); L_var_learner.refresh(); value_learner.refresh()
        value_trace[episode, 0] = evaluate(value_learner.w_curr, 'expectation')
        while not done:
            action = decide(o_curr, behavior)
            rho_curr = importance_sampling_ratio(target, behavior, o_curr, action)
            log_rho_accu += np.log(target[o_curr, action]) - np.log(behavior[o_curr, action])
            o_next, R_next, done, _ = env.step(action)
            x_next = encoder(o_next)
            if learner_type == 'togtd':
                MC_exp_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha, beta)
                L_exp_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, min(1.0, 1.1 * alpha), min(1.0, 1.1 * beta))
            elif learner_type == 'totd':
                MC_exp_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha)
                L_exp_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, min(1.0, 1.1 * alpha))
            delta_curr = R_next + float(not done) * gamma(x_next) * np.dot(x_next, value_learner.w_curr) - np.dot(x_curr, value_learner.w_curr)
            r_bar_next = delta_curr ** 2
            gamma_bar_next = (Lambda.value(x_next) * gamma(x_next)) ** 2
            if learner_type == 'togtd':
                L_var_learner.learn(r_bar_next, gamma_bar_next, 1, x_next, x_curr, 1, 1, rho_curr, alpha, beta)
            elif learner_type == 'totd':
                L_var_learner.learn(r_bar_next, gamma_bar_next, 1, x_next, x_curr, 1, 1, rho_curr, alpha)
            # SGD on meta-objective
            rho_acc = np.exp(log_rho_accu)
            v_next = np.dot(x_next, value_learner.w_curr)
            var_L_next, exp_L_next, exp_MC_next = np.dot(x_next, L_var_learner.w_curr), np.dot(x_next, L_exp_learner.w_curr), np.dot(x_next, MC_exp_learner.w_curr)
            coefficient = gamma(x_next) ** 2 * Lambda.value(x_next) * ((v_next - exp_L_next) ** 2 + var_L_next) + v_next * (exp_L_next + exp_MC_next) - v_next ** 2 - exp_L_next * exp_MC_next
            Lambda.GD(x_next, kappa * rho_acc * coefficient)
            # learn value
            if learner_type == 'togtd':
                value_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha, beta)
            elif learner_type == 'totd':
                value_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha)
            MC_exp_learner.next(); L_exp_learner.next(); L_var_learner.next(); value_learner.next()
            o_curr, x_curr = o_next, x_next
    return value_trace

def eval_MTA_per_run(env, runtime, runtimes, episodes, target, behavior, kappa, gamma, Lambda, alpha, beta, evaluate, encoder, learner_type):
    print('running %d of %d for MTA' % (runtime + 1, runtimes))
    value_trace = MTA(env, episodes, target, behavior, evaluate, Lambda, encoder, learner_type = 'togtd', gamma = gamma, alpha = alpha, beta = beta, kappa = kappa)
    return (value_trace, None)

def eval_MTA(env, behavior, target, kappa, gamma, alpha, beta, runtimes, episodes, evaluate, encoder, learner_type='togtd'):
    D = encoder(0).size
    LAMBDAS = []
    for runtime in range(runtimes):
        LAMBDAS.append(LAMBDA(env, np.ones(D) * D / env.observation_space.n, approximator = 'linear'))
    results = Parallel(n_jobs = -1)(delayed(eval_MTA_per_run)(env, runtime, runtimes, episodes, target, behavior, kappa, gamma, LAMBDAS[runtime], alpha, beta, evaluate, encoder, learner_type) for runtime in range(runtimes))
    value_traces = [entry[0] for entry in results]
    return np.concatenate(value_traces, axis = 1).T