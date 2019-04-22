import gym, numpy as np
from utils import *
from methods import *
from true_online_GTD import *

def eval_MTA_per_run(env, runtime, runtimes, episodes, target, behavior, kappa, gamma, Lambda, alpha, beta):
    print('running %d of %d for MTA' % (runtime + 1, runtimes))
    MC_exp_trace, L_exp_trace, L_var_trace, value_trace, lambda_trace = MTA(env, episodes, target, behavior, Lambda, kappa = 0.1, gamma = lambda x: 0.95, alpha = 0.05, beta = 0.0001, diagnose = False)
    return (MC_exp_trace, L_exp_trace, L_var_trace, value_trace, lambda_trace)

def eval_MTA(env, expectation, variance, stat_dist, behavior, target, kappa = 0.1, gamma = lambda x: 0.95, alpha=0.05, beta=0.05, runtimes=20, episodes=int(1e5)):
    LAMBDAS = []
    for runtime in range(runtimes):
        LAMBDAS.append(LAMBDA(env, lambda_type = 'variable'))
    results = Parallel(n_jobs = -1)(delayed(eval_MTA_per_run)(env, runtime, runtimes, episodes, target, behavior, kappa, gamma, LAMBDAS[runtime], alpha, beta) for runtime in range(runtimes))
    MC_exp_traces = [entry[0] for entry in results]
    L_exp_traces = [entry[1] for entry in results]
    L_var_traces = [entry[2] for entry in results]
    value_traces = [entry[3] for entry in results]
    lambda_trace = [entry[4] for entry in results]

    error_MC_exp = np.zeros((runtimes, episodes))
    for runtime in range(runtimes):
        w_trace = MC_exp_traces[runtime]
        for j in range(len(w_trace)):
            error_MC_exp[runtime, j] = mse(w_trace[j], expectation, stat_dist)
    
    error_L_exp = np.zeros((runtimes, episodes))
    for runtime in range(runtimes):
        w_trace = L_exp_traces[runtime]
        for j in range(len(w_trace)):
            error_L_exp[runtime, j] = mse(w_trace[j], expectation, stat_dist)
    
    error_L_var = np.zeros((runtimes, episodes))
    for runtime in range(runtimes):
        w_trace = L_var_traces[runtime]
        for j in range(len(w_trace)):
            error_L_var[runtime, j] = mse(w_trace[j], variance, stat_dist)
            
            
    error_value = np.zeros((runtimes, episodes))
    for runtime in range(runtimes):
        w_trace = value_traces[runtime]
        for j in range(len(w_trace)):
            error_value[runtime, j] = mse(w_trace[j], expectation, stat_dist)
    
    return error_MC_exp, error_L_exp, error_L_var, error_value, np.concatenate(lambda_trace, axis = 1).T

def MTA(env, episodes, target, behavior, Lambda, kappa = 0.1, gamma = lambda x: 0.95, alpha = 0.05, beta = 0.05, diagnose = False):
    N = env.observation_space.n
    lambda_trace = np.zeros((episodes, 1))
    lambda_trace[:] = np.nan
    MC_exp_learner, L_exp_learner, L_var_learner, value_learner = TRUE_ONLINE_GTD_LEARNER(env), TRUE_ONLINE_GTD_LEARNER(env), TRUE_ONLINE_GTD_LEARNER(env), TRUE_ONLINE_GTD_LEARNER(env)
    MC_exp_trace, L_exp_trace, L_var_trace, value_trace = [], [], [], []
    for epi in range(episodes):
        s_curr, done = env.reset(), False
        starting_state = s_curr
        x_curr = onehot(s_curr, N)
        rho_accu_nume, rho_accu_deno = 1.0, 1.0
        MC_exp_learner.refresh(); L_exp_learner.refresh(); L_var_learner.refresh(); value_learner.refresh()
        # MC_exp_trace, L_exp_trace, L_var_trace, value_trace
        MC_exp_trace.append(np.copy(MC_exp_learner.w_curr))
        L_exp_trace.append(np.copy(L_exp_learner.w_curr))
        L_var_trace.append(np.copy(L_var_learner.w_curr))
        value_trace.append(np.copy(value_learner.w_curr))
        while not done:
            action = decide(s_curr, behavior)
            rho_curr = importance_sampling_ratio(target, behavior, s_curr, action)
            rho_accu_nume *= target[s_curr, action]; rho_accu_deno *= behavior[s_curr, action]
            if rho_accu_nume / rho_accu_deno > 1e6: break # for stability issue
            s_next, R_next, done, _ = env.step(action)
            x_next = onehot(s_next, N)
            if s_curr == starting_state and s_next == starting_state + 1:
                lambda_trace[epi, 0] = Lambda.value(x_next)
            # learn expectation of MC-return!
            MC_exp_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha, beta)
            # learn expectation of \Lambda-return!
            L_exp_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha, beta)
            # learn variance of \Lambda-return!
            delta_curr = R_next + gamma(x_next) * np.dot(x_next, value_learner.w_curr) - np.dot(x_curr, value_learner.w_curr)
            try:
                r_bar_next = delta_curr ** 2
            except RuntimeWarning:
                pass
            # gamma_bar_next = (rho_curr * Lambda.value(x_next) * gamma(x_next)) ** 2
            gamma_bar_next = (Lambda.value(x_next) * gamma(x_next)) ** 2
            L_var_learner.learn(r_bar_next, gamma_bar_next, 1, x_next, x_curr, 1, 1, rho_curr, alpha, beta)
            # GD on greedy meta-objective
            v_next = np.dot(x_next, value_learner.w_curr)
            var_L_next, exp_L_next, exp_MC_next = np.dot(x_next, L_var_learner.w_curr), np.dot(x_next, L_exp_learner.w_curr), np.dot(x_next, MC_exp_learner.w_curr)
            coefficient = Lambda.value(x_next) * ((v_next - exp_L_next) ** 2 + var_L_next) + v_next * (exp_L_next + exp_MC_next) - v_next ** 2 - exp_L_next * exp_MC_next
            Lambda.gradient_descent(x_next, kappa * rho_accu_nume / rho_accu_deno * coefficient)
            # learn value
            value_learner.learn(R_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha, beta)
            MC_exp_learner.next(); L_exp_learner.next(); L_var_learner.next(); value_learner.next()
            s_curr, x_curr = s_next, x_next
    return MC_exp_trace, L_exp_trace, L_var_trace, value_trace, lambda_trace