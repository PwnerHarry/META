import gym, warnings, numpy as np
from utils import *
from TOGTD import *
from TOTD import *

def MTA(env, episodes, target, behavior, evaluate, Lambda, encoder, learner_type='togtd', gamma=lambda x: 0.95, alpha=0.05, beta=0.05, kappa=0.01):
    D = encoder(0).size
    value_trace = np.empty(episodes); value_trace[:] = np.nan
    if learner_type == 'totd':
        LEARNER = TOTD_LEARNER; lr_dict = {'alpha_curr': alpha}; lr_larger_dict = {'alpha_curr': 1.1 * alpha}
    elif learner_type == 'togtd':
        LEARNER = TOGTD_LEARNER; lr_dict = {'alpha_curr': alpha, 'beta_curr': beta}; lr_larger_dict = {'alpha_curr': min(1.0, 1.1 * alpha), 'beta_curr': min(1.0, 1.1 * beta)}
    MC_exp_learner, L_exp_learner, L_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D), LEARNER(env, D)
    for episode in range(episodes):
        o_curr, done, log_rho_accu = env.reset(), False, 0; x_curr = encoder(o_curr)
        MC_exp_learner.refresh(); L_exp_learner.refresh(); L_var_learner.refresh(); value_learner.refresh()
        value_trace[episode] = evaluate(value_learner.w_curr, 'expectation')
        warnings.filterwarnings("error")
        try:
            while not done:
                action = decide(o_curr, behavior)
                rho_curr = importance_sampling_ratio(target, behavior, o_curr, action)
                log_rho_accu += np.log(rho_curr)
                o_next, r_next, done, _ = env.step(action)
                x_next = encoder(o_next)
                MC_exp_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, **lr_dict)
                L_exp_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, **lr_larger_dict)
                v_next = np.dot(x_next, value_learner.w_curr)
                delta_curr = r_next + float(not done) * gamma(x_next) * v_next - np.dot(x_curr, value_learner.w_curr)
                gamma_bar_next = (Lambda.value(x_next) * gamma(x_next)) ** 2
                L_var_learner.learn(delta_curr ** 2, gamma_bar_next, 1, x_next, x_curr, 1, 1, rho_curr, **lr_dict)
                # SGD on meta-objective
                VmE = v_next - np.dot(x_next, L_exp_learner.w_curr)
                coefficient = gamma(x_next) ** 2 * (Lambda.value(x_next) * (VmE ** 2 + np.dot(x_next, L_var_learner.w_curr)) + VmE * (v_next - np.dot(x_next, MC_exp_learner.w_curr)))
                Lambda.GD(x_next, kappa * np.exp(log_rho_accu) * coefficient)
                # learn value
                value_learner.learn(r_next, float(not done) * gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, **lr_dict)
                MC_exp_learner.next(); L_exp_learner.next(); L_var_learner.next(); value_learner.next()
                o_curr, x_curr = o_next, x_next
            warnings.filterwarnings("default")
        except RuntimeWarning:
            print('RuntimeWarning captured, possibly due to numerical stability issues')
            break
    return value_trace

def eval_MTA_per_run(env, runtime, runtimes, episodes, target, behavior, kappa, gamma, Lambda, alpha, beta, evaluate, encoder, learner_type):
    np.random.seed(seed=runtime)
    if learner_type == 'togtd':
        print('%d of %d for MTA, alpha: %g, beta: %g, kappa: %g' % (runtime + 1, runtimes, alpha, beta, kappa))
    elif learner_type == 'totd':
        print('%d of %d for MTA, alpha: %g, kappa: %g' % (runtime + 1, runtimes, alpha, kappa))
    value_trace = MTA(env, episodes, target, behavior, evaluate, Lambda, encoder, learner_type='togtd', gamma=gamma, alpha=alpha, beta=beta, kappa=kappa)
    return value_trace.reshape(1, -1)

def eval_MTA(env, behavior, target, kappa, gamma, alpha, beta, runtimes, episodes, evaluate, encoder, learner_type='togtd', parametric_lambda=True):
    if parametric_lambda:
        initial_weights_lambda = np.linalg.lstsq(get_state_set_matrix(env, encoder), np.ones(env.observation_space.n), rcond=None)[0]
    else:
        initial_weights_lambda = np.ones(env.observation_space.n)
    LAMBDAS = []
    for runtime in range(runtimes):
        if parametric_lambda:
            LAMBDAS.append(LAMBDA(env, initial_value=initial_weights_lambda, approximator='linear'))
        else:
            LAMBDAS.append(LAMBDA(env, initial_value=initial_weights_lambda, approximator='tabular', state_set_matrix=get_state_set_matrix(env, encoder)))
    results = Parallel(n_jobs=-1)(delayed(eval_MTA_per_run)(env, runtime, runtimes, episodes, target, behavior, kappa, gamma, LAMBDAS[runtime], alpha, beta, evaluate, encoder, learner_type) for runtime in range(runtimes))
    return np.concatenate(results, axis=0)