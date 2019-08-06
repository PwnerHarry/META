import gym, warnings, numpy as np
from utils import *
from TOGTD import *
from TOTD import *

def MTA(env, episodes, target, behavior, evaluate, Lambda, encoder, learner_type='togtd', gamma=lambda x: 0.95, alpha=0.05, beta=0.05, kappa=0.01):
    D = np.size(encoder(env.reset()))
    if np.size(Lambda.w) == D:
        encoder_lambda = encoder
    else:
        encoder_lambda = lambda x: onehot(x, env.observation_space.n)
    value_trace = np.empty(episodes); value_trace[:] = np.nan
    if learner_type == 'togtd':
        LEARNER, slow_lr_dict, fast_lr_dict = TOGTD_LEARNER, {'alpha_curr': alpha, 'beta_curr': beta}, {'alpha_curr': min(1.0, 2 * alpha), 'beta_curr': min(1.0, 2 * alpha)}
    elif learner_type == 'totd':
        LEARNER, slow_lr_dict, fast_lr_dict = TOTD_LEARNER, {'alpha_curr': alpha}, {'alpha_curr': min(1.0, 2 * alpha)}
    MC_exp_learner, L_exp_learner, L_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D), LEARNER(env, D)
    learners = [MC_exp_learner, L_exp_learner, L_var_learner, value_learner]
    warnings.filterwarnings("error")
    for episode in range(episodes):
        o_curr, done, log_rho_accu = env.reset(), False, 0; x_curr = encoder(o_curr)
        for learner in learners: learner.refresh()
        value_trace[episode] = evaluate(value_learner.w_curr, 'expectation')
        try:
            while not done:
                action = decide(o_curr, behavior)
                rho_curr = importance_sampling_ratio(target, behavior, o_curr, action)
                log_rho_accu += np.log(rho_curr)
                o_next, r_next, done, _ = env.step(action)
                x_next = encoder(o_next)
                lambda_curr, lambda_next = Lambda.value(encoder_lambda(o_curr)), Lambda.value(encoder_lambda(o_next))
                MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **fast_lr_dict)
                L_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, lambda_next, lambda_curr, rho_curr, **fast_lr_dict)
                v_next = float(not done) * np.dot(x_next, value_learner.w_curr)
                delta_curr = r_next + gamma(x_next) * v_next - np.dot(x_curr, value_learner.w_curr)
                L_var_learner.learn(delta_curr ** 2, done, (lambda_next * gamma(x_next)) ** 2, 1, x_next, x_curr, 1, 1, rho_curr, **fast_lr_dict)
                # SGD on meta-objective
                VmE = v_next - np.dot(x_next, L_exp_learner.w_curr)
                L_var_next = np.dot(x_next, L_var_learner.w_curr)
                if L_var_next > np.sqrt(np.finfo(float).eps):
                    coefficient = gamma(x_next) ** 2 * (lambda_next * (VmE ** 2 + L_var_next) + VmE * (v_next - np.dot(x_next, MC_exp_learner.w_curr)))                        
                    Lambda.GD(encoder_lambda(o_next), kappa * np.exp(log_rho_accu) * coefficient)
                    lambda_next = Lambda.value(encoder_lambda(o_next))
                # learn value
                value_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, lambda_next, lambda_curr, rho_curr, **slow_lr_dict)
                for learner in learners: learner.next()
                o_curr, x_curr = o_next, x_next
        except RuntimeWarning:
            print('RuntimeWarning captured, possibly due to numerical stability issues')
            break
    warnings.filterwarnings("default")
    return value_trace

def eval_MTA_per_run(env_name, runtime, runtimes, episodes, target, behavior, kappa, gamma, Lambda, alpha, beta, evaluate, encoder, learner_type):
    np.random.seed(seed=runtime)
    env = gym.make(env_name)
    env.seed(runtime)
    if learner_type == 'togtd':
        print('%d of %d for MTA, alpha: %g, beta: %g, kappa: %g' % (runtime + 1, runtimes, alpha, beta, kappa))
    elif learner_type == 'totd':
        print('%d of %d for MTA, alpha: %g, kappa: %g' % (runtime + 1, runtimes, alpha, kappa))
    value_trace = MTA(env, episodes, target, behavior, evaluate, Lambda, encoder, learner_type='togtd', gamma=gamma, alpha=alpha, beta=beta, kappa=kappa)
    return value_trace.reshape(1, -1)

def eval_MTA(env_name, behavior, target, kappa, gamma, alpha, beta, runtimes, episodes, evaluate, encoder, learner_type='togtd', parametric_lambda=False):
    env = gym.make(env_name)
    if parametric_lambda:
        initial_weights_lambda = np.zeros(np.size(encoder(0)))
    else:
        initial_weights_lambda = np.zeros(np.size(onehot(0, env.observation_space.n)))
    LAMBDAS = []
    for runtime in range(runtimes):
        LAMBDAS.append(LAMBDA(env, initial_value=initial_weights_lambda, approximator='linear'))
    results = Parallel(n_jobs=-1)(delayed(eval_MTA_per_run)(env_name, runtime, runtimes, episodes, target, behavior, kappa, gamma, LAMBDAS[runtime], alpha, beta, evaluate, encoder, learner_type) for runtime in range(runtimes))
    return np.concatenate(results, axis=0)