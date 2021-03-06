import gym, warnings, numpy as np
from utils import *
from TOGTD import *
from TOTD import *

def greedy(env, steps, target, behavior, evaluate, Lambda, encoder, learner_type, gamma=lambda x: 0.95, alpha=0.05, beta=0.05):
    D = np.size(encoder(env.reset()))
    if learner_type == 'togtd':
        LEARNER, slow_lr_dict, fast_lr_dict = TOGTD_LEARNER, {'alpha_curr': alpha, 'beta_curr': beta}, {'alpha_curr': min(1.0, 2 * alpha), 'beta_curr': min(1.0, 2.0 * alpha)}
    elif learner_type == 'totd':
        LEARNER, slow_lr_dict, fast_lr_dict = TOTD_LEARNER, {'alpha_curr': alpha}, {'alpha_curr': min(1.0, 2 * alpha)}
    MC_exp_learner, MC_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D)
    learners = [MC_exp_learner, MC_var_learner, value_learner]
    value_trace = np.empty(steps // 1000); value_trace[:] = np.nan
    warnings.filterwarnings("error")
    step = 0
    while step < steps:
        o_curr, done = env.reset(), False; x_curr = encoder(o_curr)
        for learner in learners: learner.refresh()
        if step % 1000 == 0:
            value_trace[step // 1000] = evaluate(value_learner.w_curr, 'expectation')
        try:
            while not done:
                action = decide(o_curr, behavior)
                rho_curr = importance_sampling_ratio(target, behavior, o_curr, action)
                o_next, r_next, done, _ = env.step(action); x_next = encoder(o_next); step += 1
                MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **fast_lr_dict)
                delta_curr = r_next + float(not done) * gamma(x_next) * np.dot(x_next, value_learner.w_curr) - np.dot(x_curr, value_learner.w_curr)
                gamma_bar_next = (rho_curr * gamma(x_next)) ** 2
                MC_var_learner.learn(delta_curr ** 2, done, gamma_bar_next, 1, x_next, x_curr, 1, 1, 1, **fast_lr_dict)
                errsq = (np.dot(x_next, MC_exp_learner.w_next) - np.dot(x_next, value_learner.w_curr)) ** 2
                varg = max(0, np.dot(x_next, MC_var_learner.w_next))
                Lambda.w[o_next] = 1
                if step > 0.1 * steps and errsq + varg > np.sqrt(np.finfo(float).eps): # a safer threshold for numerical stability
                    try:
                        Lambda.w[o_next] = errsq / (errsq + varg)
                    except RuntimeWarning:
                        pass
                value_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.w[o_next], Lambda.w[o_curr], rho_curr, **slow_lr_dict)
                for learner in learners: learner.next()
                o_curr, x_curr = o_next, x_next
        except RuntimeWarning:
            print('RuntimeWarning captured, possibly due to numerical stability issues')
            break
    warnings.filterwarnings("default")
    return value_trace

def eval_greedy_per_run(env_name, runtime, runtimes, steps, target, behavior, encoder, gamma, Lambda, alpha, beta, evaluate, learner_type):
    np.random.seed(seed=runtime)
    env = gym.make(env_name)
    env.seed(runtime)
    if learner_type == 'togtd':
        print('%d of %d for greedy, alpha: %g, beta: %g' % (runtime + 1, runtimes, alpha, beta))
    elif learner_type == 'totd':
        print('%d of %d for greedy, alpha: %g' % (runtime + 1, runtimes, alpha))
    value_trace = greedy(env, steps, target, behavior, evaluate=evaluate, Lambda=Lambda, encoder=encoder,gamma=gamma, alpha=alpha, beta=beta, learner_type=learner_type)
    return value_trace.reshape(1, -1)

def eval_greedy(env_name, behavior, target, evaluate, gamma, alpha, beta, runtimes, steps, encoder, learner_type='togtd'):
    LAMBDAS = []
    env = gym.make(env_name)
    for runtime in range(runtimes):
        LAMBDAS.append(LAMBDA(env, approximator='tabular', initial_value=np.ones(env.observation_space.n)))
    results = Parallel(n_jobs = -1)(delayed(eval_greedy_per_run)(env_name, runtime, runtimes, steps, target, behavior, encoder, gamma, LAMBDAS[runtime], alpha, beta, evaluate, learner_type=learner_type) for runtime in range(runtimes))
    return np.concatenate(results, axis=0)