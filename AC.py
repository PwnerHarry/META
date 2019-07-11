import gym, numpy as np
from utils import *
from greedy import *
from mta import *
from TOGTD import *
from TOTD import *

def AC(env, episodes, evaluate, encoder, critic_type='MTA', learner_type='togtd', gamma=lambda x: 0.95, alpha=0.0001, beta=0.0001, alpha_W=0.0001, constant_lambda=1, kappa=0.001):
    # suppose we use exponential softmax on values
    D = encoder(0).size
    value_trace = np.empty((episodes, 1)); value_trace[:] = np.nan
    W = np.zeros((env.action_space.n, D)) # W is the $|A|\times|S|$ parameter matrix for policy
    I = 1
    if learner_type == 'totd':
        LEARNER = TOTD_LEARNER
    elif learner_type == 'togtd':
        LEARNER = TOGTD_LEARNER
    
    if critic_type == 'baseline':
        Lambda = LAMBDA(env, constant_lambda, approximator='constant')
        value_learner = LEARNER(env, D)
    elif critic_type == 'greedy':
        Lambda = LAMBDA(env, approximator='tabular', initial_value=np.ones(env.observation_space.n))
        MC_exp_learner, MC_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D)
    elif critic_type == 'MTA':
        Lambda = LAMBDA(env, np.ones(D) * D / env.observation_space.n, approximator = 'linear')
        MC_exp_learner, L_exp_learner, L_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D), LEARNER(env, D)
    
    for episode in range(episodes):
        o_curr, done = env.reset(), False
        x_curr = encoder(o_curr)
        value_trace[episode, 0] = evaluate(value_learner.w_curr, 'expectation') # Bookkeeping
        if critic_type == 'MTA':
            log_rho_accu = 0
        while not done:
            prob_behavior = softmax(np.matmul(W, x_curr)) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
            prob_target = softmax(np.matmul(W, x_curr))
            action = np.random.choice(range(len(prob_behavior)), p=prob_behavior)
            rho_curr = prob_target[action] / prob_behavior[action]
            if critic_type == 'MTA':
                log_rho_accu += np.log(prob_target[action]) - np.log(prob_behavior[action])
            o_next, r_next, done, _ = env.step(action)
            x_next = encoder(o_next)
            # one-step of policy evaluation of the critic!
            if critic_type == 'baseline':
                if not done:
                    value_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha, beta)
                else:
                    value_learner.learn(r_next, 0, gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha, beta)
            elif critic_type == 'greedy':
                if learner_type == 'togtd':
                    MC_exp_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha, beta)
                elif learner_type == 'totd':
                    MC_exp_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha)
                if not done:
                    delta_curr = r_next + gamma(x_next) * np.dot(x_next, value_learner.w_curr) - np.dot(x_curr, value_learner.w_curr)
                else:
                    delta_curr = r_next - np.dot(x_curr, value_learner.w_curr)
                r_bar_next = delta_curr ** 2
                gamma_bar_next = (rho_curr * gamma(x_next)) ** 2
                if learner_type == 'togtd':
                    MC_var_learner.learn(r_bar_next, gamma_bar_next, 1, x_next, x_curr, 1, 1, 1, alpha, beta)
                elif learner_type == 'totd':
                    MC_var_learner.learn(r_bar_next, gamma_bar_next, 1, x_next, x_curr, 1, 1, 1, alpha)
                errsq, varg = (np.dot(x_next, MC_exp_learner.w_next) - np.dot(x_next, value_learner.w_curr)) ** 2, max(0, np.dot(x_next, MC_var_learner.w_next))
                if errsq + varg > 0:
                    Lambda.w[o_next] = errsq / (errsq + varg)
                else:
                    Lambda.w[o_next] = 1
                if learner_type == 'togtd':
                    value_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.w[o_next], Lambda.w[o_curr], rho_curr, alpha, beta)
                elif learner_type == 'totd':
                    value_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.w[o_next], Lambda.w[o_curr], rho_curr, alpha)
            elif critic_type == 'MTA':
                if learner_type == 'togtd':
                    MC_exp_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha, beta)
                    L_exp_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, 1.1 * alpha, 1.1 * beta)
                elif learner_type == 'totd':
                    MC_exp_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, 1.0, 1.0, rho_curr, alpha)
                    L_exp_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, 1.1 * alpha)
                if not done:
                    delta_curr = r_next + gamma(x_next) * np.dot(x_next, value_learner.w_curr) - np.dot(x_curr, value_learner.w_curr)
                else:
                    delta_curr = r_next - np.dot(x_curr, value_learner.w_curr)
                r_bar_next, gamma_bar_next = delta_curr ** 2, (Lambda.value(x_next) * gamma(x_next)) ** 2
                if learner_type == 'togtd':
                    L_var_learner.learn(r_bar_next, gamma_bar_next, 1, x_next, x_curr, 1, 1, rho_curr, alpha, beta)
                elif learner_type == 'totd':
                    L_var_learner.learn(r_bar_next, gamma_bar_next, 1, x_next, x_curr, 1, 1, rho_curr, alpha)
                # SGD on meta-objective
                rho_acc = np.exp(log_rho_accu)
                # if rho_acc > 1e6: break # too much, not trustworthy
                v_next = np.dot(x_next, value_learner.w_curr)
                var_L_next, exp_L_next, exp_MC_next = np.dot(x_next, L_var_learner.w_curr), np.dot(x_next, L_exp_learner.w_curr), np.dot(x_next, MC_exp_learner.w_curr)
                coefficient = gamma(x_next) ** 2 * Lambda.value(x_next) * ((v_next - exp_L_next) ** 2 + var_L_next) + v_next * (exp_L_next + exp_MC_next) - v_next ** 2 - exp_L_next * exp_MC_next
                Lambda.gradient_descent(x_next, kappa * rho_acc * coefficient)
                # learn value
                if learner_type == 'togtd':
                    value_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha, beta)
                elif learner_type == 'totd':
                    value_learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha)
            
            # one-step of policy improvement of the actor (gradient descent on $W$)! (https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative)
            dsoftmax = jacobian_softmax(prob_behavior)[action, :]
            dlog = dsoftmax / prob_target[0, action]
            grad_W = np.matmul(dlog.reshape(-1, 1), x_curr.reshape(1, -1)) / W # TODO: check if this gradient is correct!
            W += alpha_W * I * rho_curr * grad_W # TODO: make sure the correction of importance sampling ratio is correct
            
            # timestep++
            value_learner.next()
            if critic_type == 'greedy':
                MC_exp_learner.next(); MC_var_learner.next()
            elif critic_type == 'MTA':
                MC_exp_learner.next(); L_exp_learner.next(); L_var_learner.next()
            o_curr, x_curr = o_next, x_next
            I *= gamma(x_next) # TODO: know how the gamma accumulation is implemented!
    return value_trace

def eval_AC_per_run(env, runtime, runtimes, episodes, critic_type, learner_type, gamma, alpha, beta, alpha_W, evaluate, encoder, constant_lambda, kappa):
    print('running %d of %d for AC (%s, %s)' % (runtime + 1, runtimes, critic_type, learner_type))
    value_trace = AC(env, episodes, evaluate, encoder, critic_type, learner_type, gamma=gamma, alpha=alpha, beta=beta, alpha_W=alpha_W, constant_lambda=constant_lambda, kappa=kappa)
    return value_trace.T

def eval_AC(env, critic_type, learner_type, gamma, alpha, beta, alpha_W, runtimes, episodes, evaluate, encoder, constant_lambda=1, kappa=0.001):
    results = Parallel(n_jobs = 1)(delayed(eval_AC_per_run)(env, runtime, runtimes, episodes, critic_type, learner_type, gamma, alpha, beta, alpha_W, evaluate, encoder, constant_lambda, kappa) for runtime in range(runtimes))
    results = np.concatenate(results, axis=0)
    return results