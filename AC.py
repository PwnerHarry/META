import gym, numpy.random, numpy as np
from utils import *
from greedy import *
from mta import *
from TOGTD import *
from TOTD import *

'''
Actor-Critic with Linear Function Approximator and Softmax Policy
Status: Extremely Ugly and Depracated! However, functional!
'''
def AC(env, steps, encoder, encoder_lambda, gamma, alpha, beta, eta, kappa, critic_type='MTA', learner_type='togtd', constant_lambda=1):
    D = np.size(encoder(env.reset()))
    if learner_type == 'togtd':
        LEARNER, slow_lr_dict, fast_lr_dict = TOGTD_LEARNER, {'alpha_curr': alpha, 'beta_curr': beta}, {'alpha_curr': min(1.0, 2 * alpha), 'beta_curr': min(1.0, 2 * alpha)}
    elif learner_type == 'totd':
        LEARNER, slow_lr_dict, fast_lr_dict = TOTD_LEARNER, {'alpha_curr': alpha}, {'alpha_curr': min(1.0, 2 * alpha)}
    if critic_type == 'baseline':
        Lambda = LAMBDA(env, constant_lambda, approximator='constant')
        lambda_curr, lambda_next = constant_lambda, constant_lambda
        value_learner = LEARNER(env, D); learners = [value_learner]
    elif critic_type == 'greedy':
        MC_exp_learner, MC_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D); learners = [MC_exp_learner, MC_var_learner, value_learner]
    elif critic_type == 'MTA':
        Lambda = LAMBDA(env, initial_value=np.zeros(np.size(encoder_lambda(env.reset()))), approximator='linear')
        MC_exp_learner, L_exp_learner, L_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D), LEARNER(env, D); learners = [MC_exp_learner, L_exp_learner, L_var_learner, value_learner]
    W = np.zeros((env.action_space.n, D))
    # W = np.random.uniform(low=-1e-2, high=1e-2, size=(env.action_space.n, D))
    step = 0
    return_trace = np.empty(steps); return_trace[:] = np.nan
    while step < steps:
        for learner in learners: learner.refresh()
        o_curr, done, log_rho_accu, lambda_curr, return_cumulative, I = env.reset(), False, 0, 1, 0, 1; x_curr = encoder(o_curr); x_start = x_curr
        try:
            while not done:
                prob_behavior = softmax(np.matmul(W, x_curr)) # prob_behavior, prob_target = softmax(np.matmul(W, x_curr)), softmax(np.matmul(W, x_curr))
                action = np.random.choice(range(len(prob_behavior)), p=prob_behavior)
                rho_curr = 1 # rho_curr = prob_target[action] / prob_behavior[action]
                o_next, r_next, done, _ = env.step(action); x_next = encoder(o_next); step += 1   
                v_next = float(not done) * np.dot(x_next, value_learner.w_curr)
                delta_curr = r_next + gamma(x_next) * v_next - np.dot(x_curr, value_learner.w_curr)
                if critic_type == 'greedy' or critic_type == 'MTA':
                    warnings.filterwarnings("error")
                    if critic_type == 'greedy':
                        MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **fast_lr_dict)
                        gamma_bar_next = (rho_curr * gamma(x_next)) ** 2
                        MC_var_learner.learn(delta_curr ** 2, done, gamma_bar_next, 1, x_next, x_curr, 1, 1, 1, **fast_lr_dict)
                        errsq, varg = (np.dot(x_next, MC_exp_learner.w_next) - np.dot(x_next, value_learner.w_curr)) ** 2, max(0, np.dot(x_next, MC_var_learner.w_next))
                        lambda_next = 1
                        if step > 0.1 * steps and errsq + varg > np.sqrt(np.finfo(float).eps): lambda_next = errsq / (errsq + varg)
                    elif critic_type == 'MTA':
                        lambda_curr, lambda_next = Lambda.value(encoder_lambda(o_curr)), Lambda.value(encoder_lambda(o_next))
                        # log_rho_accu += np.log(prob_target[action]) - np.log(prob_behavior[action])
                        MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **fast_lr_dict)
                        L_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, lambda_next, lambda_next, rho_curr, **fast_lr_dict)
                        L_var_learner.learn(delta_curr ** 2, done, (lambda_next * gamma(x_next)) ** 2, 1, x_next, x_curr, 1, 1, rho_curr, **fast_lr_dict)
                        VmE = v_next - np.dot(x_next, L_exp_learner.w_curr)
                        L_var_next = np.dot(x_next, L_var_learner.w_curr)
                        if step > 100 and np.linalg.norm(x_next - x_curr, 2) > 0 and L_var_next > np.sqrt(np.finfo(float).eps):
                            coefficient = gamma(x_next) ** 2 * (lambda_next * (VmE ** 2 + L_var_next) + VmE * (v_next - np.dot(x_next, MC_exp_learner.w_curr)))                        
                            Lambda.GD(encoder_lambda(o_next), kappa * np.exp(log_rho_accu) * coefficient)
                            lambda_next = Lambda.value(encoder_lambda(o_next))
                    warnings.filterwarnings("default")
                # one-step of policy evaluation of the critic!
                value_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, lambda_next, lambda_curr, rho_curr, **slow_lr_dict)
                # one-step of policy improvement of the actor (gradient ascent on $W$)! (https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative)
                delta_curr_new = r_next + float(not done) * gamma(x_next) * np.dot(x_next, value_learner.w_next) - np.dot(x_curr, value_learner.w_next)
                if step / steps >= 0.5:
                    W += eta * I * rho_curr * delta_curr_new * get_grad_W(W, prob_behavior, np.diagflat(prob_behavior), action, x_curr) # TODO: make sure the correction of importance sampling ratio is correct            
                # timestep++
                return_cumulative += I * r_next
                o_curr, x_curr, lambda_curr, I = o_next, x_next, lambda_next, I * gamma(x_next) # TODO: know how the gamma accumulation is implemented!
                for learner in learners: learner.next()
        except RuntimeWarning:
            break
        except ValueError:
            break
        if step < steps:
            return_trace[step] = return_cumulative
    warnings.filterwarnings("default")
    return_trace[-1] = return_cumulative
    return return_trace

def eval_AC_per_run(env_name, runtime, runtimes, steps, critic_type, learner_type, gamma, alpha, beta, eta, encoder, encoder_lambda, constant_lambda, kappa):
    np.random.seed(seed=runtime)
    env = gym.make(env_name)
    env.seed(runtime)
    if critic_type == 'baseline':
        print('%d of %d for AC(%g, %s), alpha: %g, beta: %g, eta: %g' % (runtime + 1, runtimes, constant_lambda, learner_type, alpha, beta, eta))
    elif critic_type == 'greedy':
        print('%d of %d for AC(greedy, %s), alpha: %g, beta: %g, eta: %g' % (runtime + 1, runtimes, learner_type, alpha, beta, eta))
    elif critic_type == 'MTA':
        print('%d of %d for AC(MTA, %s), alpha: %g, beta: %g, eta: %g, kappa: %g' % (runtime + 1, runtimes, learner_type, alpha, beta, eta, kappa))
    return_trace = AC(env, steps, encoder, encoder_lambda, gamma=gamma, alpha=alpha, beta=beta, eta=eta, kappa=kappa, critic_type=critic_type, learner_type=learner_type, constant_lambda=constant_lambda)
    return return_trace.reshape(1, -1)

def eval_AC(env_name, critic_type, learner_type, gamma, alpha, beta, eta, runtimes, steps, encoder, encoder_lambda, constant_lambda=1, kappa=0.001):
    results = Parallel(n_jobs=-1)(delayed(eval_AC_per_run)(env_name, runtime, runtimes, steps, critic_type, learner_type, gamma, alpha, beta, eta, encoder, encoder_lambda, constant_lambda, kappa) for runtime in range(runtimes))
    return np.concatenate(results, axis=0)