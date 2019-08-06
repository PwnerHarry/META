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
def AC(env, episodes, encoder, gamma, alpha, beta, eta, kappa, critic_type='MTA', learner_type='togtd', constant_lambda=1):
    D = np.size(encoder(env.reset()))
    if learner_type == 'togtd':
        LEARNER, slow_lr_dict, fast_lr_dict = TOGTD_LEARNER, {'alpha_curr': alpha, 'beta_curr': beta}, {'alpha_curr': min(1.0, 2 * alpha), 'beta_curr': min(1.0, 2 * alpha)}
    elif learner_type == 'totd':
        LEARNER, slow_lr_dict, fast_lr_dict = TOTD_LEARNER, {'alpha_curr': alpha}, {'alpha_curr': min(1.0, 2 * alpha)}
    if critic_type == 'baseline':
        Lambda = LAMBDA(env, constant_lambda, approximator='constant')
        value_learner = LEARNER(env, D); learners = [value_learner]
    elif critic_type == 'greedy':
        MC_exp_learner, MC_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D); learners = [MC_exp_learner, MC_var_learner, value_learner]
    elif critic_type == 'MTA':
        Lambda = LAMBDA(env, initial_value=np.zeros(D), approximator='linear')
        MC_exp_learner, L_exp_learner, L_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D), LEARNER(env, D); learners = [MC_exp_learner, L_exp_learner, L_var_learner, value_learner]
    W = np.zeros((env.action_space.n, D))
    # W = numpy.random.normal(0, eta, env.action_space.n * D).reshape(env.action_space.n, D) # W is the $|A|\times|S|$ parameter matrix for policy
    # W = numpy.random.uniform(low=-eta, high=eta, size=env.action_space.n * D).reshape(env.action_space.n, D) # W is the $|A|\times|S|$ parameter matrix for policy
    # W = np.load('W_file.npy')
    return_trace = np.empty(episodes); return_trace[:] = np.nan
    break_flag = False
    for episode in range(episodes):
        if break_flag: break
        for learner in learners: learner.refresh()
        o_curr, done, log_rho_accu, lambda_curr, return_cumulative, I = env.reset(), False, 0, 1, 0, 1; x_curr = encoder(o_curr); x_start = x_curr
        while not done:
            prob_behavior = softmax(np.matmul(W, x_curr)) # prob_behavior, prob_target = softmax(np.matmul(W, x_curr)), softmax(np.matmul(W, x_curr))
            action = np.random.choice(range(len(prob_behavior)), p=prob_behavior)
            rho_curr = 1 # rho_curr = prob_target[action] / prob_behavior[action]
            o_next, r_next, done, _ = env.step(action); x_next = encoder(o_next)        
            v_next = float(not done) * np.dot(x_next, value_learner.w_curr)
            delta_curr = r_next + gamma(x_next) * v_next - np.dot(x_curr, value_learner.w_curr)
            if critic_type == 'greedy' or critic_type == 'MTA':
                warnings.filterwarnings("error")
                try:
                    if critic_type == 'greedy':
                        MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **fast_lr_dict)
                        gamma_bar_next = (rho_curr * gamma(x_next)) ** 2
                        MC_var_learner.learn(delta_curr ** 2, done, gamma_bar_next, 1, x_next, x_curr, 1, 1, 1, **fast_lr_dict)
                        errsq, varg = (np.dot(x_next, MC_exp_learner.w_next) - np.dot(x_next, value_learner.w_curr)) ** 2, max(0, np.dot(x_next, MC_var_learner.w_next))
                        lambda_next = 1
                        if errsq + varg > np.sqrt(np.finfo(float).eps): lambda_next = errsq / (errsq + varg)
                    elif critic_type == 'MTA':
                        # log_rho_accu += np.log(prob_target[action]) - np.log(prob_behavior[action])
                        MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **fast_lr_dict)
                        L_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, **fast_lr_dict)
                        L_var_learner.learn(delta_curr ** 2, done, (Lambda.value(x_next) * gamma(x_next)) ** 2, 1, x_next, x_curr, 1, 1, rho_curr, **fast_lr_dict)
                        VmE = v_next - np.dot(x_next, L_exp_learner.w_curr)
                        L_var_next = np.dot(x_next, L_var_learner.w_curr)
                        if L_var_next > np.sqrt(np.finfo(float).eps):
                            coefficient = gamma(x_next) ** 2 * (Lambda.value(x_next) * (VmE ** 2 + L_var_next) + VmE * (v_next - np.dot(x_next, MC_exp_learner.w_curr)))                        
                            Lambda.GD(x_next, kappa * np.exp(log_rho_accu) * coefficient, normalize=True)
                except RuntimeWarning:
                    break_flag = True
                    break
                warnings.filterwarnings("default")
            if critic_type != 'greedy':
                lambda_curr, lambda_next = Lambda.value(x_curr), Lambda.value(x_next)
            # one-step of policy evaluation of the critic!
            value_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, lambda_next, lambda_curr, rho_curr, **slow_lr_dict)
            # one-step of policy improvement of the actor (gradient ascent on $W$)! (https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative)
            delta_curr_new = r_next + float(not done) * gamma(x_next) * np.dot(x_next, value_learner.w_next) - np.dot(x_curr, value_learner.w_next)
            W += eta * I * rho_curr * delta_curr_new * get_grad_W(W, prob_behavior, np.diagflat(prob_behavior), action, x_curr) # TODO: make sure the correction of importance sampling ratio is correct            
            # timestep++
            return_cumulative += I * r_next
            o_curr, x_curr, lambda_curr, I = o_next, x_next, lambda_next, I * gamma(x_next) # TODO: know how the gamma accumulation is implemented!
            for learner in learners: learner.next()
        return_trace[episode] = return_cumulative
        # if return_cumulative:
        #     print('episode: %g,\t lambda(0): %.2f,\t return_cumulative: %g' % (episode, Lambda.value(x_start), return_cumulative))
    warnings.filterwarnings("default")
    return return_trace

def eval_AC_per_run(env_name, runtime, runtimes, episodes, critic_type, learner_type, gamma, alpha, beta, eta, encoder, constant_lambda, kappa):
    np.random.seed(seed=runtime)
    env = gym.make(env_name)
    env.seed(runtime)
    if critic_type == 'baseline':
        print('%d of %d for AC(%g, %s), alpha: %g, beta: %g, eta: %g' % (runtime + 1, runtimes, constant_lambda, learner_type, alpha, beta, eta))
    elif critic_type == 'greedy':
        print('%d of %d for AC(greedy, %s), alpha: %g, beta: %g, eta: %g' % (runtime + 1, runtimes, learner_type, alpha, beta, eta))
    elif critic_type == 'MTA':
        print('%d of %d for AC(MTA, %s), alpha: %g, beta: %g, eta: %g, kappa: %g' % (runtime + 1, runtimes, learner_type, alpha, beta, eta, kappa))
    return_trace = AC(env, episodes, encoder, gamma=gamma, alpha=alpha, beta=beta, eta=eta, kappa=kappa, critic_type=critic_type, learner_type=learner_type, constant_lambda=constant_lambda)
    return return_trace.reshape(1, -1)

def eval_AC(env_name, critic_type, learner_type, gamma, alpha, beta, eta, runtimes, episodes, encoder, constant_lambda=1, kappa=0.001):
    results = Parallel(n_jobs=-1)(delayed(eval_AC_per_run)(env_name, runtime, runtimes, episodes, critic_type, learner_type, gamma, alpha, beta, eta, encoder, constant_lambda, kappa) for runtime in range(runtimes))
    return np.concatenate(results, axis=0)