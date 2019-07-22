import gym, torch, numpy.random, numpy as np
from utils import *
from greedy import *
from mta import *
from TOGTD import *
from TOTD import *
from torch.autograd import Variable

'''
Actor-Critic with Linear Function Approximator and Softmax Policy
Status: Extremely Ugly and Depracated! However, functional!
'''
def AC(env, episodes, encoder, gamma, alpha, beta, eta, kappa, critic_type='MTA', learner_type='togtd', constant_lambda=1):
    if encoder is None:
        D = np.size(env.reset())
    else:
        D = encoder(0).size
    if learner_type == 'totd':
        LEARNER = TOTD_LEARNER; lr_dict = {'alpha_curr': alpha}; lr_larger_dict = {'alpha_curr': 1.1 * alpha}
    elif learner_type == 'togtd':
        LEARNER = TOGTD_LEARNER; lr_dict = {'alpha_curr': alpha, 'beta_curr': beta}; lr_larger_dict = {'alpha_curr': min(1.0, 1.1 * alpha), 'beta_curr': min(1.0, 1.1 * beta)}
    if critic_type == 'baseline':
        Lambda = LAMBDA(env, constant_lambda, approximator='constant')
        value_learner = LEARNER(env, D); learners = [value_learner]
    elif critic_type == 'greedy':
        MC_exp_learner, MC_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D); learners = [MC_exp_learner, MC_var_learner, value_learner]
    elif critic_type == 'MTA':
        Lambda = LAMBDA(env, initial_value=np.zeros(D), approximator='linear')
        MC_exp_learner, L_exp_learner, L_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D), LEARNER(env, D); learners = [MC_exp_learner, L_exp_learner, L_var_learner, value_learner]
    # W = np.zeros((env.action_space.n, D))
    W = numpy.random.normal(0, 1, env.action_space.n * D).reshape(env.action_space.n, D) # W is the $|A|\times|S|$ parameter matrix for policy
    return_trace = np.empty(episodes); return_trace[:] = np.nan
    break_flag = False
    for episode in range(episodes):
        if break_flag:
            break
        for learner in learners: learner.refresh()
        o_curr, done, log_rho_accu, lambda_curr, return_cumulative, I = env.reset(), False, 0, 1, 0, 1
        if encoder is None: 
            x_curr = o_curr
        else:
            x_curr = encoder(o_curr)
        while not done:
            prob_behavior = softmax(np.matmul(W, x_curr)) # prob_behavior, prob_target = softmax(np.matmul(W, x_curr)), softmax(np.matmul(W, x_curr))
            action = np.random.choice(range(len(prob_behavior)), p=prob_behavior)
            rho_curr = 1 # rho_curr = prob_target[action] / prob_behavior[action]
            o_next, r_next, done, _ = env.step(action)
            if encoder is None: 
                x_next = o_next
            else:
                x_next = encoder(o_next)        
            v_next = float(not done) * np.dot(x_next, value_learner.w_curr)
            delta_curr = r_next + gamma(x_next) * v_next - np.dot(x_curr, value_learner.w_curr)
            if critic_type == 'greedy':
                MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **lr_dict)
                gamma_bar_next = (rho_curr * gamma(x_next)) ** 2
                MC_var_learner.learn(delta_curr ** 2, done, gamma_bar_next, 1, x_next, x_curr, 1, 1, 1, **lr_dict)
                errsq, varg = (np.dot(x_next, MC_exp_learner.w_next) - np.dot(x_next, value_learner.w_curr)) ** 2, max(0, np.dot(x_next, MC_var_learner.w_next))
                lambda_next = 1
                if errsq + varg > np.sqrt(np.finfo(float).eps): # a safer threshold for numerical stability
                    warnings.filterwarnings("error")
                    try:
                        lambda_next = errsq / (errsq + varg)
                    except RuntimeWarning:
                        pass
                    warnings.filterwarnings("default")
            else:
                if critic_type == 'MTA':
                    warnings.filterwarnings("error")
                    try:
                        # log_rho_accu += np.log(prob_target[action]) - np.log(prob_behavior[action])
                        MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **lr_dict)
                        L_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, **lr_larger_dict)
                        L_var_learner.learn(delta_curr ** 2, done, (Lambda.value(x_next) * gamma(x_next)) ** 2, 1, x_next, x_curr, 1, 1, rho_curr, **lr_dict)
                        VmE = v_next - np.dot(x_next, L_exp_learner.w_curr)
                        coefficient = gamma(x_next) ** 2 * (Lambda.value(x_next) * (VmE ** 2 + np.dot(x_next, L_var_learner.w_curr)) + VmE * (v_next - np.dot(x_next, MC_exp_learner.w_curr)))
                        Lambda.GD(x_next, kappa * np.exp(log_rho_accu) * coefficient)
                    except RuntimeWarning:
                        break_flag = True
                        break
                    warnings.filterwarnings("default")
                lambda_curr, lambda_next = Lambda.value(x_curr), Lambda.value(x_next)
            # one-step of policy evaluation of the critic!
            value_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, lambda_next, lambda_curr, rho_curr, **lr_dict)
            # one-step of policy improvement of the actor (gradient descent on $W$)! (https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative)
            W += eta * I * rho_curr * delta_curr * get_grad_W(W, prob_behavior, np.diagflat(prob_behavior), action, x_curr) # TODO: make sure the correction of importance sampling ratio is correct            
            # timestep++
            return_cumulative += I * r_next
            o_curr, x_curr, lambda_curr, I = o_next, x_next, lambda_next, I * gamma(x_next) # TODO: know how the gamma accumulation is implemented!
            for learner in learners: learner.next()
        return_trace[episode] = return_cumulative
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