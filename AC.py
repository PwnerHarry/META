import gym, numpy as np
from utils import *
from greedy import *
from mta import *
from TOGTD import *
from TOTD import *

def AC(env, episodes, encoder, gamma, alpha, beta, eta, kappa, critic_type='MTA', learner_type='togtd', constant_lambda=1):
    # suppose we use exponential softmax on values
    D = encoder(0).size
    return_trace = np.empty(episodes); return_trace[:] = np.nan
    W = np.ones((env.action_space.n, D)) # W is the $|A|\times|S|$ parameter matrix for policy
    if learner_type == 'totd':
        LEARNER = TOTD_LEARNER; lr_dict = {'alpha_curr': alpha}; lr_larger_dict = {'alpha_curr': 1.1 * alpha}
    elif learner_type == 'togtd':
        LEARNER = TOGTD_LEARNER; lr_dict = {'alpha_curr': alpha, 'beta_curr': beta}; lr_larger_dict = {'alpha_curr': min(1.0, 1.1 * alpha), 'beta_curr': min(1.0, 1.1 * beta)}
    if critic_type == 'baseline':
        Lambda = LAMBDA(env, constant_lambda, approximator='constant')
        value_learner = LEARNER(env, D); learners = [value_learner]
    elif critic_type == 'greedy':
        Lambda = LAMBDA(env, approximator='tabular', initial_value=np.ones(env.observation_space.n))
        MC_exp_learner, MC_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D); learners = [MC_exp_learner, MC_var_learner, value_learner]
    elif critic_type == 'MTA':
        Lambda = LAMBDA(env, initial_value=np.linalg.lstsq(get_state_set_matrix(env, encoder), np.ones(env.observation_space.n), rcond=None)[0], approximator='linear')
        MC_exp_learner, L_exp_learner, L_var_learner, value_learner = LEARNER(env, D), LEARNER(env, D), LEARNER(env, D), LEARNER(env, D); learners = [MC_exp_learner, L_exp_learner, L_var_learner, value_learner]
    for episode in range(episodes):
        o_curr, done, log_rho_accu, return_cumulative, I = env.reset(), False, 0, 0, 1; x_curr = encoder(o_curr)
        for learner in learners:
            learner.refresh()
        while not done:
            prob_behavior, prob_target = softmax(np.matmul(W, x_curr)), softmax(np.matmul(W, x_curr)) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html
            action = np.random.choice(range(len(prob_behavior)), p=prob_behavior); rho_curr = prob_target[action] / prob_behavior[action]
            o_next, r_next, done, _ = env.step(action); x_next = encoder(o_next)
            if critic_type == 'greedy':
                MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **lr_dict)
                delta_curr = r_next + float(not done) * gamma(x_next) * np.dot(x_next, value_learner.w_curr) - np.dot(x_curr, value_learner.w_curr)
                gamma_bar_next = (rho_curr * gamma(x_next)) ** 2
                MC_var_learner.learn(delta_curr ** 2, done, gamma_bar_next, 1, x_next, x_curr, 1, 1, 1, **lr_dict)
                errsq, varg = (np.dot(x_next, MC_exp_learner.w_next) - np.dot(x_next, value_learner.w_curr)) ** 2, max(0, np.dot(x_next, MC_var_learner.w_next))
                if errsq + varg > 0:
                    Lambda.w[o_next] = errsq / (errsq + varg)
                else:
                    Lambda.w[o_next] = 1
            elif critic_type == 'MTA':
                log_rho_accu += np.log(prob_target[action]) - np.log(prob_behavior[action])
                MC_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, 1, 1, rho_curr, **lr_dict)
                L_exp_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, **lr_larger_dict)
                v_next = np.dot(x_next, value_learner.w_curr)
                delta_curr = r_next + float(not done) * gamma(x_next) * v_next - np.dot(x_curr, value_learner.w_curr)
                gamma_bar_next = (Lambda.value(x_next) * gamma(x_next)) ** 2
                L_var_learner.learn(delta_curr ** 2, done, gamma_bar_next, 1, x_next, x_curr, 1, 1, rho_curr, **lr_dict)
                # SGD on meta-objective
                VmE = v_next - np.dot(x_next, L_exp_learner.w_curr)
                coefficient = gamma(x_next) ** 2 * (Lambda.value(x_next) * (VmE ** 2 + np.dot(x_next, L_var_learner.w_curr)) + VmE * (v_next - np.dot(x_next, MC_exp_learner.w_curr)))
                Lambda.GD(x_next, kappa * np.exp(log_rho_accu) * coefficient)
            # one-step of policy evaluation of the critic!
            value_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, **lr_dict)
            # one-step of policy improvement of the actor (gradient descent on $W$)! (https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative)
            dsoftmax = jacobian_softmax(prob_behavior)[action, :]
            dlog = dsoftmax / prob_target[action]
            grad_W = np.matmul(dlog.reshape(-1, 1), x_curr.reshape(1, -1)) / W # TODO: This gives divided by 0 at the first, check if this gradient is correct!
            W += eta * I * rho_curr * grad_W # TODO: make sure the correction of importance sampling ratio is correct
            # timestep++
            return_cumulative += I * r_next
            for learner in learners:
                learner.next()
            o_curr, x_curr = o_next, x_next
            I *= gamma(x_next) # TODO: know how the gamma accumulation is implemented!
        return_trace[episode] = return_cumulative # Bookkeeping
    return return_trace

def eval_AC_per_run(env, runtime, runtimes, episodes, critic_type, learner_type, gamma, alpha, beta, eta, encoder, constant_lambda, kappa):
    np.random.seed(seed=runtime)
    print('%d of %d for AC (%s, %s)' % (runtime + 1, runtimes, critic_type, learner_type))
    return_trace = AC(env, episodes, encoder, gamma=gamma, alpha=alpha, beta=beta, eta=eta, kappa=kappa, critic_type=critic_type, learner_type=learner_type, constant_lambda=constant_lambda)
    return return_trace.reshape(1, -1)

def eval_AC(env, critic_type, learner_type, gamma, alpha, beta, eta, runtimes, episodes, encoder, constant_lambda=1, kappa=0.001):
    results = Parallel(n_jobs=-1)(delayed(eval_AC_per_run)(env, runtime, runtimes, episodes, critic_type, learner_type, gamma, alpha, beta, eta, encoder, constant_lambda, kappa) for runtime in range(runtimes))
    return np.concatenate(results, axis=0)