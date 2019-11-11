import numpy as np
from joblib import Parallel, delayed
from utils import *
from numba import jit

# TODO: this file hasn't been tested. Also, I am concerned about $\lambda^{(t+1)}$ being unused!

class TOTD_LEARNER():
    def __init__(self, env, D):
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.observation_space.D = D # dimension after encoding
        self.w_curr, self.w_prev = np.zeros(self.observation_space.D), np.zeros(self.observation_space.D)
        self.e_curr, self.e_prev = np.zeros(self.observation_space.D), np.zeros(self.observation_space.D)
        self.rho_prev = 1

    def refresh(self):
        self.e_curr[:], self.e_prev[:] = 0.0, 0.0
        self.rho_prev = 1

    def learn(self, r_next, done, gamma_next, gamma_curr, x_next, x_curr, lambda_next, lambda_curr, rho_curr, alpha_curr):
        self.rho_curr = rho_curr
        self.w_next, self.e_curr = \
            self.true_online_td_step(r_next, done, gamma_next, gamma_curr, x_next, x_curr, self.w_curr, self.w_prev,
         lambda_next, lambda_curr, self.rho_curr, self.rho_prev,
         self.e_prev,
         alpha_curr)
        
    def next(self):
        self.w_curr, self.w_prev = np.copy(self.w_next), np.copy(self.w_curr)
        self.e_prev = np.copy(self.e_curr)
        self.rho_prev = np.copy(self.rho_curr)
        del self.w_next

    @staticmethod
    @jit(nopython=True, cache=True)
    def true_online_td_step(r_next, done, gamma_next, gamma_curr, x_next, x_curr, w_curr, w_prev, lambda_next, lambda_curr, rho_curr, rho_prev, e_prev, alpha_curr):
        # TODO: double-check, rho_prev, lambda_next not used!
        # True Online Temporal-Difference Learning - Harm van Seijen et al.
        delta_curr = r_next + (not done) * gamma_next * np.dot(w_curr, x_next) - np.dot(w_prev, x_curr)
        # TODO: check things about the second $\rho$
        e_curr = rho_curr * (gamma_curr * lambda_curr * e_prev + alpha_curr * x_curr - alpha_curr * rho_curr * gamma_curr * lambda_curr * np.dot(x_curr, e_prev) * x_curr)
        w_next = w_curr + delta_curr * e_curr - alpha_curr * (np.dot(w_curr, x_curr) - np.dot(w_prev, x_curr)) * x_curr
        return w_next, e_curr

def totd(env, steps, target, behavior, evaluate, Lambda, encoder, gamma=lambda x: 0.95, alpha=0.05):
    """
    steps:   number of steps
    target:     target policy matrix (|S|*|A|)
    behavior:   behavior policy matrix (|S|*|A|)
    Lambda:     LAMBDA object determining each lambda for each feature (or state or observation)
    gamma:      anonymous function determining each lambda for each feature (or state or observation)
    alpha:      learning rate for the weight vector of the values
    """
    D = np.size(encoder(env.reset()))
    value_learner = TOTD_LEARNER(env, D)
    value_trace = np.empty(steps // 1000); value_trace[:] = np.nan
    step = 0
    while step < steps:
        o_curr, done = env.reset(), False
        x_curr = encoder(o_curr)
        value_learner.refresh()
        if step % 1000 == 0:
            value_trace[step // 1000] = evaluate(value_learner.w_curr, 'expectation')
        while not done:
            action = decide(o_curr, behavior)
            rho_curr = importance_sampling_ratio(target, behavior, o_curr, action)
            o_next, r_next, done, _ = env.step(action); x_next = encoder(o_next); step += 1
            value_learner.learn(r_next, done, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha)
            value_learner.next()
            x_curr = x_next
    return value_trace

def eval_totd_per_run(env_name, runtime, runtimes, steps, target, behavior, gamma, Lambda, alpha, evaluate, encoder):
    np.random.seed(seed=runtime)
    env = gym.make(env_name)
    env.seed(runtime)
    print('%d of %d for totd(%g), alpha: %g' % (runtime + 1, runtimes, Lambda.value(encoder(0)), alpha))
    value_trace = totd(env, steps, target, behavior, evaluate, Lambda, encoder, gamma=gamma, alpha=alpha)
    return value_trace.reshape(1, -1)

def eval_totd(env_name, behavior, target, Lambda, gamma, alpha, runtimes, steps, evaluate, encoder):
    results = Parallel(n_jobs=-1)(delayed(eval_totd_per_run)(env_name, runtime, runtimes, steps, target, behavior, gamma, Lambda, alpha, evaluate, encoder) for runtime in range(runtimes))
    return np.concatenate(results, axis=0)