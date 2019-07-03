import numpy as np
from joblib import Parallel, delayed
from utils import *

# TODO: this file hasn't been tested. Also, I am concerned about $\lambda^{(t+1)}$ being unused!

class TOTD_LEARNER():
    def __init__(self, env, D):
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.observation_space.D = D # dimension after encoding
        self.w_curr, self.w_prev = np.zeros(self.observation_space.D), np.zeros(self.observation_space.D)
        self.refresh()

    def learn(self, R_next, gamma_next, gamma_curr, x_next, x_curr, lambda_next, lambda_curr, rho_curr, alpha_curr):
        self.rho_curr = rho_curr
        self.w_next, self.e_curr = \
            self.true_online_td_step(R_next, gamma_next, gamma_curr, x_next, x_curr, self.w_curr, self.w_prev,
         lambda_next, lambda_curr, self.rho_curr, self.rho_prev,
         self.e_prev,
         alpha_curr)
        pass
        
    def next(self):
        self.w_curr, self.w_prev = np.copy(self.w_next), np.copy(self.w_curr)
        self.e_prev = np.copy(self.e_curr)
        self.rho_prev = np.copy(self.rho_curr)
        del self.w_next, self.e_curr, self.rho_curr

    def refresh(self):
        self.e_curr, self.e_prev = np.zeros(self.observation_space.D), np.zeros(self.observation_space.D)
        self.rho_prev = 1

    @staticmethod
    def true_online_td_step(R_next, gamma_next, gamma_curr, x_next, x_curr, w_curr, w_prev, lambda_next, lambda_curr, rho_curr, rho_prev, e_prev, alpha_curr):
        # TODO: double-check, rho_prev, lambda_next not used!
        delta_curr = R_next + gamma_next * np.dot(x_next, w_curr) - np.dot(x_curr, w_curr)
        e_curr = rho_curr * (gamma_curr * lambda_curr * e_prev + alpha_curr * (1 - rho_curr * gamma_curr * lambda_curr * np.dot(x_curr, e_prev)) * x_curr)
        w_next = w_curr + delta_curr * e_curr + alpha_curr * (np.dot(x_curr, w_prev) - np.dot(x_curr, w_curr)) * x_next
        return w_next, e_curr

def true_online_td(env, episodes, target, behavior, evaluate, Lambda, encoder, gamma = lambda x: 0.95, alpha = 0.05):
    """
    episodes:   number of episodes
    target:     target policy matrix (|S|*|A|)
    behavior:   behavior policy matrix (|S|*|A|)
    Lambda:     LAMBDA object determining each lambda for each feature (or state or observation)
    gamma:      anonymous function determining each lambda for each feature (or state or observation)
    alpha:      learning rate for the weight vector of the values
    """
    D = encoder(0).size
    learner = TOTD_LEARNER(env, D)
    value_trace = np.empty((episodes, 1)); value_trace[:] = np.nan
    for episode in range(episodes):
        o_curr, done = env.reset(), False
        x_curr = encoder(o_curr)
        learner.refresh()
        value_trace[episode, 0] = evaluate(learner.w_curr, 'expectation')
        while not done:
            action = decide(o_curr, behavior)
            rho_curr = importance_sampling_ratio(target, behavior, o_curr, action)
            o_next, r_next, done, _ = env.step(action)
            x_next = encoder(o_next)
            learner.learn(r_next, gamma(x_next), gamma(x_curr), x_next, x_curr, Lambda.value(x_next), Lambda.value(x_curr), rho_curr, alpha)
            learner.next()
            x_curr = x_next
    return value_trace

def eval_totd_per_run(env, runtime, runtimes, episodes, target, behavior, gamma, Lambda, alpha, evaluate, encoder):
    print('running %d of %d for totd' % (runtime + 1, runtimes))
    value_trace = true_online_td(env, episodes, target, behavior, evaluate, Lambda, encoder, gamma=gamma, alpha=alpha)
    return value_trace.T

def eval_totd(env, behavior, target, Lambda, gamma, alpha, runtimes, episodes, evaluate, encoder):
    results = Parallel(n_jobs = -1)(delayed(eval_totd_per_run)(env, runtime, runtimes, episodes, target, behavior, gamma, Lambda, alpha, evaluate, encoder) for runtime in range(runtimes))
    results = np.concatenate(results, axis=0)
    return results