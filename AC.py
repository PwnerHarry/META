# TODO: implement actor-critic(\lambda)
# TODO: should be able to have APIs to the policy evaluation methods!
import gym, numpy as np
from utils import *
from greedy import *
from mta import *
from TOGTD import *
from TOTD import *

def actor_critic(env, episodes, target, behavior, evaluate, encoder, critic_type, gamma = lambda x: 0.95, alpha=0.05, beta=0.05):
    # suppose we use exponential softmax on values
    theta, w = np.zeros(env.action_space.n), np.zeros(env.observation_space.n)
    e_theta, e_w = np.zeros(env.action_space.n), np.zeros(env.observation_space.n)
    I = 1
    # TODO: RL Book 2018, pp. 354
    for episode in range(episodes):
        o_curr, done = env.reset(), False
        x_curr = encoder(o_curr)
        while not done:
            # TODO: how do you calculate the behavior?
            action = decide(o_curr, behavior)

    if critic_type == 'togtd':
        pass
    elif critic_type == 'MTA':
        pass
    elif critic_type == 'greedy':
        pass
    
    # TODO: to implement the main thing here.