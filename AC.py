# TODO: implement actor-critic(\lambda)
# TODO: should be able to have APIs to the policy evaluation methods!
import gym, numpy as np
from utils import *
from greedy import *
from mta import *
from TOGTD import *
from TOTD import *

def actor_critic(env, episodes, target, behavior, evaluate, encoder, critic_type, gamma = lambda x: 0.95, alpha=0.05, beta=0.05):
    if critic_type == 'togtd':
        pass
    elif critic_type == 'MTA':
        pass
    elif critic_type == 'greedy':
        pass
    # TODO: to implement the main thing here.