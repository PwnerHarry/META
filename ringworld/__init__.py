from gym.envs.registration import register

register(
    id='RingWorld-v0',
    entry_point='ringworld.RingWorld:RingWorldEnv',
)