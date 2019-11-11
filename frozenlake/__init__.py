from gym.envs.registration import register

register(
    id='FrozenLake-v1',
    entry_point='frozenlake.FrozenLake:FrozenLakeEnv'
)