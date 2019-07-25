from gym.envs.registration import register

register(
    id='MountainCar-v1',
    entry_point='mountaincar.MountainCar:MountainCarEnv',
)