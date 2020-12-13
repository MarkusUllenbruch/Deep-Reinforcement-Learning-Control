from gym.envs.registration import register

register(
    id='InversePendulum-v0',
    entry_point='gym_foo.envs:InversePendulum',
)
