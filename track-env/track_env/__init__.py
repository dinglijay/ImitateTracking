from gym.envs.registration import register

register(
    id='track-v0',
    entry_point='track_env.envs:TrackEnv',
)