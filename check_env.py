import gym
import track_env
import numpy as np
from stable_baselines.common.env_checker import check_env
from configs import ADNetConf

ADNetConf.get('conf/dylan.yaml')
env= gym.make('track-v0')
obs = env.reset()
ob_space = env.observation_space

check_env(env)

