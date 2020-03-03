import argparse
import multiprocessing
import sys
import os

import gym
import track_env

import ppo2
from baselines.common.vec_env import VecFrameStack
from baselines.common.cmd_util import make_vec_env
from baselines.common.models import get_network_builder
from baselines import bench
from baselines import logger
from baselines.run import get_env_type

from conf.configs import ADNetConf
import track_policies

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def argsparser():
    parser = argparse.ArgumentParser("Baselines PPO2")
    parser.add_argument('--env',            default='track-v0')
    parser.add_argument('--env_type',       default='Track', type=str)
    parser.add_argument('--num_env',        default=12, type=int)
    parser.add_argument('--seed',           default=123, type=int)
    parser.add_argument('--network',        default='track_2cnn_fc1')
    parser.add_argument('--value_network',  default=None) #'value_cnn_fc12')
    # Traing Configuration
    parser.add_argument('--num_timesteps',  default=4e6, type=int)
    parser.add_argument('--nsteps',         default=256, type=int)
    parser.add_argument('--nminibatches',   default=4, type=int)
    parser.add_argument('--load_path',      default='log/0228_trackCnnFc12/checkpoints/01300')
    parser.add_argument('--log_dir',        default='log/0228_trackCnnFc12+')
    return parser.parse_args()


if __name__ == '__main__':
    args = argsparser()
    ADNetConf.get('conf/dylan.yaml')

    # configure logger, disable logging in child MPI processes (with rank > 0)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(args.log_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        logger.configure(args.log_dir)

    # env_type, env_id = get_env_type(args)
    env_id = args.env
    env_type = 'Track'
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    frame_stack_size = 1
    env = make_vec_env(env_id, env_type, nenv, args.seed)
    env = VecFrameStack(env, frame_stack_size)

    logger.log('Training ppo2 on {}:{} with arguments \n{}'.format(env_type, env_id, args))
    logger.log(ADNetConf.conf.__dict__)

    value_network =get_network_builder(args.value_network) if args.value_network else None
    model = ppo2.learn(
        network=args.network,
        value_network=value_network,
        env=env, 
        total_timesteps=args.num_timesteps,
        seed=args.seed, 
        nsteps=args.nsteps,
        log_interval=10,
        save_interval=50,
        load_path=args.load_path,
        nminibatches=args.nminibatches
    )

    print('Training ended')
    env.close()
