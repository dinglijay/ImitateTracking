import argparse
import os.path as osp
import multiprocessing
import sys
import os

import gym
import track_env

import ppo2
from baselines.common.vec_env import VecFrameStack
from baselines.common.cmd_util import make_vec_env
from baselines import bench
from baselines import logger
from baselines.run import get_env_type

from configs import ADNetConf
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
    # parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--env', help='environment ID', default='track-v0')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str, default='Track')
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=1, type=int)
    parser.add_argument('--seed', help='RNG seed', type=int, default=123)
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='track_vggm')
    # Traing Configuration
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=3e6)
    parser.add_argument('--nsteps', help='number of steps of the vectorized environment per update', type=int, default=2048)
    parser.add_argument('--load_path', help='the directory to save log file', default='./checkpoint/ACT1_4.ckpt')#'log/checkpoints/00001')
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



    model = ppo2.learn(
        network=args.network,
        env=env, 
        total_timesteps=args.num_timesteps,
        seed=args.seed, 
        nsteps=args.nsteps,
        log_interval=10,
        save_interval=50,
        load_path=args.load_path

    )




    # env = gym.make(args.env_id)
    # env = bench.Monitor(env, args.log_dir and
    #                     osp.join(args.log_dir, "monitor.json"))

    # learn(network='cnn', env=args.env_id, total_timesteps=args.num_timesteps, seed=args.seed)

    # model, env = train(args)
