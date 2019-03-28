import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.vec_env import VecNormalize

from models import a2c, acer, acktr, deepq, ddpg, ppo1, ppo2, sac, trpo


def main():
    with open('config.json', 'r') as fp:
        cfg = json.load(fp)

    # Making policies callable
    FUNC_DICT = {
        'a2c': a2c.A2C,
        'acer': acer.ACER,
        'acktr': acktr.ACKTR,
        'dqn': deepq.DQN,
        'ddpg': ddpg.DDPG,
        'ppo1': ppo1.PPO1,
        'ppo2': ppo2.PPO2,
        'sac': sac.SAC,
        'trpo': trpo.TRPO,
    }

    # Setting log levels to cut out minor errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Create and wrap the environment
    logging.info('Starting {env_name}'.format(**cfg))
    env = make_atari_env(env_id=cfg['env_name'], num_env=1, seed=cfg['train_seed'])
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env)

    # Setting all known random seeds
    random.seed(cfg['train_seed'])
    np.random.seed(cfg['train_seed'])
    set_global_seeds(cfg['train_seed'])
    tf.random.set_random_seed(cfg['train_seed'])

    logging.info('Running {algo}'.format(**cfg))
    model = FUNC_DICT[cfg['algo']](policy=cfg['policy_type'], env=env)
    model.verbose = 1

    if cfg['train_time'] == 0:
        logging.info('Training for unlimited time')
    else:
        logging.info('Limiting train time to {train_time} seconds'.format(**cfg))

    # Loading file if available

    # Training for chosen length of time

    start_time = time.time()
    while cfg['train_time'] == 0 or (time.time() - start_time) < cfg['train_time']:
        model.learn(total_timesteps=200, log_interval=100)

    # Saving
    logging.info('Saving model')
    model.save(str(Path(cfg['model_save_dir']).expanduser() / '{env_name}-{algo}-{policy_type}'.format(**cfg)))

    # Displaying gameplay
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)-15s] %(levelname)s: %(message)s'
    )
    main()
