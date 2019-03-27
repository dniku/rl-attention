"""

Instructions

To use the launcher, just run main.py.

It will ask you for a choice of model and how long to train it for.

Other important parameters like the Atari environment can be edited near the top under the parameters section.
To add a new model architecture, it would be easiest if you added it to the FUNC_DICT so that it can be called like the
baseline architectures. Be warned that the stable_baselines functions come with lots of built-in functions so you might
want to start by copying one of those in their entirety. We should probably delete everything not necessary for our
purposes for clarity.

The complete trained model is stored in stored models as env_name-model_name-policy_type.pkl

These can then be loaded easily. Saving the metrics had a bug so I've cut that out and will sort it out soon, but since
the full model is saved this isn't super urgent.

"""

import json
import logging
import os
import random
import time

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
    model.save('./saved_models/{env_name}-{algo}-{policy_type}'.format(**cfg))

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
