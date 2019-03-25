# TODO put on group github and add how to run docs

import numpy as np
import random
import os
import time
import tensorflow as tf

from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.vec_env import VecFrameStack
from models import a2c, acer, acktr, deepq, ddpg, ppo1, ppo2, sac, trpo

# Basic parameters
env_name = 'PongNoFrameskip-v4'
time_steps = 1000
policy_type = 'CnnPolicy'

# Making policies callable
FUNC_DICT = {
    'a2c': lambda e: a2c.A2C(policy=policy_type, env=e),
    'acer': lambda e: acer.ACER(policy=policy_type, env=e),
    'acktr': lambda e: acktr.ACKTR(policy=policy_type, env=e),
    'dqn': lambda e: deepq.DQN(policy=policy_type, env=e),
    'ddpg': lambda e: ddpg.DDPG(policy=policy_type, env=e),
    'ppo1': lambda e: ppo1.PPO1(policy=policy_type, env=e),
    'ppo2': lambda e: ppo2.PPO2(policy=policy_type, env=e),
    'sac': lambda e: sac.SAC(policy=policy_type, env=e),
    'trpo': lambda e: trpo.TRPO(policy=policy_type, env=e)
}

# Setting log levels to cut out minor errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Create and wrap the environment
print('Starting', env_name)
env = make_atari_env(env_id=env_name, num_env=1, seed=0)
env = VecFrameStack(env, n_stack=4)
env = VecNormalize(env)


# Setting all known random seeds
random.seed(0)
np.random.seed(0)
set_global_seeds(0)
tf.random.set_random_seed(0)

# Setting up model of your choice
while True:
    try:
        print("Available models: a2c, acer, acktr, dqn, ddpg, ppo1, ppo2, sac, trpo")
        model_name = input("Choose model: ")
        model = FUNC_DICT[model_name](env)
        model.verbose = 1
        print('Running ', model_name)
        break
    except KeyError:
        print('Not a recognised model')

while True:
    try:
        train_time = int(input("Choose training time (s): "))
        print('Training for %d seconds' % train_time)
        break
    except ValueError:
        print('Not a number')

# Loading file if available


# Training for chosen length of time
start_time = time.time()
while (time.time() - start_time) < train_time:
    model.learn(total_timesteps=200, log_interval=100)

# Saving
print('Saving model')
model.save('./saved_models/%s-%s-%s' % (env_name, model_name, policy_type))

# Displaying gameplay
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
