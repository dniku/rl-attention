import json
import logging
import os
from pathlib import Path

import tensorflow as tf
from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TRPO
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.vec_env import VecNormalize

try:
    from stable_baselines.common import set_global_seeds
except ImportError:
    import warnings


    def set_global_seeds(_):
        warnings.warn(
            "Seed can't be fixed, set_global_seeds doesn't exist. "
            "Please use this version: https://github.com/RerRayne/stable-baselines"
        )


def set_model_seed(model, seed):
    if hasattr(model.env, 'seed'):
        model.env.seed(seed)
    else:
        model.env.env_method("seed", seed)

    return model


# All available training algorithms
ALGOS_DICT = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'ppo1': PPO1,
    'ppo2': PPO2,
    'sac': SAC,
    'trpo': TRPO,
}


def main():
    with open('config.json', 'r') as fp:
        cfg = json.load(fp)

    # Setting log levels to cut out minor errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Create and wrap the environment
    logging.info('Starting {env_name}'.format(**cfg))
    env = make_atari_env(env_id=cfg['env_name'], num_env=1, seed=cfg['train_seed'])
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env)

    # Setting all known random seeds (Python, Numpy, TF, Gym if available)
    set_global_seeds(cfg['train_seed'])

    logging.info('Running {algo}'.format(**cfg))
    model = ALGOS_DICT[cfg['algo']](
        policy=cfg['policy_type'],
        env=env,
        tensorboard_log=cfg['log_dir']
    )
    model.verbose = 1

    logging.info('Training for {time_steps} steps'.format(**cfg))

    # Training
    model.learn(total_timesteps=cfg['time_steps'], log_interval=cfg['log_interval'])

    # Saving
    logging.info('Saving model and metrics')

    base_name = '{env_name}-{algo}-{policy_type}'.format(**cfg)

    save_dir = Path('/tmp/rl-attention')
    model.save(str(save_dir / base_name))

    with (save_dir / (base_name + '.txt')).open('w+') as f:
        f.write('%s\n' % json.dumps(cfg))

    if cfg['enjoy']:
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
