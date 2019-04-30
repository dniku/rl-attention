import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TRPO
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.logger import configure
from tqdm.auto import tqdm

from losses import get_loss
from models import get_network_builder

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


def make_run_name(cfg):
    run_ts = datetime.now().isoformat(sep='_', timespec='milliseconds').replace(':', '-')
    return '{env_name},{algo},{network},{train_seed},{run_ts}'.format(run_ts=run_ts, **cfg)


def main(cfg, run_dir):
    run_name = make_run_name(cfg)
    output_dir = run_dir / run_name
    output_dir.mkdir(parents=True)

    with (output_dir / 'config.json').open('w') as fp:
        json.dump(cfg, fp, indent=2)

    # Setting log levels to cut out minor errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    log_dir = output_dir / cfg['log_dir']
    tensorboard_dir = output_dir / cfg['tb_dir']

    configure(
        log_dir=str(log_dir),
        format_strs=['log', 'csv', 'tensorboard'],
        tensorboard_dir=str(tensorboard_dir)
    )

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
        verbose=1,
        learning_rate=lambda frac: 0.00025 * frac,
        attn_loss=get_loss(cfg['attn_loss'])(),
        attn_coef=cfg['attn_coef'],
        policy_kwargs={
            'cnn_extractor': get_network_builder(cfg['network'])
        },
        tensorboard_log=str(tensorboard_dir),
    )

    logging.info('Training for {time_steps} steps'.format(**cfg))

    # Training
    model.learn(
        total_timesteps=cfg['time_steps'],
        log_interval=cfg['log_interval'],
        tb_log_name=None,
    )

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=Path('/tmp/rl-attention'),
                        help='Path for directories with per-run outputs')
    args = parser.parse_args()

    with open('config.json', 'r') as fp:
        cfg = json.load(fp)

    main(cfg, args.run_dir)
