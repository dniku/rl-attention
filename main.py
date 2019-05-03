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
class Callback(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.pbar = None

    def __call__(self, _locals, _globals):
        if self.pbar is None:
            self.pbar = tqdm(total=_locals['nupdates'] * _locals['self'].n_batch)

        self.pbar.update(_locals['self'].n_batch)
        self.pbar.set_postfix_str('{update}/{nupdates} updates'.format(**_locals))

        if _locals['update'] == _locals['nupdates']:
            self.pbar.close()
            self.pbar = None

        if _locals['update'] % 100 == 1 or _locals['update'] == _locals['nupdates']:
            _locals['self'].save(str(self.output_dir / 'model.pkl'))

        return True


def make_run_name(cfg):
    run_ts = datetime.now().isoformat(sep='_', timespec='milliseconds').replace(':', '-')
    return '{env_name},{train_seed},{attn_coef},{run_ts}'.format(run_ts=run_ts, **cfg)


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
    env = make_atari_env(env_id=cfg['env_name'], num_env=8, seed=cfg['train_seed'])
    env = VecFrameStack(env, n_stack=4)
    if cfg['normalize']:
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
        callback=Callback(output_dir),
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)-15s] %(levelname)s: %(message)s'
    )

    with open('config.json', 'r') as fp:
        cfg = json.load(fp)

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=Path('/tmp/rl-attention'),
                        help='Path for directories with per-run outputs')
    parser.add_argument('--train-seed', type=int, default=cfg['train_seed'],
                        help='Random seed applied before training')
    parser.add_argument('--attn-coef', type=float, default=cfg['attn_coef'],
                        help='Coefficient before attention loss')
    args = parser.parse_args()

    cfg.update({
        'train_seed': args.train_seed,
        'attn_coef': args.attn_coef,
    })

    main(cfg, args.run_dir)
