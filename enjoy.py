import argparse
import json
import logging
from pathlib import Path

from skimage.transform import resize
import numpy as np
import tensorflow as tf
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize
from tqdm.auto import tqdm

from algos import get_algo
from losses import get_loss
from models import get_network_builder
from visualization import VideoWriter, render_attn


def main(cfg, model_path, env_name, eval_seed):
    set_global_seeds(eval_seed)

    env = make_atari_env(env_name, num_env=9, seed=eval_seed)
    env = VecFrameStack(env, n_stack=4)  # stack 4 frames
    if cfg['normalize']:
        # Not setting training=False because that seems to ruin performance
        env = VecNormalize(env)

    model = get_algo(cfg['algo']).load(
        str(model_path),
        env,
        verbose=1,
        learning_rate=lambda frac: 0.00025 * frac,
        attn_loss=get_loss(cfg['attn_loss'])(),
        attn_coef=cfg['attn_coef'],
        policy_kwargs={
            'cnn_extractor': get_network_builder(cfg['network'])
        },
    )

    human_obs = True
    observations = []
    attention = []

    attn_tensor = model.sess.graph.get_tensor_by_name('model/attn:0')
    attn_tensor = tf.reduce_sum(attn_tensor, axis=-1)

    obs = env.reset()
    for _ in tqdm(range(1000), desc='playing'):
        action, _states, attn = model.predict(obs, extra=attn_tensor)

        if human_obs:
            stored_obs = np.stack(env.get_images()) / 255
        else:
            stored_obs = obs[:, :, :, -1].copy()

        # mn, mx = stored_obs.min(), stored_obs.max()
        # assert 0 <= mn and mx <= 1, (mn, mx)

        observations.append(stored_obs)
        attention.append(attn)

        obs, rewards, dones, info = env.step(action)

        env.render()

    attention = render_attn(attention, 36, 8, 0)
    attn_max = max(attn.max() for attn in attention)
    for attn in attention:
        attn /= attn_max

    with VideoWriter(Path('/tmp/rl-attention/foo.mp4')) as writer:
        for obs, attn in tqdm(zip(observations, attention), desc='writing video', total=len(observations)):
            if human_obs:
                b, h, w = obs.shape[:-1]
                assert obs.shape[-1] == 3
                resized_attn = np.stack([resize(attn[bb, ...], (h, w)) for bb in range(b)])
                frame = 0.5 * (obs + resized_attn[..., np.newaxis])
            else:
                frame = np.stack([
                    np.zeros_like(obs),
                    attn,
                    obs.astype(np.float32)  # / 255
                ], axis=-1)
            writer.write_frame(frame)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)-15s] %(levelname)s: %(message)s'
    )

    with open('config.json', 'r') as fp:
        cfg = json.load(fp)

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default=cfg['env_name'],
                        help='Gym Atari environment name')
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Path to model Pickle file')
    parser.add_argument('--eval-seed', type=int, default=1000,
                        help='Random seed used in evaluation environment')
    args = parser.parse_args()

    main(cfg, args.model_path, args.env_name, args.eval_seed)
