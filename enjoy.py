import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from skimage.transform import resize
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize
from tqdm.auto import tqdm

from algos import get_algo
from losses import get_loss
from models import get_network_builder
from saliency_renderer import SaliencyRenderer
from visualization import VideoWriter, render_attn


def main(cfg, model_path, video_path, visualization_method, n_gradient_samples, obs_style):
    set_global_seeds(cfg['eval_seed'])

    env = make_atari_env(cfg['env_name'], num_env=1, seed=cfg['eval_seed'])
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

    observations = []
    saliency_maps = []

    input_tensor = model.sess.graph.get_tensor_by_name("input/Ob:0")
    input_cast_tensor = model.sess.graph.get_tensor_by_name("input/Cast:0")
    a2_activations = model.sess.graph.get_tensor_by_name("model/a2/add:0")

    attn_tensor = model.sess.graph.get_tensor_by_name('model/attn:0')
    attn_tensor = tf.reduce_sum(attn_tensor, axis=-1)

    sr = SaliencyRenderer(
        sess=model.sess,
        gradient_source_tensor=input_cast_tensor,
        attention_tensor=a2_activations,
        selection_method='SUM',
    )

    obs = env.reset()
    for _ in tqdm(range(300), postfix='playing', ncols=76):
        if obs_style == 'human':
            stored_obs = np.stack(env.get_images()) / 255
        else:
            stored_obs = obs[:, :, :, -1].copy()
        observations.append(stored_obs)

        if visualization_method == 'conv2d_transpose':
            action, _states, attn = model.predict(obs, extra=attn_tensor)
            saliency_maps.append(attn)
        else:
            action, _states = model.predict(obs)

            smap = sr.get_basic_input_saliency_map(
                input_tensor,
                obs,
                n_gradient_samples=n_gradient_samples,
                gradient_sigma_spread=0.15,
                aggregation_method={
                    'simonyan': None,
                    'smoothgrad': 'smoothgrad',
                    'vargrad': 'vargrad',
                }[visualization_method]
            )[..., -1]

            saliency_maps.append(smap)

        obs, rewards, dones, info = env.step(action)

    if visualization_method == 'conv2d_transpose':
        saliency_maps = render_attn(saliency_maps, 36, 8, 0)

    saliency_cutoff = max(np.percentile(attn, 99) for attn in saliency_maps)
    for smap in saliency_maps:
        smap /= saliency_cutoff
        np.clip(smap, a_min=0, a_max=1, out=smap)

    with VideoWriter(video_path, fps=10) as writer:
        for obs, smap in tqdm(zip(observations, saliency_maps), postfix='writing video', total=len(observations), ncols=76):
            if obs_style == 'human':
                b, h, w = obs.shape[:-1]
                assert obs.shape[-1] == 3
                resized_attn = np.stack([resize(smap[bb, ...], (h, w)) for bb in range(b)])
                frame = 0.5 * (obs + resized_attn[..., np.newaxis])
            else:
                frame = np.stack([
                    np.zeros_like(obs),
                    smap,
                    obs.astype(np.float32)  # / 255
                ], axis=-1)
                frame = resize(frame, (1, 160, 160, 3))
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
    parser.add_argument('--video-path', type=Path, default=Path('/tmp/rl-attention/video.mp4'),
                        help='Path to generated video')
    parser.add_argument('--eval-seed', type=int, default=cfg['eval_seed'],
                        help='Random seed used in evaluation environment')
    parser.add_argument('--visualization-method', type=str, default='conv2d_transpose',
                        choices=['conv2d_transpose', 'simonyan', 'smoothgrad', 'vargrad'],
                        help='Method to turn activations into images')
    parser.add_argument('--n-gradient-samples', type=int, default=50,
                        help='Number of noisy images for SmoothGrad and VarGrad')
    parser.add_argument('--obs-style', type=str, default=['human'],
                        choices=['human', 'processed'],
                        help='Display original large color images or 84x84 grayscale model inputs')
    args = parser.parse_args()

    cfg['env_name'] = args.env_name
    cfg['eval_seed'] = args.eval_seed

    main(
        cfg=cfg,
        model_path=args.model_path,
        video_path=args.video_path,
        visualization_method=args.visualization_method,
        n_gradient_samples=args.n_gradient_samples,
        obs_style=args.obs_style,
    )
