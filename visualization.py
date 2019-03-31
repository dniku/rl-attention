from pathlib import Path

import numpy as np
import tensorflow as tf
from stable_baselines.common.tf_util import make_session
from stable_baselines.common.tile_images import tile_images
from gym.wrappers.monitoring.video_recorder import ImageEncoder


def render_attn(attn: np.ndarray, kernel_size: int, stride: int, padding: int):
    # attn: batch * time * height * width

    b, t, h, w = attn.shape

    tensor = tf.transpose(tf.convert_to_tensor(attn, dtype=tf.float32), [0, 2, 3, 1])
    flt = tf.ones(shape=(kernel_size, kernel_size, 1, 1), dtype=tf.float32)

    output_shape = (
        1,
        kernel_size + (h - 1) * stride - 2 * padding,
        kernel_size + (w - 1) * stride - 2 * padding,
        1,
    )

    channelwise_ops = []

    for bb in range(b):
        for tt in range(t):
            layer_1hw1 = tf.nn.conv2d_transpose(
                tensor[bb, :, :, tt][None, ..., None],
                flt,
                output_shape=output_shape,
                strides=(1, stride, stride, 1),
                padding=('SAME' if padding > 0 else 'VALID'),
                data_format='NHWC',
            )
            channelwise_ops.append(tf.squeeze(layer_1hw1, axis=[0, 3]))

    channelwise_results = make_session(make_default=False).run(channelwise_ops)

    rendered_attn = np.empty((
        b, t,
        kernel_size + (h - 1) * stride - 2 * padding,
        kernel_size + (w - 1) * stride - 2 * padding,
    ))

    i = 0
    for bb in range(b):
        for tt in range(t):
            rendered_attn[bb, tt, :, :] = channelwise_results[i]
            i += 1

    rendered_attn = rendered_attn.clip(min=0)

    rendered_attn_max = rendered_attn.max()
    if not np.allclose(rendered_attn_max, 0):
        rendered_attn /= rendered_attn_max

    return rendered_attn


def render_obs_with_attn(obs: np.ndarray, attn: np.ndarray,
                         kernel_size: int, stride: int, padding: int,
                         last_frame_only: bool = True):
    # obs: batch * time * num_frames * height * width
    # attn: batch * time * height * width

    if last_frame_only:
        obs = np.expand_dims(obs[:, :, -1, :, :], axis=2)

    b, t, f, h, w = obs.shape
    ha = ((h + 2 * padding) - kernel_size) // stride + 1
    wa = ((w + 2 * padding) - kernel_size) // stride + 1
    assert attn.shape == (b, t, ha, wa), (obs.shape, attn.shape, (b, t, ha, wa))

    obs /= 255

    attn = render_attn(attn, kernel_size, stride, padding)

    c = 3
    video = np.zeros((b, t, c, f, h, w), dtype=np.float32)

    video[:, :, 2, :, :, :] = obs

    for i in range(f):
        video[:, :, 1, i, :, :] = attn

    # noinspection PyArgumentList
    return video.reshape(b, t, c, f * h, w)


def save_video(video: np.ndarray, path: Path, fps: int = 30):
    # video: batch * time * channels * height * width
    video = np.einsum('btchw->tbhwc', video)

    image_encoder = None
    for frame in video:
        frame = tile_images(frame)
        frame = (frame * 255).astype(np.uint8)
        if image_encoder is None:
            image_encoder = ImageEncoder(
                output_path=str(path),
                frame_shape=frame.shape,
                frames_per_sec=fps,
            )
        image_encoder.capture_frame(frame)
