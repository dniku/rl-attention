from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from stable_baselines.common.tile_images import tile_images
from tqdm.auto import tqdm

from utils import get_session


def render_attn(attention: List[np.ndarray], kernel_size: int, stride: int, padding: int):
    b, h, w = attention[0].shape

    input_ph = tf.placeholder(tf.float32, attention[0].shape)
    tensor = tf.expand_dims(input_ph, axis=-1)
    flt = tf.ones(shape=(kernel_size, kernel_size, 1, 1), dtype=tf.float32)

    output_shape = (
        1,
        kernel_size + (h - 1) * stride - 2 * padding,
        kernel_size + (w - 1) * stride - 2 * padding,
        1,
    )

    batchwise_ops = []

    for bb in range(b):
        layer_1hw1 = tf.nn.conv2d_transpose(
            tensor[bb, :, :, :][None, ...],
            flt,
            output_shape=output_shape,
            strides=(1, stride, stride, 1),
            padding=('SAME' if padding > 0 else 'VALID'),
            data_format='NHWC',
        )
        batchwise_ops.append(tf.squeeze(layer_1hw1, axis=[0, 3]))

    result = []
    sess = get_session()

    for attn in tqdm(attention, postfix='rendering attn', ncols=76):
        batchwise_results = sess.run(batchwise_ops, {input_ph: attn})

        rendered_attn = np.empty((
            b,
            kernel_size + (h - 1) * stride - 2 * padding,
            kernel_size + (w - 1) * stride - 2 * padding,
        ))

        for bb in range(b):
            rendered_attn[bb, :, :] = batchwise_results[bb]

        rendered_attn = rendered_attn.clip(min=0)
        result.append(rendered_attn)

    return result


class VideoWriter:
    def __init__(self, path: Path, fps: int = 30):
        self.path = path
        self.fps = fps
        self.image_encoder = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.image_encoder is not None:
            self.image_encoder.close()

    def write_frame(self, frame):
        """
        :param frame: batch * height * width * channels, dtype=float, ranging from 0 to 1
        :return: None
        """
        frame = tile_images(frame)
        frame = (frame * 255).astype(np.uint8)
        if self.image_encoder is None:
            self.image_encoder = ImageEncoder(
                output_path=str(self.path),
                frame_shape=frame.shape,
                frames_per_sec=self.fps
            )
        self.image_encoder.capture_frame(frame)
