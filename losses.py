import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.tf_util import make_session

mapping = {}


def register(loss_fn):
    mapping[loss_fn.__name__] = loss_fn
    return loss_fn


def get_loss(loss_name):
    return mapping[loss_name]


def entropy_2d(probs):
    """
    :param probs: [batch * height * width * channels].
    Assumes that each channel is a probability distribution.
    :return: [batch * channels]. Entropy for each channel.
    """
    return tf.einsum('bhwc,bhwc->bc', tf.log(probs), probs)


@register
def attention_entropy(get_tensor: bool = False, input_tensor=None):
    if get_tensor:
        sess = tf.get_default_session()
        if sess is None:
            sess = make_session()
        a2 = sess.graph.get_tensor_by_name('train_model/model/a2_1:0')
        return a2

    a2_entropy = entropy_2d(input_tensor)
    attn_entropy = tf.reduce_sum(a2_entropy, -1)
    attn_entropy = tf.reduce_mean(attn_entropy)
    return attn_entropy


