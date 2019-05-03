import tensorflow as tf

from utils import get_session

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
    probs = probs + tf.keras.backend.epsilon()
    return -tf.einsum('bhwc,bhwc->bc', tf.log(probs), probs)


@register
def attention_entropy():
    def return_tensor():
        attn = get_session().graph.get_tensor_by_name('train_model/model/attn:0')
        attn_entropy = entropy_2d(attn)
        attn_entropy = tf.reduce_sum(attn_entropy, axis=-1)
        attn_entropy = tf.reduce_mean(attn_entropy)
        return attn_entropy

    return return_tensor
