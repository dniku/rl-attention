import tensorflow as tf
from stable_baselines.common.tf_util import make_session


def get_session():
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(make_default=True)
    return sess
