import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

mapping = {}


def register(network_fn):
    mapping[network_fn.__name__] = network_fn
    return network_fn


def get_network_builder(network_name):
    return mapping[network_name]


@register
def nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


# PyTorch model from Learn to Interpret Atari Agents
# https://github.com/yz93/Learn-to-Interpret-Atari-Agents/blob/master/model.py
#
# class DQN_rs(nn.Module):
#     def __init__(self, args, action_space):
#         super().__init__()
#         self.atoms = args.atoms
#         self.action_space = action_space
#
#         # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#         self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, 3)
#
#         # region-sensitive module (input_chan, output_chan, kernel_size)
#         self.conv1_attent = nn.Conv2d(64, 512, 1)
#         self.conv2_attent = nn.Conv2d(512, 2, 1)
#
#         self.fc_h_v = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
#         self.fc_h_a = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
#         self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
#         self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)
#
#     def forward(self, x, log=False):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.normalize(x, p=2, dim=1)
#         batch_size = x.size(0)
#         weights = F.elu(self.conv1_attent(x))
#         weights = self.conv2_attent(weights).view(-1, 2, 49)
#         weights = F.softmax(weights.view(batch_size * 2, -1), dim=1)  # 2D tensor by default is also dim 1
#         weights = weights.view(batch_size, 2, 7, 7)
#
#         # Broadcasting
#         x1 = x * weights[:, :1, :, :]
#         x2 = x * weights[:, 1:, :, :]
#         x = x1 + x2
#
#         x = x.view(-1, 3136)
#         v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
#         a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
#         v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
#         q = v + a - a.mean(1, keepdim=True)  # Combine streams
#         if log:  # Use log softmax for numerical stability
#             q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
#         else:
#             q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
#         return q  # shape: (-1, self.action_space, self.atoms)


@register
def attention_cnn(scaled_images, **kwargs):
    """Nature CNN with region-sensitive module"""

    def softmax_2d(tensor):
        b, h, w, c = tensor.shape
        tensor = tf.reshape(tensor, (-1, h * w, c))
        tensor = tf.nn.softmax(tensor, axis=1)
        tensor = tf.reshape(tensor, (-1, h, w, c), name='a2')
        return tensor

    c1 = tf.nn.relu(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    c2 = tf.nn.relu(conv(c1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    c3 = tf.nn.relu(conv(c2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    c3 = tf.nn.l2_normalize(c3, axis=-1)

    a1 = tf.nn.elu(conv(c3, 'a1', n_filters=512, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
    a2 = softmax_2d(conv(a1, 'a2', n_filters=2, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
    attn = tf.reduce_sum(a2, axis=-1, keepdims=True, name='attn')

    a2_entropy = entropy_2d(a2)
    attn_entropy = tf.reduce_sum(a2_entropy, -1)
    print(attn_entropy)

    x = c3 * attn

    x = conv_to_fc(x)
    return tf.nn.relu(linear(x, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
