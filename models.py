import tensorflow as tf


def nature_cnn(scaled_images, **kwargs):
    """Set up the CNN's architecture"""

    layer1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', name='c1')(scaled_images)
    layer2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', name='c2')(layer1)
    layer3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', name='c3')(layer2)
    layer4 = tf.keras.layers.Dense(units=512, activation='relu', name='c4')(tf.keras.layers.Flatten()(layer3))

    return layer4


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


def attention_cnn(scaled_images, **kwargs):
    """Nature CNN with region-sensitive module"""

    def softmax_2d(tensor):
        b, h, w, c = tensor.shape
        tensor = tf.reshape(tensor, (-1, h * w, c))
        tensor = tf.nn.softmax(tensor, axis=1)
        tensor = tf.reshape(tensor, (-1, h, w, c))
        return tensor

    c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', name='c1')(scaled_images)
    c2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu', name='c2')(c1)
    c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', name='c3')(c2)
    c3 = tf.keras.backend.l2_normalize(c3, axis=-1)  # TODO Axis correct?

    a1 = tf.keras.layers.Conv2D(filters=512, kernel_size=1, strides=1, activation='elu', name='a1')(c3)
    a2 = softmax_2d(tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1, name='a2')(a1))

    # TODO Multiplications correct?
    c4 = tf.multiply(c3, tf.expand_dims(a2[:, :, :, 0], axis=-1))
    c4 += tf.multiply(c3, tf.expand_dims(a2[:, :, :, 1], axis=-1))

    d = tf.keras.layers.Dense(units=512, activation='relu', name='d')(tf.keras.layers.Flatten()(c4))

    return d


def entropy_2d(probs):
    """
    :param probs: [batch * height * width * channels].
    Assumes that each channel is a probability distribution.
    :return: [batch * channels]. Entropy for each channel.
    """
    return tf.einsum('bhwc,bhwc->bc', tf.log(probs), probs)
