import numpy as np
import tensorflow as tf


class SaliencyRenderer():
    def __init__(self, sess, gradient_source_tensor, attention_tensor, selection_method='MAX_SUM'):
        self.sess = sess
        gradient_target = self.get_gradient_target(attention_tensor, selection_method)
        self.gradients_graph_node = tf.gradients(gradient_target, gradient_source_tensor)

    @staticmethod
    def get_gradient_target(attention_tensor, selection_method):
        """Returns a tensor of shape [batch_size] containing the aggregate attention"""
        b, h, w, c = attention_tensor.shape
        if selection_method == 'MAX':
            return tf.nn.max_pool(
                attention_tensor,
                ksize=[1, h, w, c],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
        elif selection_method == 'SUM_MAX':
            return tf.nn.max_pool(
                tf.reduce_sum(attention_tensor, axis=-1, keepdims=True),
                ksize=[1, h, w, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
            )
        elif selection_method == 'MAX_SUM':
            return tf.reduce_sum(
                tf.nn.max_pool(
                    attention_tensor,
                    ksize=[1, h, w, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                ),
                axis=-1
            )
        elif selection_method == 'SUM':
            return tf.reduce_sum(attention_tensor, axis=[1, 2, 3])

    def get_basic_input_saliency_map(self, input_tensor, input_values, n_gradient_samples=1,
                                     gradient_sigma_spread=0.15, aggregation_method=None):
        """
        input_tensor: the uint8 placeholder containing the raw observations (root of the graph)
        input_values: the raw observations
        gradient_source_tensor: the tensor against which to evaluate the gradient (b/c input_tensor is uint8, and doesn't work with tf.gradients)
        attention_tensor: the output tensor
        selection_method: one of 'MAX', 'MAX_SUM', 'SUM_MAX', 'SUM'
        n_gradient_samples: the number of samples taken (1=Simonyan, >1=Smoothgrad)
        gradient_sigma_spread: the variance of the samples taken, scaled by the range of values in the input
        """

        assert n_gradient_samples >= 1, "n_gradient_samples should be at least 1"
        if aggregation_method is not None:
            assert n_gradient_samples > 1, "Aggregation methods require n_gradient_samples > 1"

        b, h, w, c = input_values.shape

        if aggregation_method is not None:
            gradient_sigma = gradient_sigma_spread * (np.max(input_values) - np.min(input_values))

            noise = np.random.normal(0, gradient_sigma, size=(b * n_gradient_samples, h, w, c))
            noise[0::n_gradient_samples, ...] = 0  # first-batch noise is zero

            # Repeat batches [0 0 0 0 1 1 1 1 2 2 2 2 3 3...]
            repeated_input_values = np.repeat(input_values, axis=0, repeats=n_gradient_samples)

            backprop_input = repeated_input_values + noise
        else:
            backprop_input = input_values

        gradient_values = self.sess.run(
            self.gradients_graph_node,
            feed_dict={input_tensor: backprop_input},
        )[0] ** 2  # magnitude

        if aggregation_method is not None:
            # Now group by channel: for batch 0, create 4 different channel values
            gradient_values = gradient_values.reshape((b, n_gradient_samples, h, w, c))

            if aggregation_method == 'smoothgrad':
                return np.mean(gradient_values, axis=1)
            elif aggregation_method == 'vargrad':
                return np.var(gradient_values, axis=1)
            else:
                raise NotImplementedError
        else:
            return gradient_values
