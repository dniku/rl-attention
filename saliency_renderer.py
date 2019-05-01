import tensorflow as tf
import numpy as np

class SaliencyRenderer():
    def __init__(self, callback_locals):
        self.session = callback_locals['self'].sess
        self.graph = callback_locals['self'].graph
        self.env = callback_locals['self'].env

    def get_gradient_target(self, attention_tensor, selection_method):
        """Returns a tensor of shape [batch_size] containing the aggregate attention"""
        if selection_method == 'MAX':
            return tf.nn.max_pool(attention_tensor,
                                  ksize=[1, attention_tensor.shape[1], attention_tensor.shape[2],
                                         attention_tensor.shape[3]],
                                  strides=[1, 1, 1, 1],
                                  padding='VALID')
        elif selection_method == 'SUM_MAX':
            return tf.nn.max_pool(tf.reduce_sum(attention_tensor, axis=[-1], keepdims=True),
                      ksize=[1, attention_tensor.shape[1], attention_tensor.shape[2], 1],
                      strides=[1, 1, 1, 1],
                      padding='VALID')
        elif selection_method == 'MAX_SUM':
            return tf.reduce_sum(tf.nn.max_pool(attention_tensor,
                      ksize=[1, attention_tensor.shape[1], attention_tensor.shape[2], 1],
                      strides=[1, 1, 1, 1],
                      padding='VALID'), axis=[-1])
        elif selection_method == 'SUM':
            return tf.reduce_sum(attention_tensor, axis=[1, 2, 3])

    def get_basic_input_saliency_map(self, input_tensor, input_values, gradient_source_tensor, attention_tensor,
                                     selection_method='MAX_SUM', n_gradient_samples=1, gradient_sigma_spread=0.15,
                                     aggregation_method='SMOOTHGRAD'):
        """
        input_tensor: the uint8 placeholder containing the raw observations (root of the graph)
        input_values: the raw observations
        gradient_source_tensor: the tensor against which to evaluate the gradient (b/c input_tensor is uint8, and doesn't work with tf.gradients)
        attention_tensor: the output tensor
        selection_method: one of 'MAX', 'MAX_SUM', 'SUM_MAX', 'SUM'
        n_gradient_samples: the number of samples taken (1=Simonyan, >1=Smoothgrad)
        gradient_sigma_spread: the variance of the samples taken, scaled by the range of values in the input
        """

        gradient_target = self.get_gradient_target(attention_tensor, selection_method)

        gradient_sigma = gradient_sigma_spread * (np.max(input_values) - np.min(input_values))

        noise_shape = input_values.shape
        noise_shape[0] *= n_gradient_samples
        noise = np.random.normal(0, gradient_sigma, noise_shape)
        noise[0::n_gradient_samples, ...] = 0 # first-batch noise is zero

        # Repeat batches [0 0 0 0 1 1 1 1 2 2 2 2 3 3...]
        repeated_input_values = np.repeat(input_values, axis=0, repeats=n_gradient_samples)

        gradient_values = self.session.run(tf.gradients(gradient_target, gradient_source_tensor), {input_tensor: repeated_input_values + noise})[0] ** 2 # magnitude

        # Now group by channel: for batch 0, create 4 different channel values
        gradient_values = np.reshape(gradient_values, [input_values.shape[0], n_gradient_samples, input_values.shape[1], input_values.shape[2], input_values.shape[3]])

        if aggregation_method == 'SMOOTHGRAD':
            return np.mean(gradient_values, axis=1)
        elif aggregation_method == 'VARGRAD':
            return np.var(gradient_values, axis=1)
        else:
            raise NotImplementedError
