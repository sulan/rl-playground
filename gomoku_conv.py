import numpy as np

import random

from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.base_layer import InputSpec
from keras import initializers, regularizers

from rl.core import Processor

class GomokuConv(Layer):
    """
    A 2D convolutional layer; the only trainable weights in the kernel are the
    diagonals and the middle row/column.
    """

    def __init__(self, filters, kernel_size,
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 **kwargs):
        super().__init__(**kwargs)
        self.num_filters = filters
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        if isinstance(kernel_size, int):
            self.kernel_shape = (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2
            self.kernel_shape = tuple(kernel_size)
        assert self.kernel_shape[0] % 2 == 1, \
                'Even kernel sizes are not supported'
        assert self.kernel_shape[1] % 2 == 1, \
                'Even kernel sizes are not supported'
        assert self.kernel_shape[0] == self.kernel_shape[1], \
                'Non-square kernels are not supported'
        self.input_spec = InputSpec(shape = (None, 2, None, None))
        self.mask = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name = 'kernel',
            shape = self.kernel_shape + (2, self.num_filters),
            initializer = self.kernel_initializer)
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape = (self.num_filters,),
                initializer = self.bias_initializer)
        mask = np.zeros(self.kernel_shape)
        mask[self.kernel_shape[0] // 2, :] = 1
        mask[:, self.kernel_shape[1] // 2] = 1
        np.fill_diagonal(mask, 1)
        mask[range(self.kernel_shape[0]),
             range(self.kernel_shape[1] - 1, -1, -1)] = 1
        mask.shape = self.kernel_shape + (1, 1)
        self.mask = K.constant(mask)
        super().build(input_shape)

    def call(self, inputs):
        real_kernel = self.kernel * self.mask
        outputs = K.conv2d(inputs,
                           real_kernel,
                           padding = 'same',
                           data_format = 'channels_first')
        if self.use_bias:
            outputs = K.bias_add(outputs,
                                 self.bias,
                                 data_format = 'channels_first')
        return outputs

    def compute_output_shape(self, input_shape):
        # input_shape:  (batch_size, num_players (2), board height, board width)
        assert input_shape and len(input_shape) >= 4
        output_shape = list(input_shape)
        output_shape[1] = self.num_filters
        return tuple(output_shape)

    def get_config(self):
        config = {
            'filters': self.num_filters,
            'use_bias': self.use_bias,
            'kernel_size': self.kernel_shape,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GomokuProcessor(Processor):
    """
    Applies a random transformation to the state batch for Gomoku

    The transformations are flips and rotations of 90 degrees, which should not
    change value of the state.

    The batch shape should be: (batch_size, channels, height, width), where
    width == height.
    """

    def process_state_batch(self, batch):
        assert len(batch.shape) == 5, batch.shape
        assert batch.shape[-2] == batch.shape[-1], batch.shape

        # Dummy "window_length" dimension removal
        batch = np.reshape(batch, (batch.shape[0], batch.shape[2], batch.shape[3], batch.shape[4]))

        def transform(state):
            r = random.randrange(8)
            if r == 0:
                # Flip along the horizontal axis
                return state[:, ::-1, :]
            if r == 1:
                # Flip along the vertical axis
                return state[:, :, ::-1]
            if r == 2:
                # Central inversion (rotation of 180 degrees)
                return state[:, ::-1, ::-1]
            transpose = np.transpose(state, (0, 2, 1))
            if r == 3:
                # Transpose (horizontal flip of rotation of 90 degrees)
                return transpose
            if r == 4:
                # Rotation of 90 degrees
                return transpose[::-1, :]
            if r == 5:
                # Rotation of 270 degrees
                return transpose[:, ::-1]
            if r == 6:
                # Vertical flip of rotation of 90 degrees
                return transpose[::-1, ::-1]
            if r == 7:
                return state
            assert False

        return np.array([transform(s) for s in batch])
