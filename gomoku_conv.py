import numpy as np

from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.base_layer import InputSpec

class GomokuConv(Layer):
    """
    A 2D convolutional layer; the only trainable weights in the kernel are the
    diagonals and the middle row/column.
    """

    def __init__(self, filters,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 **kwargs):
        super().__init__(**kwargs)
        self.num_filters = filters
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        # Kernel size is fixed for now
        kernel_size = 9
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
        mask = np.array([mask] * 2)
        mask.shape = self.kernel_shape + (2, self.num_filters)
        self.mask = K.constant(mask)
        super().build(input_shape)

    def call(self, inputs):
        real_kernel = self.kernel * self.mask
        # Calculate flips and rotations
        # Horizontal flip
        vflip = K.reverse(real_kernel, axes = 0)
        # Vertical flip
        hflip = K.reverse(real_kernel, axes = 1)
        # Central inversion (rotation of 180 degrees)
        cflip = K.reverse(real_kernel, axes = [0, 1])
        # Transpose (horizontal flip of rotation of 90 degrees)
        transpose = K.permute_dimensions(real_kernel, (1, 0, 2, 3))
        # Rotation of 90 degrees
        rotate90 = K.reverse(transpose, axes = 0)
        # Rotation of 270 degrees
        rotate270 = K.reverse(transpose, axes = 1)
        # Vertical flip of rotation of 90 degrees
        rotate90_flip = K.reverse(transpose, axes = [0, 1])
        effective_kernel = K.concatenate([
            real_kernel,
            vflip,
            hflip,
            cflip,
            transpose,
            rotate90,
            rotate270,
            rotate90_flip], axis = -1)
        outputs = K.conv2d(inputs,
                           effective_kernel,
                           padding = 'same',
                           data_format = 'channels_first')
        outputs = K.bias_add(outputs,
                             self.bias,
                             data_format = 'channels_first')
        return outputs

    def compute_output_shape(self, input_shape):
        # input_shape:  (batch_size, num_players (2), board height, board width)
        assert input_shape and len(input_shape) >= 4
        output_shape = list(input_shape)
        output_shape[1] = self.num_filters * 8
        return tuple(output_shape)
