from keras.models import Model
from keras.layers import MaxPooling2D, Activation, Flatten, Conv2DTranspose, Conv2D, Reshape, BatchNormalization, merge, Input
from gomoku_conv import GomokuConv


def create_kaiki_model(state_shape):
    inputs = Input(shape = (1,) + state_shape)
    reshape = Reshape(target_shape = state_shape)(inputs)
    s = GomokuConv(filters = 128, kernel_size = 9, use_bias = False) (reshape)

    c1 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (s)
    b1 = BatchNormalization(c1, axis=1)
    a1 = Activation(b1, activation='relu')
    p1 = MaxPooling2D((2, 2)) (a1)

    c2 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (a1)
    b2 = BatchNormalization(c2, axis=1)
    a2 = Activation(b2, activation='relu')
    p2 = MaxPooling2D((2, 2)) (a2)

    s3 = merge.concatenate([p1, p2])
    c3 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (s3)
    b3 = BatchNormalization(c3, axis=1)
    a3 = Activation(b3, activation='relu')
    p3 = MaxPooling2D((2, 2)) (a3)

    s4 = merge.concatenate([p2, c3])
    c4 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (s4)
    b4 = BatchNormalization(c4, axis=1)
    a4 = Activation(b4, activation='relu')
    p4 = MaxPooling2D((2, 2)) (a4)

    s5 = merge.concatenate([p3, p4])
    c5 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (s5)
    b5 = BatchNormalization(c5, axis=1)
    a5 = Activation(b5, activation='relu')
    t5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (a5)

    s6 = merge.concatenate([p4, a5])
    c6 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (s6)
    b6 = BatchNormalization(c6, axis=1)
    a6 = Activation(b6, activation='relu')
    t6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (a6)

    s7 = merge.concatenate([c4, t5, t6])
    c7 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (s7)
    b7 = BatchNormalization(c7, axis=1)
    a7 = Activation(b7, activation='relu')
    t7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (a7)

    s8 = merge.concatenate([t6, a7])
    c8 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (s8)
    b8 = BatchNormalization(c8, axis=1)
    a8 = Activation(b8, activation='relu')
    t8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (a8)

    s9 = merge.concatenate([a2, t7, t8])
    c9 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (s9)
    b9 = BatchNormalization(c9, axis=1)
    a9 = Activation(b9, activation='relu')

    s10 = merge.concatenate([a1, t8, a9])
    c10 = Conv2D(128, (3, 3), padding='same', data_format = 'channels_first', use_bias = False) (s10)
    b10 = BatchNormalization(c10, axis=1)
    a10 = Activation(b10, activation='relu')

    s11 = merge.concatenate([s, a10])
    outputs = Conv2D(1, (1, 1), padding='same', data_format = 'channels_first', use_bias = False) (s11)
    outputs = Flatten()(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
