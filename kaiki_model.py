from keras.models import Model
from keras.layers import MaxPooling2D, Activation, Flatten, Conv2DTranspose, Conv2D, Reshape, BatchNormalization, merge, Input
from gomoku_conv import GomokuConv


def create_kaiki_model(state_shape):
    inputs = Input(shape = state_shape)
    s = GomokuConv(filters=128, kernel_size=9, use_bias=True) (inputs)

    #c0 = Conv2D(512, (9, 9), padding='same', data_format='channels_first')(reshape)
    #b0 = BatchNormalization(axis=1)(c0)
    #a0 = Activation(activation='selu')(c0)

    c1 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(s)
    #b1 = BatchNormalization(axis=1)(c1)
    a1 = Activation(activation='selu')(c1)
    p1 = MaxPooling2D((2, 2), data_format='channels_first')(a1)

    c2 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(a1)
    #b2 = BatchNormalization(axis=1)(c2)
    a2 = Activation(activation='selu')(c2)
    p2 = MaxPooling2D((2, 2), data_format='channels_first')(a2)

    s3 = merge.concatenate([p1, p2], axis=1)
    c3 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(s3)
    #b3 = BatchNormalization(axis=1)(c3)
    a3 = Activation(activation='selu')(c3)
    p3 = MaxPooling2D((2, 2), data_format='channels_first')(a3)

    s4 = merge.concatenate([p2, c3], axis=1)
    c4 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(s4)
    #b4 = BatchNormalization(axis=1)(c4)
    a4 = Activation(activation='selu')(c4)
    p4 = MaxPooling2D((2, 2), data_format='channels_first')(a4)

    s5 = merge.concatenate([p3, p4], axis=1)
    c5 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(s5)
    #b5 = BatchNormalization(axis=1)(c5)
    a5 = Activation(activation='selu')(c5)
    t5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', data_format='channels_first')(a5)

    s6 = merge.concatenate([p4, a5], axis=1)
    c6 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(s6)
    #b6 = BatchNormalization(axis=1)(c6)
    a6 = Activation(activation='selu')(c6)
    t6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', data_format='channels_first')(a6)

    s7 = merge.concatenate([c4, t5, t6], axis=1)
    c7 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(s7)
    #b7 = BatchNormalization(axis=1)(c7)
    a7 = Activation(activation='selu')(c7)
    t7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', data_format='channels_first')(a7)

    s8 = merge.concatenate([t6, a7], axis=1)
    c8 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(s8)
    #b8 = BatchNormalization(axis=1)(c8)
    a8 = Activation(activation='selu')(c8)
    t8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', data_format='channels_first')(a8)

    s9 = merge.concatenate([a2, t7, t8], axis=1)
    c9 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(s9)
    #b9 = BatchNormalization(axis=1)(c9)
    a9 = Activation(activation='selu')(c9)

    s10 = merge.concatenate([a1, t8, a9], axis=1)
    c10 = Conv2D(128, (3, 3), padding='same', data_format='channels_first', use_bias=True)(s10)
    #b10 = BatchNormalization(axis=1)(c10)
    a10 = Activation(activation='selu')(c10)

    s11 = merge.concatenate([s, a10], axis=1)
    outputs = Conv2D(1, (1, 1), padding='same', data_format='channels_first', use_bias=True)(s11)
    #outputs = Flatten()(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
