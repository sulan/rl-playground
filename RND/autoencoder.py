import numpy as np
import matplotlib.pyplot as plt
import tqdm

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.layers import Conv2DTranspose, Activation, Reshape
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K

#  MNIST data {{{1 #
(X, Y), (X_test, Y_test) = mnist.load_data('/tmp/mnist.npz')
# Add channel count
X.shape += (1,)
X = X.astype('float32') / 255
X_test.shape += (1,)
X_test = X_test.astype('float32') / 255
#  1}}} #

#  Model {{{ #
# From the Keras `mnist_denoising_autoencoder.py' example
def get_model():
    kernel_size = 3
    strides = 2
    activation = 'relu'
    latent_dim = 20
    filters = [32, 64]
    inputs = Input(shape = (28, 28, 1), name = 'encoder_input')
    x = inputs
    x = Conv2D(filters = filters[0], kernel_size = kernel_size,
               strides = strides,
               activation = activation, padding = 'same')(x)
    x = Conv2D(filters = filters[1], kernel_size = kernel_size,
               strides = strides,
               activation = activation, padding = 'same')(x)
    shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, name = 'latent_vector')(x)
    encoder = Model(inputs = inputs, outputs = latent, name = 'encoder')
    latent_inputs = Input((latent_dim,), name = 'decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape(shape[1:])(x)
    # x = Conv2DTranspose(filters = filters[1], kernel_size = kernel_size,
    #                     strides = strides, activation = activation,
    #                     padding = 'same')(x)
    x = Conv2DTranspose(filters = filters[0], kernel_size = kernel_size,
                        strides = strides, activation = activation,
                        padding = 'same')(x)
    x = Conv2DTranspose(filters = 1, kernel_size = kernel_size,
                        strides = strides,
                        padding = 'same')(x)
    outputs = Activation('sigmoid', name = 'decoder_output')(x)
    decoder = Model(inputs = latent_inputs, outputs = outputs, name = 'decoder')
    autoencoder = Model(inputs, decoder(encoder(inputs)), name = 'autoencoder')
    autoencoder.compile(loss = 'mse', optimizer = Adam(0.0002))
    return encoder, decoder, autoencoder
encoder, decoder, autoencoder = get_model()
#  }}} Model #

#  Fit {{{ #
autoencoder.fit(X, X, epochs = 100,
                validation_data = (X_test, X_test),
                callbacks = [EarlyStopping(patience = 5)])
#  }}} Fit #

#  Plot input/output {{{ #
def plot_img(x):
    x = (np.squeeze(x) * 255).astype(np.uint8)
    plt.imshow(x, interpolation = 'none', cmap = 'gray')
reconstructed = autoencoder.predict(X[:10,:,:,:])
latent = encoder.predict(X[:10, :, :, :])
plt.figure()
for i in range(10):
    plt.subplot(10, 2, 2 * i + 1)
    plt.axis('off')
    plot_img(X[i])
    plt.subplot(10, 2, 2 * i + 2)
    plt.axis('off')
    plot_img(reconstructed[i])
    print(Y[i], latent[i])
plt.tight_layout()
#  }}} Plot input/output #

#  Generate samples {{{ #
latent = encoder.predict(X[:1, :, :, :])[0]
noises = np.random.randn(10, *latent.shape) * 2
factors = np.linspace(0.1, 5, 10)
latents = np.array([latent + n * f for f in factors for n in noises])
outputs = decoder.predict(latents)
outputs.shape = (10, 10, 28, 28)
imgs = np.concatenate([np.concatenate(o, axis = -1) for o in outputs], axis = 0)
plt.figure()
plt.axis('off')
plot_img(imgs)
#  }}} Generate samples #

# vim:set et sw=4 ts=4 fdm=marker:
