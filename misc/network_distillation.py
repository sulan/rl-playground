import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

INPUT_DIM = 2
OUTPUT_DIM = 2
N = 2000
N_test = 200
TRAIN_EPOCHS = 30
SAME_ARCHITECTURE = True # or a more complex one

# Target network
activation = 'relu'
target_model = Sequential([
    Dense(100, input_shape = (INPUT_DIM,), activation = activation),
    Dense(100, activation = activation),
    Dense(100, activation = activation),
    Dense(10, activation = activation),
    Dense(OUTPUT_DIM)
    ])
target_model.compile(optimizer = 'sgd', loss = 'mse')
target_model.summary()

# Predictor network
def gen_predictor():
    activation = 'relu'
    predictor = Sequential([
        Dense(100, input_shape = (INPUT_DIM,), activation = activation),
        Dense(100, activation = activation),
        Dense(100, activation = activation),
        Dense(10, activation = activation),
        Dense(OUTPUT_DIM)
        ])
    predictor.compile(optimizer = 'adam', loss = 'mse')
    return predictor

# Datasets
def generate_sample(a, b, n):
    """
    Generates samples from within a ring with radii a and b

    Uses Marsaglia's algorithm
    """
    assert 0 < a < b
    surface_points = np.random.randn(n, INPUT_DIM)
    surface_points = surface_points \
            / np.linalg.norm(surface_points, axis = 1).reshape((-1, 1))
    r = np.random.rand(n, 1)
    r = (b**INPUT_DIM - a**INPUT_DIM) * r + a**INPUT_DIM
    r = r ** (1 / INPUT_DIM)
    return r * surface_points

#  Simple {{{1 #
sigma = 0.50
X_all = np.random.randn(int(N * 1.1), INPUT_DIM) * sigma
training_ind = np.linalg.norm(X_all, axis = 1) <= 1
X = X_all[training_ind, :]
X_test = X_all[np.logical_not(training_ind), :]
Y = target_model.predict(X)
Y_test = target_model.predict(X_test)
print('Training dataset size:', X.shape[0])
print('Test dataset size:    ', X_test.shape[0])

# Plot data
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection = '3d')
ax.plot(X[:,0], X[:,1], Y[:,0], '.')
ax = fig.add_subplot(1, 2, 2, projection = '3d')
ax.plot(X[:,0], X[:,1], Y[:,1], '.')
fig.show()

# Train predictor
history = predictor.fit(X, Y, epochs = TRAIN_EPOCHS)
loss = history.history['loss']
plt.plot(loss)
plt.ylim([0, loss[0]])

# Evaluate predictor
print('Training loss:', predictor.evaluate(X, Y, verbose = 0))
print('Test loss:', predictor.evaluate(X_test, Y_test, verbose = 0))
#  1}}} #

#  Iterative {{{ #
sigma = 1
X = np.zeros((0, INPUT_DIM))
Y = np.zeros((0, OUTPUT_DIM))
N = 100
loss = np.zeros((4, N, 10))
for test_no in range(10):
    predictor = gen_predictor()
    for i in range(N):
        old_sigma = sigma
        sigma *= 1.01
        # Old
        num_new_samples = 3
        old_samples = generate_sample(old_sigma / 2, old_sigma, num_new_samples)
        old_samples_Y = target_model.predict(old_samples)
        # X = np.concatenate((X, old_samples), axis = 0)
        # Y = np.concatenate((Y, old_samples_Y), axis = 0)
        # X = old_samples
        # Y = old_samples_Y
        history = predictor.fit(old_samples, old_samples_Y, verbose = 0, epochs = 1)
        # loss[0, i] = history.history['loss'][0]

        # New
        new_samples = generate_sample(old_sigma, sigma, num_new_samples)
        new_samples_Y = target_model.predict(new_samples)
        X = np.concatenate((X, new_samples), axis = 0)
        Y = np.concatenate((Y, new_samples_Y), axis = 0)
        # X = new_samples
        # Y = new_samples_Y
        history = predictor.fit(new_samples, new_samples_Y, verbose = 0, epochs = 1)
        loss[2, i] = history.history['loss'][0]
        loss[3, i] = predictor.evaluate(new_samples, new_samples_Y, verbose = 0)
        loss[1, i] = predictor.evaluate(X, Y, verbose = 0)
loss = np.mean(loss, axis = -1)
plt.figure()
plt.plot(loss.T)
plt.legend(['old before fit', 'old after fit', 'new before fit', 'new after fit'])

#  }}} Iterative #

#  MNSIT experiment (from RND paper) {{{ #

# Target model
activation = 'relu'
target_model = Sequential([
    Conv2D(100, 3, data_format = 'channels_last', input_shape = (28, 28, 1), activation = activation),
    Conv2D(100, 3, data_format = 'channels_last', activation = activation),
    Conv2D(100, 3, data_format = 'channels_last', activation = activation),
    Conv2D(1, 3, data_format = 'channels_last', activation = activation),
    Dense(OUTPUT_DIM)
    ])
target_model.compile(optimizer = 'sgd', loss = 'mse')
target_model.summary()

# Predictor network
def gen_predictor():
    activation = 'relu'
    predictor = Sequential([
        Conv2D(100, 3, data_format = 'channels_last', input_shape = (28, 28, 1), activation = activation),
        Conv2D(100, 3, data_format = 'channels_last', activation = activation),
        Conv2D(100, 3, data_format = 'channels_last', activation = activation),
        Conv2D(1, 3, data_format = 'channels_last', activation = activation),
        Dense(OUTPUT_DIM)
        ])
    predictor.compile(optimizer = 'adam', loss = 'mse')
    return predictor

# Dataset
(X, Y), (X_test, Y_test) = mnist.load_data('/tmp/mnist.npz')
X_0 = X[Y == 0]
Y_0 = Y[Y == 0]
target_class = 1
X_target = X[Y == target_class]
Y_target = Y[Y == target_class]
N = 5000
num_experiments = 10
num_proportions = 10
output0 = np.array([1, 0])
output1 = np.array([0, 1])
loss = np.zeros((num_proportions, num_experiments))
for test_no in range(num_experiments):
    for i, proportion in enumerate(np.linspace(0, 1, num_proportions)):
        num_0 = int(N * proportion)
        inds_0 = np.random.choice(X_0.shape[0], size = num_0, replace = False)
        inds_target = np.random.choice(X_target.shape[0], size = N - num_0,
                                       replace = False)
        train_x = np.r_[X_0[inds_0], X_target[inds_target]]
        train_x.shape += (1,)
        train_y = target_model.predict(train_x)
        predictor = gen_predictor()
        predictor.fit(train_x, train_y)

        test_x = np.r_[X_test[Y_test == 0], X_test[Y_test == target_class]]
        test_x.shape += (1,)
        test_y = target_model.predict(test_x)
        loss[i, test_no] = predictor.evaluate(test_x, test_y)
plt.plot(np.mean(loss, axis = -1))
#  }}} MNSIT experiment (from RND paper) #

# vim:set et sw=4 ts=4 fdm=marker:

