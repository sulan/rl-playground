import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm
import joblib

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
import keras.backend as K

from sklearn.neighbors import BallTree

INPUT_DIM = 2
OUTPUT_DIM = 2
N = 2000
N_test = 200
TRAIN_EPOCHS = 30
SAME_ARCHITECTURE = True # or a more complex one

#  Dummy tqdm {{{ #
class tqdm:
    @classmethod
    def tqdm(*args, **kwargs):
        class t:
            def __init__(self):
                self.progress = 0
            def update(self):
                self.progress += 1
                print(self.progress)
        return t()
#  }}} Dummy tqdm #

#  Init {{{1 #
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
#  1}}} #

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

(X, Y), (X_test, Y_test) = mnist.load_data('/tmp/mnist.npz')
# Add channel count
X.shape += (1,)
X = X.astype(np.float32) / 255
X_test.shape += (1,)
X_test = X_test.astype(np.float32) / 255
activation = 'relu'

def train_classifier(classifier):
    print('Training classifier...')
    Y_ = to_categorical(Y, 10)
    classifier.fit(X.reshape((-1, 28, 28, 1)),
                   {
                       'y_pred': Y_,
                       'features': np.zeros((len(X), 0))},
                   epochs = 25000,
                   validation_split = 0.1,
                   callbacks = [EarlyStopping(patience = 5)],
                   verbose = 1)
    Y_test_ = to_categorical(Y_test, 10)
    score = classifier.evaluate(
        X_test.reshape((-1, 28, 28, 1)),
        {'y_pred': Y_test_, 'features': np.zeros((len(X_test), 0))},
        verbose = 0)
    print('Test loss:', score)
    Y_test_pred = classifier.predict(X_test.reshape((-1, 28, 28, 1)),
                                     verbose = 0)[0]
    Y_test_pred = np.argmax(Y_test_pred, axis = -1)
    print('Test accuracy:', np.mean(Y_test == Y_test_pred))

def gen_feature_extractor():
    return Sequential([
        Conv2D(20, 3, data_format = 'channels_last', input_shape = (28, 28, 1),
               activation = activation),
        Conv2D(20, 3, data_format = 'channels_last', activation = activation),
        MaxPooling2D(pool_size = (4, 4), data_format = 'channels_last'),
        Flatten(),
        ], name = 'features')

def pretrain():
    feature_extractor = gen_feature_extractor()
    inputs = Input(shape = (28, 28, 1))
    features = feature_extractor(inputs)
    x = Dense(20, activation = activation)(features)
    x = Dense(10, activation = 'softmax', name = 'y_pred')(x)
    def sparsity_loss(_, y_pred):
        # L2
        # return K.sum(K.square(y_pred), axis = -1)
        # L1
        return K.sum(K.abs(y_pred), axis = -1)
    classifier = Model(inputs, [x, features])
    classifier.summary()
    classifier.compile(optimizer = Adam(0.0002),
                       loss = {
                           'y_pred': 'categorical_crossentropy',
                           'features': sparsity_loss},
                       loss_weights = {
                           'y_pred': 1,
                           'features': 0.01})
    train_classifier(classifier)
    return feature_extractor

def gen_target(feature_space_size = None):
    """
    # Arguments:
    feature_space_size (int): size of the feature space. If None, then
        assumes it is the pixel space and adds a feature extractor layer (2
        conv and a maxpool).
    """
    target = Sequential()
    if not feature_space_size:
        target.add(gen_feature_extractor())
        assert K.int_shape(target.outputs[0])[1:] == (720,)
    else:
        # Approximately the same number of parameters
        # target.add(Dense((3820 + 720*20) // 740, input_shape = (720,),
        # target.add(Dense((3820 + 720*20) // 40, input_shape = (20,),
        target.add(Dense((3820 + 720*20) // (feature_space_size + 20),
                         input_shape = (feature_space_size,),
                         activation = activation))
    target.add(Dense(20, activation = activation))
    target.add(Dense(20, activation = activation))
    target.add(Dense(OUTPUT_DIM, activation = 'linear'))
    target.compile('sgd', 'mse')
    return target

#  novelty_vs_loss {{{2 #
def novelty_vs_loss(base_class, target_class, num_experiments = 50):
    print('Novelty vs loss, base {} target{}'.format(base_class, target_class))
    num_proportions = 5
    # 0th axis: shared vs not shared vs autoencoded
    # 2nd axis: target, every other class: min/max; base
    loss = np.zeros((3, num_proportions, 4, num_experiments))
    target_nums = np.logspace(0, np.log10(5000), num_proportions, dtype = 'i')
    X_base = X[Y == base_class]
    X_base_feat = X_feat[Y == base_class]
    X_base_encoded = X_encoded[Y == base_class]
    input_bases = [X_base, X_base_feat, X_base_encoded]
    X_target = X[Y == target_class]
    X_target_feat = X_feat[Y == target_class]
    X_target_encoded = X_encoded[Y == target_class]
    input_targets = [X_target, X_target_feat, X_target_encoded]
    input_tests = [X_test, X_test_feat, X_test_encoded]
    other_classes = [i for i in range(10)
                     if i not in (base_class, target_class)]
    other_class_inds_test = [(Y_test == i).nonzero() for i in other_classes]
    def one_measurement(test_no, feature_space):
        result = np.zeros((num_proportions, 4))
        for i, num_target in enumerate(target_nums):
            input_base = input_bases[feature_space]
            input_target = input_targets[feature_space]
            feature_space_size = None if feature_space == 0 \
                                      else input_base.shape[1]
            target_model = gen_target(feature_space_size)
            predictor = gen_target(feature_space_size)
            target_model.summary()
            num_base = 5000
            inds_base = np.random.choice(input_base.shape[0], size = num_base,
                                         replace = False)
            inds_target = np.random.choice(input_target.shape[0],
                                           size = num_target,
                                           replace = False)
            train_x = np.r_[input_base[inds_base], input_target[inds_target]]
            train_y = target_model.predict(train_x)
            predictor.fit(train_x, train_y, epochs = 100,
                          verbose = test_no == 0,
                          callbacks = [EarlyStopping(monitor = 'loss')])

            input_test = input_tests[feature_space]
            test_target_vals = target_model.predict(input_test)
            test_predict_vals = predictor.predict(input_test)
            scores = (test_target_vals - test_predict_vals)**2
            score_base = np.mean(scores[Y_test == base_class])
            score_target = np.mean(scores[Y_test == target_class])
            score_others = [np.mean(scores[inds])
                            for inds in other_class_inds_test]
            result[i, 0] = score_target
            result[i, 1] = min(score_others)
            result[i, 2] = max(score_others)
            result[i, 3] = score_base
        return result

    backend = 'loky'
    r = joblib.Parallel(n_jobs = 4, backend = backend, verbose = 51)(
        joblib.delayed(one_measurement)(test_no, 0)
        for test_no in range(num_experiments))
    loss[0, :, :, :] = np.transpose(r, axes = (1, 2, 0))
    r = joblib.Parallel(n_jobs = 4, backend = backend, verbose = 51)(
        joblib.delayed(one_measurement)(test_no, 1)
        for test_no in range(num_experiments))
    loss[1, :, :, :] = np.transpose(r, axes = (1, 2, 0))
    r = joblib.Parallel(n_jobs = 4, backend = backend, verbose = 51)(
        joblib.delayed(one_measurement)(test_no, 1)
        for test_no in range(num_experiments))
    loss[2, :, :, :] = np.transpose(r, axes = (1, 2, 0))

    return target_nums, loss
#  2}}} #

shared = pretrain()
X_feat = shared.predict(X)
X_test_feat = shared.predict(X_test)
X_encoded = encoder.predict(X)
X_test_encoded = encoder.predict(X_test)
target_nums, loss = novelty_vs_loss(base_class = 0, target_class = 1,
                                    num_experiments = 10)
def plot_medians(target_nums, loss):
    loss_mean = np.median(loss, axis = -1)
    plt.figure()
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.plot(target_nums, loss_mean[i, :, 0], label = 'loss of target')
        plt.plot(target_nums, loss_mean[i, :, 1], label = 'loss of others (min)')
        plt.plot(target_nums, loss_mean[i, :, 2], label = 'loss of others (max)')
        plt.plot(target_nums, loss_mean[i, :, 3], label = 'loss of base')
        plt.title(['Not shared', 'Shared'][i])
        plt.legend()
        plt.xscale('log')
        plt.xlabel('# of target samples')
    plt.savefig('/tmp/loss.pdf')

def plot_differences(target_nums, loss):
    plt.figure(figsize = (15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        l = loss[i, :, :, :] - loss[i, :, 3, :].reshape(
            (len(target_nums), 1, -1))
        med = np.median(l, axis = -1)
        err = np.var(l, axis = -1)**0.5
        plt.plot(target_nums, med[:, 0], color = 'b', label = 'loss of target')
        plt.fill_between(target_nums, med[:, 0] - err[:, 0], med[:, 0] + err[:, 0], facecolor = 'b', alpha = 0.5)
        plt.plot(target_nums, med[:, 1], color = 'orange', label = 'loss of others (min)')
        plt.fill_between(target_nums, med[:, 1] - err[:, 1], med[:, 1] + err[:, 1], facecolor = 'orange', alpha = 0.5)
        plt.plot(target_nums, med[:, 2], color = 'g', label = 'loss of others (max)')
        plt.fill_between(target_nums, med[:, 2] - err[:, 2], med[:, 2] + err[:, 2], facecolor = 'g', alpha = 0.5)
        plt.title(['Not shared', 'Pretrained', 'Autoencoded'][i])
        plt.legend()
        plt.xscale('log')
        plt.xlabel('# of target samples')
    plt.tight_layout()
    plt.savefig('/tmp/loss.pdf')

# }}} MNSIT experiment (from RND paper) #

#  KNN novelty vs loss {{{ #
def KNN_novelty(base_class, target_class, num_experiments = 1, k = 10):
    num_proportions = 10
    # 1st axis: target - zero, every other class - zero: min/max
    loss = np.zeros((num_proportions, 3, num_experiments))
    target_nums = np.logspace(0, np.log10(5000), num_proportions, dtype = 'i')
    num_base = 5000
    X_base = X[Y == base_class]
    X_target = X[Y == target_class]
    other_classes = [i for i in range(10)
                     if i not in (base_class, target_class)]
    other_class_inds_test = [(Y_test == i).nonzero() for i in other_classes]
    progbar = tqdm.tqdm(total = num_experiments * num_proportions,
                        desc = 'Progress',
                        bar_format = '{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]')
    def one_measurement(test_no):
        for i, num_target in enumerate(target_nums):
            progbar.update()
            inds_base = np.random.choice(X_base.shape[0], size = num_base,
                                         replace = False)
            inds_target = np.random.choice(X_target.shape[0], size = num_target,
                                           replace = False)
            train_x = np.r_[X_base[inds_base], X_target[inds_target]]
            train_x.shape = (-1, 28*28)
            tree = BallTree(train_x)

            dists, _ = tree.query(X_test.reshape(-1, 28*28), k = k)
            scores = np.mean(dists, axis = -1)
            score_base = np.mean(scores[Y_test == base_class])
            score_target = np.mean(scores[Y_test == target_class])
            score_others = [np.mean(scores[inds])
                            for inds in other_class_inds_test]
            loss[i, 0, test_no] = score_target - score_base
            loss[i, 1, test_no] = min(score_others) - score_base
            loss[i, 2, test_no] = max(score_others) - score_base

    for test_no in range(num_experiments):
        one_measurement(test_no)

    return target_nums, loss

target_nums, loss = KNN_novelty(base_class = 1, target_class = 0,
                                num_experiments = 1)
loss_mean = np.mean(loss, axis = -1)
plt.figure()
plt.plot(target_nums, loss_mean[:, 0], label = 'novelty of target')
plt.plot(target_nums, loss_mean[:, 1], label = 'novelty of others (min)')
plt.plot(target_nums, loss_mean[:, 2], label = 'novelty of others (max)')
plt.legend()
plt.xscale('log')
plt.xlabel('# of target samples')
plt.savefig('loss.pdf')
#  }}} KNN novelty vs loss #

#  Novelty vs autoencoder feature loss {{{ #
def novelty_vs_autoencoder_loss(base_class, target_class, encoder,
                                num_experiments = 50):
    num_proportions = 10
    # 1st axis: target - zero, every other class - zero: min/max
    loss = np.zeros((num_proportions, 3, num_experiments))
    target_nums = np.logspace(0, np.log10(5000), num_proportions, dtype = 'i')
    X_base = X[Y == base_class]
    X_target = X[Y == target_class]
    other_classes = [i for i in range(10)
                     if i not in (base_class, target_class)]
    other_class_inds_test = [(Y_test == i).nonzero() for i in other_classes]
    progbar = tqdm.tqdm(total = num_experiments * num_proportions,
                        desc = 'Progress',
                        bar_format = '{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]')
    def one_measurement(test_no):
        for i, num_target in enumerate(target_nums):
            progbar.update()
            _, target_model = gen_target(encoder)
            predictor = gen_predictor(encoder)
            num_base = 5000
            inds_base = np.random.choice(X_base.shape[0], size = num_base,
                                         replace = False)
            inds_target = np.random.choice(X_target.shape[0], size = num_target,
                                           replace = False)
            train_x = np.r_[X_base[inds_base], X_target[inds_target]]
            train_y = target_model.predict(train_x)
            predictor.fit(train_x, train_y, verbose = 0)

            test_target_vals = target_model.predict(X_test)
            test_predict_vals = predictor.predict(X_test)
            scores = (test_target_vals - test_predict_vals)**2
            score_base = np.mean(scores[Y_test == base_class])
            score_target = np.mean(scores[Y_test == target_class])
            score_others = [np.mean(scores[inds])
                            for inds in other_class_inds_test]
            loss[i, 0, test_no] = score_target - score_base
            loss[i, 1, test_no] = min(score_others) - score_base
            loss[i, 2, test_no] = max(score_others) - score_base

    for test_no in range(num_experiments):
        one_measurement(test_no)

    return target_nums, loss

# FIXME freeze encoder
target_nums, loss = novelty_vs_autoencoder_loss(base_class = 0,
                                                target_class = 1,
                                                encoder = encoder,
                                                num_experiments = 20)
loss_mean = np.mean(loss, axis = -1)
loss_var = np.var(loss, axis = -1)
plt.figure()
plt.plot(target_nums, loss_mean[:, 0], label = 'loss of target')
plt.plot(target_nums, loss_mean[:, 1], label = 'loss of others (min)')
plt.plot(target_nums, loss_mean[:, 2], label = 'loss of others (max)')
plt.plot(target_nums, loss_mean[:, 0] + loss_var[:, 0]**0.5, ':',
         label = 'var of target')
plt.plot(target_nums, loss_mean[:, 0] - loss_var[:, 0]**0.5, ':',
         label = 'var of target')
plt.legend()
plt.xscale('log')
plt.xlabel('# of target samples')
plt.savefig('loss.pdf')
#  }}} Novelty vs autoencoder feature loss #

# vim:set et sw=4 ts=4 fdm=marker:
