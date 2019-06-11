import numpy as np
np.random.seed(1)

from pprint import pprint
import pickle

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam

from gridworld import LabyrinthEnv, replay_game

from a2c import A2C
from ppo import PPOLearner
from callbacks import TrainingStatisticsLogger

def get_model():
    shared = Sequential()
    shared.add(Reshape(LabyrinthEnv.NUM_SENSORS,
                       input_shape = (1,) + LabyrinthEnv.NUM_SENSORS))
    for filters in [10, 10]:
        shared.add(Conv2D(filters = filters,
                          kernel_size = 3,
                          padding = 'same',
                          activation = 'relu',
                          data_format = 'channels_first'))
        shared.add(MaxPooling2D(3, data_format = 'channels_first'))
    shared.add(Flatten())
    shared.add(Dense(10, activation = 'relu'))
    policy = Dense(LabyrinthEnv.NUM_ACTIONS, activation = 'softmax',
                   name = 'policy')(shared.outputs[0])
    value = Dense(1, activation = 'linear', name = 'value')(
        shared.outputs[0])
    return Model(inputs = shared.inputs, outputs = [policy, value])

def test_labyrinth():
    model = get_model()
    learner = PPOLearner(model, 10, gamma = 0.99, lam = 0.9, fit_epochs = 3,
                         entropy_coeff = 0.01, vfloss_coeff = 0.5)
    agent = A2C(learner, num_actors = 32)
    agent.compile(optimizer = Adam(lr = 0.0002))
    envs = []
    def env_factory(_):
        nonlocal envs
        envs.append(LabyrinthEnv(save_replay = True))
        return envs[-1]
    num_updates = 10000
    callbacks = [
        TrainingStatisticsLogger('train.out.hdf5',
                                 'labyrinth',
                                 num_updates + 1, env_factory,
                                 loss_per_update = 3 * 3),
        ]
    history = agent.fit(env_factory, num_updates * 32, callbacks = callbacks)
    # print('Fit history:')
    # pprint(history.history)
    envs = []
    history = agent.test(env_factory, 3, visualize = False)
    print('Test history:')
    pprint(history.history)
    # input('Press enter to see the replay')
    # replay_game(0, envs[-1].replay[-1])
    with open('replays', mode = 'wb') as f:
        pickle.dump(envs[0].replay, f)


if __name__ == "__main__":
    test_labyrinth()
