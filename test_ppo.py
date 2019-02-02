import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten

from dm_env import DumbMars1DEnvironment

from a2c import A2C
from ppo import PPOLearner


def get_model():
    shared = Sequential([
        Flatten(input_shape = (1,) + DumbMars1DEnvironment.NUM_SENSORS),
        Dense(20, activation = 'relu'),
        ])
    policy = Dense(DumbMars1DEnvironment.NUM_ACTIONS, activation = 'softmax')(
        shared.output)
    value = Dense(1, activation = 'linear')(shared.output)
    return Model(inputs = shared.inputs, outputs = [policy, value])

def main():
    model = get_model()
    learner = PPOLearner(model, 10, fit_epochs = 10)
    agent = A2C(learner)
    agent.compile(optimizer = 'sgd')
    def env_factory(_):
        return DumbMars1DEnvironment(2)
    agent.fit(env_factory, 10)

if __name__ == "__main__":
    main()
