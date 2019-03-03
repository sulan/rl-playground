import numpy as np
np.random.seed(1)

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten

from dm_env import DumbMars1DEnvironment

from a2c import A2C
from ppo import PPOLearner
from train import Runner


def get_model():
    shared = Sequential([
        Flatten(input_shape = (1,) + DumbMars1DEnvironment.NUM_SENSORS),
        Dense(20, activation = 'relu'),
        ])
    policy = Dense(DumbMars1DEnvironment.NUM_ACTIONS, activation = 'softmax')(
        shared.output)
    value = Dense(1, activation = 'linear')(shared.output)
    return Model(inputs = shared.inputs, outputs = [policy, value])

def test_ppo():
    model = get_model()
    learner = PPOLearner(model, 10, fit_epochs = 10)
    agent = A2C(learner, num_actors = 2)
    agent.compile(optimizer = 'sgd')
    def env_factory(_):
        return DumbMars1DEnvironment(2)
    agent.fit(env_factory, 10)
    history = agent.test(env_factory, 1)
    print(history.history)
    agent.fit(env_factory, 10000)
    history = agent.test(env_factory, 1)
    print(history.history)

def test_ppo_runner():
    runner = Runner(DumbMars1DEnvironment)
    runner.config['algorithm'] = 'PPO'
    runner.config['env_ctor_params'] = {
        'height' : 2,
        }
    runner.createAgent()
    runner.fit(10)
    m, v = runner.test()
    print('Test result: {} (+/- {})'.format(m, v))

if __name__ == "__main__":
    # test_ppo()
    test_ppo_runner()
