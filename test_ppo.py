import numpy as np
np.random.seed(1)

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten

from dm_env import DumbMars1DEnvironment

from a2c import A2C
from ppo import PPOLearner
# from train import Runner

class DebugProcessor:
    def process_step(self, observation, reward, done, info):
        print('Processor :: step:')
        print(observation, reward, done, info)
        print('==================')
        return observation, reward, done, info

    def process_observation(self, observation):
        print('Processor :: observation:')
        print(observation)
        print('==================')
        return observation

    def process_action(self, action):
        print('Processor :: action:')
        print(action)
        print('==================')
        return action

    def process_reward(self, reward):
        print('Processor :: reward:')
        print(reward)
        print('==================')
        return reward

    def process_info(self, info):
        print('Processor :: info:')
        print(info)
        print('==================')
        return info

    def process_state_batch(self, batch):
        print('Processor :: state_batch:')
        print(batch)
        print('Shape:', batch.shape)
        print('==================')
        return batch

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
    processor = None
    # processor = DebugProcessor()
    learner = PPOLearner(model, 10, fit_epochs = 10,
                         processor = processor)
    agent = A2C(learner, num_actors = 2, processor = processor)
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
    runner.config['algorithm_params'] = {
        'num_actors': 32,
        }
    runner.config['env_ctor_params'] = {
        'height' : 2,
        }
    runner.createAgent()
    runner.fit(1000 * runner.config['algorithm_params']['num_actors'])
    m, v = runner.test()
    print('Test result: {} (+/- {})'.format(m, v))

if __name__ == "__main__":
    test_ppo()
    # test_ppo_runner()
