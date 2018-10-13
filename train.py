import json, time, sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop, Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from config_parser import ConfigParser
from dm_env import DumbMars1DEnvironment

CONFIG = ConfigParser('./config.json')
NUM_STEPS = CONFIG.getOption('num_steps', 100)
VERBOSE_TRAINING = CONFIG.getOption('verbose_training', 0)

class Configurable:
    def __init__(self, default):
        self.config = dict(default)

    def print_config(self):
        return json.dumps(self.config, sort_keys = True, indent = 4)


class Runner(Configurable):
    def __init__(self):
        super().__init__({
            'optimizer' : 'adam',
            'epsilon' : 0.3,
            'num_steps' : NUM_STEPS,
            'target_model_update' : 10e-3,
            'gamma' : 0.99,
            'double_dqn' : False,
            'starting_height' : 10,
            'seed' : int(time.time() * 10000),
            })
        self.agent = None
        self.env = None

    def _createModel(self):
        model = Sequential([
            Flatten(input_shape = (1,DumbMars1DEnvironment.NUM_SENSORS,)),
            Dense(20, activation='relu'),
            Dense(20, activation='relu'),
            Dense(DumbMars1DEnvironment.NUM_ACTIONS, activation = 'linear')
            ])
        return model

    def createAgent(self):
        model = self._createModel()
        memory = SequentialMemory(limit = 50000, window_length = 1)
        policy = EpsGreedyQPolicy(eps = self.config['epsilon'])
        test_policy = EpsGreedyQPolicy(eps = 0)

        self.agent = DQNAgent(model = model,
                              nb_actions = DumbMars1DEnvironment.NUM_ACTIONS,
                              memory = memory,
                              nb_steps_warmup = 50,
                              target_model_update = \
                                  self.config['target_model_update'],
                              gamma = self.config['gamma'],
                              policy = policy,
                              test_policy = test_policy,
                              enable_double_dqn = self.config['double_dqn'])
        self.agent.compile(self.config['optimizer'], metrics = ['mae'])

        self.env = DumbMars1DEnvironment(self.config['starting_height'])

    def fit(self, num_steps = None):
        assert self.agent is not None, "createAgent() should be run before fit"
        if num_steps is None:
            num_steps = self.config['num_steps']

        print('Starting training with configuration:')
        print(self.print_config())
        print('Real number of steps is: {}'.format(num_steps))

        start_time = time.monotonic()
        self.env.reset()
        self.agent.fit(self.env, num_steps, verbose = VERBOSE_TRAINING)
        duration = time.monotonic() - start_time

        print('Training completed. It took {} seconds, {} per step.'
              .format(duration, duration / num_steps))

    def test(self, num_episodes = 1):
        assert self.agent is not None, "createAgent() should be run before test"
        history = self.agent.test(self.env, nb_episodes = num_episodes,
                                  visualize = True, verbose = 2,
                                  nb_max_episode_steps = 1000)
        rewards = np.array(history.history['episode_reward'])
        mean = np.mean(rewards)
        c = rewards - mean
        variance = np.dot(c,c)/c.size

        return mean, variance


def main():
    num_epoch = int(sys.argv[1])
    runner = Runner()
    runner.createAgent()
    runner.fit(num_epoch)
    m, v = runner.test()
    print('Test result: {} (+/- {})'.format(m, v))


if __name__ == "__main__":
    main()
