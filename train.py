import json, time, sys
import numpy as np
import h5py

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import RMSprop, Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
import rl.callbacks

from config_parser import ConfigParser
from dm_env import DumbMars1DEnvironment


CONFIG = ConfigParser('./config.json')
NUM_STEPS = CONFIG.getOption('num_steps', 100)
VERBOSE_TRAINING = CONFIG.getOption('verbose_training', 0)
OUTPUT_DATA_FILE = CONFIG.getOption('output_data_file', 'train.out.hdf5')


class PrettyPrintEncoder(json.JSONEncoder):
    def default(self, o):
        return str(o)


class Configurable:
    def __init__(self, default):
        self.config = dict(default)

    def print_config(self):
        return json.dumps(self.config, sort_keys = True, indent = 4,
                          cls = PrettyPrintEncoder)


class TrainingStatisticsLogger(rl.callbacks.Callback):

    """
    Callback to write statistics for training steps and episodes to a HDF5
    file

    Statistics collected:
        - loss (TD-error in one update)
        - reward at the end of each training episode
        - reward prediction (predicted V value for the starting state after
          each episode)
    """

    def __init__(self, file_name, measurement_name, max_num_steps):
        """
        Opens the file for writing (deletes any current content)
        """
        self.file = h5py.File(file_name, 'a')
        self.group = self.file.create_group(measurement_name)
        self.episode_rewards_size = 8
        self.num_episodes = self.group.create_dataset('num_episodes',
                                                      shape = (1,),
                                                      dtype = 'i8')
        self.episode_rewards = self.group.create_dataset('episode_rewards',
                                                         shape = (self.episode_rewards_size,),
                                                         maxshape = (None,),
                                                         dtype = 'f4')
        self.num_steps = 0
        self.loss = self.group.create_dataset('loss', shape = (max_num_steps,),
                                              dtype = 'f4')
        self.first_observation = None
        self.reward_prediction = self.group.create_dataset('reward_prediction',
                                                           shape = (self.episode_rewards_size,),
                                                           maxshape = (None,),
                                                           dtype = 'f4')

    def on_step_end(self, step, logs):
        # TODO find out which one is the loss in a more intelligent manner
        cur_loss = logs['metrics'][0]
        self.loss[self.num_steps] = cur_loss
        self.num_steps += 1
        if step == 0:
            self.first_observation = np.array(logs['observation'])
            # It needs to be in the right shape for prediction
            self.first_observation.shape = (1, 1, -1)

    def _grow_episode_datasets(self, episode):
        '''Reallocate array if necessary'''
        while self.episode_rewards_size <= episode:
            self.episode_rewards_size *= 2
        self.episode_rewards.resize(self.episode_rewards_size, axis = 0)
        self.reward_prediction.resize(self.episode_rewards_size, axis = 0)


    def on_episode_end(self, episode, logs):
        self._grow_episode_datasets(episode)

        num_episodes = self.num_episodes[0]
        if episode != num_episodes:
            print('Warning: episode number jumped by more than 1:')
            print('  {} -> {}'.format(num_episodes - 1, episode))

        cur_reward = logs['episode_reward']
        self.episode_rewards[episode] = cur_reward
        self.num_episodes[0] = max(episode + 1, num_episodes)

        agent = self.model
        model = agent.model
        cur_reward_prediction = np.max(model.predict(self.first_observation))
        self.reward_prediction[episode] = cur_reward_prediction

    def on_train_end(self, logs):
        self.file.close()


class Runner(Configurable):
    def __init__(self):
        super().__init__({
            'double_dqn' : False,
            'epsilon' : 0.3,
            'gamma' : 0.99,
            'measurement_name' : 'default',
            'num_steps' : NUM_STEPS,
            'optimizer' : Adam(),
            'seed' : int(time.time() * 10000),
            'starting_height' : 20,
            'target_model_update' : 10e-3,
            })
        self.agent = None
        self.model = None
        self.env = None

    @staticmethod
    def _createModel():
        model = Sequential([
            Flatten(input_shape = (1,DumbMars1DEnvironment.NUM_SENSORS,)),
            Dense(20, activation='relu'),
            Dense(20, activation='relu'),
            Dense(DumbMars1DEnvironment.NUM_ACTIONS, activation = 'linear')
            ])
        return model

    def _getTrainPolicy(self):
        config = self.config['epsilon']
        if isinstance(config, float) or isinstance(config, int):
            assert 0 <= config <= 1, 'Epsilon must be in [0, 1]'
            return EpsGreedyQPolicy(eps = config)
        if isinstance(config, tuple):
            assert len(config) == 3, 'Unknown policy configuration'
            return LinearAnnealedPolicy(inner_policy = EpsGreedyQPolicy(),
                                        attr = 'eps',
                                        value_max = config[0],
                                        value_min = config[1],
                                        value_test = None,
                                        nb_steps = config[2])
        return config


    def createAgent(self):
        self.model = Runner._createModel()
        memory = SequentialMemory(limit = 50000, window_length = 1)
        policy = EpsGreedyQPolicy(eps = self.config['epsilon'])
        test_policy = EpsGreedyQPolicy(eps = 0)

        self.agent = DQNAgent(model = self.model,
                              nb_actions = DumbMars1DEnvironment.NUM_ACTIONS,
                              memory = memory,
                              nb_steps_warmup = 50,
                              target_model_update = \
                                  self.config['target_model_update'],
                              gamma = self.config['gamma'],
                              policy = self._getTrainPolicy(),
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

        callbacks = [
            TrainingStatisticsLogger(OUTPUT_DATA_FILE,
                                     self.config['measurement_name'],
                                     num_steps),
            ]

        start_time = time.monotonic()
        self.env.reset()
        self.agent.fit(self.env, num_steps, verbose = VERBOSE_TRAINING,
                       callbacks = callbacks)
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
    runner.config['epsilon'] = (0.3, 0., 100000)
    runner.createAgent()
    runner.fit(num_epoch)
    m, v = runner.test()
    print('Test result: {} (+/- {})'.format(m, v))


if __name__ == "__main__":
    main()
