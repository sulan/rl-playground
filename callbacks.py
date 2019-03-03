import numpy as np
import h5py

import rl.callbacks
from a2c import A2C
from config_parser import ConfigParser

CONFIG = ConfigParser('./config.json')
# Interval at which test rewards are measured
TEST_REWARD_INTERVAL = CONFIG.getOption('test_reward_interval', 100)

class TrainingStatisticsLogger(rl.callbacks.Callback):

    """
    Callback to write statistics for training steps and episodes to a HDF5
    file

    Statistics collected:
        - loss (TD-error in one update)
        - the step index at the end of each episode
        - reward at the end of each episode
        - test reward at regular intervals
        - reward prediction (predicted V value for the starting state after
          each episode)
    """

    def __init__(self, file_name, measurement_name, max_num_steps, env):
        """
        Opens the file for writing (deletes any current content)
        """
        self.file = h5py.File(file_name, 'w')
        self.group = self.file.create_group(measurement_name)
        # Capacities of the different arrays (we resize them when needed)
        self.episode_dataset_capacity = 8
        self.test_episode_capacity = 8
        self.num_episodes = self.group.create_dataset(
            'num_episodes',
            shape = (1,),
            dtype = 'i8')
        self.episode_ends = self.group.create_dataset(
            'episode_ends',
            shape = (self.episode_dataset_capacity,),
            maxshape = (None,),
            dtype = 'i8')
        self.episode_rewards = self.group.create_dataset(
            'episode_rewards',
            shape = (self.episode_dataset_capacity,),
            maxshape = (None,),
            dtype = 'f4')
        self.test_episode_rewards = self.group.create_dataset(
            'test_episode_rewards',
            shape = (self.test_episode_capacity,),
            maxshape = (None,),
            dtype = 'f4')
        # Number of policy/model updates (size of the loss array)
        self.num_updates = None
        # Progress of the training
        self.percent = 0
        # Maximum number of steps (max size of the loss array)
        self.max_num_updates = max_num_updates
        self.loss = self.group.create_dataset(
            'loss', shape = (max_num_updates,), dtype = 'f4')
        self.first_observation = None
        self.reward_prediction = self.group.create_dataset(
            'reward_prediction',
            shape = (self.episode_dataset_capacity,),
            maxshape = (None,),
            dtype = 'f4')

        self.env = env

    def on_train_end(self):
        if self.model is A2C:
            # We store the first observation for all the actors
            self.first_observation = {}

    def _print_progress(self, new_percent):
        if new_percent > self.percent:
            self.percent = new_percent
            print(self.percent // 10 if self.percent % 10 == 0 else '.',
                  end = '' if self.percent < 100 else '\n',
                  flush = True)

    def on_step_end(self, step, logs):
        # TODO find out which one is the loss in a more intelligent manner
        cur_loss = logs['metrics'][0]
        self.loss[self.num_updates] = cur_loss
        self.num_updates += 1
        new_percent = (self.num_updates * 100) // self.max_num_updates
        self._print_progress(new_percent)
        # Save first state
        if step == 0:
            self.first_observation = np.array(logs['observation'])
            # It needs to be in the right shape for prediction
            self.first_observation.shape = (1, 1) \
                + self.first_observation.shape

    def _grow_episode_datasets(self, episode):
        '''Reallocate array if necessary'''
        while self.episode_dataset_capacity <= episode:
            self.episode_dataset_capacity *= 2
        while self.test_episode_capacity <= episode // TEST_REWARD_INTERVAL:
            self.test_episode_capacity *= 2
        self.episode_rewards.resize(self.episode_dataset_capacity, axis = 0)
        self.episode_ends.resize(self.episode_dataset_capacity, axis = 0)
        self.reward_prediction.resize(self.episode_dataset_capacity, axis = 0)
        self.test_episode_rewards.resize(self.test_episode_capacity, axis = 0)


    def on_episode_end(self, episode, logs):
        # TODO accomodate A2C
        self._grow_episode_datasets(episode)

        num_episodes = self.num_episodes[0]
        if episode != num_episodes:
            print('Warning: episode number jumped by more than 1:')
            print('  {} -> {}'.format(num_episodes - 1, episode))

        cur_reward = logs['episode_reward']
        self.episode_rewards[episode] = cur_reward
        self.num_episodes[0] = max(episode + 1, num_episodes)
        self.episode_ends[episode] = self.num_updates

        agent = self.model
        if (episode + 1) % TEST_REWARD_INTERVAL == 0:
            # Since the test() method has side-effects, we have to save these
            # two variables and restore them after as a hack
            agent_training = agent.training
            agent_step = agent.step
            history = agent.test(
                self.env, nb_episodes = 1, visualize = False, verbose = 0,
                nb_max_episode_steps = 1000)
            agent.training = agent_training
            agent.step = agent_step
            self.test_episode_rewards[episode // TEST_REWARD_INTERVAL] = \
                history.history['episode_reward']

        model = agent.model
        cur_reward_prediction = np.max(model.predict(self.first_observation))
        self.reward_prediction[episode] = cur_reward_prediction

    def on_train_end(self, logs):
        self.file.close()
