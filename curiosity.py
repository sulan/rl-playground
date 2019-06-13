"""
Implementation of count-based exploration and Random Network Distillation
"""
from collections import Counter
import numpy as np
np.random.seed(1)
import pickle

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import rl.core

from dm_env import DumbMars1DEnvironment
from callbacks import TrainingStatisticsLogger
from a2c import A2C
from ppo import PPOLearner


class CountCuriosityProcessor(rl.core.Processor):
    """
    Count-based exploration via intrinsic rewards
    """
    def __init__(self, scale = 1):
        super().__init__()
        self.state_count = Counter()
        self._intrinsic_rewards = []
        self.scale = scale

    def process_step(self, observation, reward, done, info):
        count = self.state_count[tuple(observation.flatten())]
        self.state_count[tuple(observation.flatten())] += 1
        intrinsic_reward = 1 / (1 + count ** 0.5)
        self._intrinsic_rewards.append(intrinsic_reward)
        reward = reward + self.scale * intrinsic_reward
        return observation, reward, done, info

    @property
    def metrics_names(self):
        return ['intrinsic_reward']

    @property
    def metrics(self):
        intrinsic_rewards = self._intrinsic_rewards
        self._intrinsic_rewards = []
        return [intrinsic_rewards]

def get_model():
    shared = Sequential([
        Flatten(input_shape = (1,) + DumbMars1DEnvironment.NUM_SENSORS),
        Dense(20, activation = 'relu'),
        Dense(20, activation = 'relu'),
        ])
    policy = Dense(DumbMars1DEnvironment.NUM_ACTIONS, activation = 'softmax')(
        shared.output)
    value = Dense(1, activation = 'linear')(shared.output)
    return Model(inputs = shared.inputs, outputs = [policy, value])

def run_count_based(run_baseline = False):
    """
    Run PPO with count-based exploration, or optionally without it (this will
    still count the states for visualisation).

    # Arguments
    run_baseline (boolean): if true, discards intrinsic reward; also changes
                            the output names
    """
    model = get_model()
    processor = CountCuriosityProcessor(0 if run_baseline else 1)
    fit_epochs = 3
    learner = PPOLearner(model, 10, gamma = 0.99, lam = 0.9,
                         fit_epochs = fit_epochs,
                         entropy_coeff = 0.01, vfloss_coeff = 0.5,
                         processor = processor)
    agent = A2C(learner, num_actors = 32, processor = processor)
    agent.compile(optimizer = Adam(0.0002))
    def env_factory(_):
        return DumbMars1DEnvironment(30)
    num_updates = 5000
    measurement_name = 'baseline' if run_baseline else 'count_based'
    callbacks = [
        TrainingStatisticsLogger('train.out.hdf5', measurement_name,
                                 num_updates + 1, env_factory,
                                 loss_per_update = 3 * fit_epochs),
        ]
    agent.fit(env_factory, 32 * num_updates, callbacks = callbacks)
    history = agent.test(env_factory, 1)
    print('Test history:')
    print(history.history)
    # Save state counts
    with open(measurement_name + '.state.counts', 'wb') as f:
        pickle.dump(processor.state_count, f)

if __name__ == "__main__":
    run_count_based(True)
    run_count_based(False)
