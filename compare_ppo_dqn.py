import numpy as np
np.random.seed(1)

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, Reshape, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.utils import CustomObjectScope

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
import rl.callbacks

from dm_env import DumbMars1DEnvironment

# from train import *
from ppo import *
from a2c import A2C

STARTING_HEIGHT = 30
N = 100
NUM_STEPS = 1000

def get_ppo_model():
    shared = Sequential([
        Flatten(input_shape = (1,) + DumbMars1DEnvironment.NUM_SENSORS),
        Dense(20, activation = 'relu'),
        Dense(20, activation = 'relu'),
        ])
    policy = Dense(DumbMars1DEnvironment.NUM_ACTIONS, activation = 'softmax')(
        shared.output)
    value = Dense(1, activation = 'linear')(shared.output)
    return Model(inputs = shared.inputs, outputs = [policy, value])

def get_dqn_model():
    model = Sequential([
        Flatten(input_shape = (1,) + DumbMars1DEnvironment.NUM_SENSORS),
        Dense(20, activation = 'relu'),
        Dense(20, activation = 'relu'),
        Dense(DumbMars1DEnvironment.NUM_ACTIONS, activation = 'linear'),
        ])
    return model

def train_dqn():
    rewards = []
    for _ in range(N):
        model = get_dqn_model()
        agent = DQNAgent(model = model,
                nb_actions = DumbMars1DEnvironment.NUM_ACTIONS,
                memory = SequentialMemory(limit = 50000, window_length = 1),
                nb_steps_warmup = 32,
                target_model_update = 0.99,
                gamma = 0.99,
                policy = EpsGreedyQPolicy(0.1),
                test_policy = GreedyQPolicy(),
                enable_double_dqn = True)
        agent.compile(optimizer = 'adam')
        env = DumbMars1DEnvironment(STARTING_HEIGHT)
        env.reset()
        agent.fit(env, NUM_STEPS, verbose = False)
        history = agent.test(env, nb_episodes = 1, visualize = False,
                             verbose = 0, nb_max_episode_steps = 100)
        reward = history.history['episode_reward'][0]
        rewards += [reward]
    return rewards

def train_ppo():
    rewards = []
    for _ in range(N):
        model = get_ppo_model()
        learner = PPOLearner(model, trajectory_length = 1, fit_epochs = 3)
        agent = A2C(learner, num_actors = 10)
        agent.compile(optimizer = 'adam')
        def env_factory(_):
            return DumbMars1DEnvironment(STARTING_HEIGHT)
        agent.fit(env_factory, NUM_STEPS)
        history = agent.test(env_factory, 1, nb_max_episode_steps = 100)
        reward = history.history['episode_reward'][0]
        rewards += [reward]
    return rewards

def main():
    print('DQN:')
    print(train_dqn())
    print('PPO:')
    print(train_ppo())

if __name__ == "__main__":
    main()
