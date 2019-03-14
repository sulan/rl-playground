import json, time, sys
import numpy as np

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.utils import CustomObjectScope

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from config_parser import ConfigParser
from gomoku_conv import GomokuConv, GomokuProcessor
from callbacks import TrainingStatisticsLogger
from a2c import A2C
from ppo import PPOLearner

from kaiki_model import create_kaiki_model


CONFIG = ConfigParser('./config.json')
NUM_STEPS = CONFIG.getOption('num_steps', 100)
# Wether to use the verbose output in keras-rl
VERBOSE_TRAINING = CONFIG.getOption('verbose_training', 0)
OUTPUT_DATA_FILE = CONFIG.getOption('output_data_file', 'train.out.hdf5')
# If not None, the Runner will load this file at agent creation
INPUT_MODEL = CONFIG.getOption('input_model', None)


class PrettyPrintEncoder(json.JSONEncoder):
    def default(self, o):
        return str(o)


class Configurable:
    def __init__(self, default):
        self.config = dict(default)

    def print_config(self):
        return json.dumps(self.config, sort_keys = True, indent = 4,
                          cls = PrettyPrintEncoder)



class Runner(Configurable):
    DEFAULT_PARAMS = {
        'algorithm' : 'DQN',
        'algorithm_params' : {
            },
        'env_ctor_params' : {
            },
        'gamma' : 0.99,
        'measurement_name' : 'default',
        'model_type' : 'dense',
        'num_steps' : NUM_STEPS,
        'optimizer' : Adam(),
        'seed' : int(time.time() * 10000),
        }

    DQN_DEFAULT_PARAMS = {
        'double_dqn' : False,
        'epsilon' : 0.3,
        'target_model_update' : 10e-3,
        }
    PPO_DEFAULT_PARAMS = {
        'clipping_epsilon' : 0.2,
        'entropy_coeff' : 0.01,
        'fit_epochs' : 1,
        'num_actors' : 1,
        'lambda' : 1,
        'trajectory_length' : 1,
        'vfloss_coeff' : 0.5,
        }

    def __init__(self, env_cls):
        super().__init__(Runner.DEFAULT_PARAMS)
        self.agent = None
        self.model = None
        self.env = None
        self.env_cls = env_cls

    def _createModel(self):
        if self.config['model_type'] == 'dense':
            shared = Sequential([
                Flatten(input_shape = (1,) + self.env_cls.NUM_SENSORS),
                Dense(20, activation='relu'),
                Dense(20, activation='relu'),
                ])
            if self.config['algorithm'] == 'PPO':
                policy = Dense(
                    self.env_cls.NUM_ACTIONS, activation = 'softmax')(
                        shared.output)
                value = Dense(1, activation = 'linear')(shared.output)
                model = Model(inputs = shared.inputs, outputs = [policy, value])
            else:
                shared.add(
                    Dense(self.env_cls.NUM_ACTIONS, activation = 'linear'))
                model = shared
        elif self.config['model_type'] == 'gomoku':
            assert self.config['algorithm'] == 'DQN', \
                'Kaiki doesn''t support A2C yet.'
            model = create_kaiki_model(self.env_cls.NUM_SENSORS)
        else:
            raise ValueError(self.config['model_type'])
        return model

    def _getTrainPolicy(self):
        config = self.config['algorithm_params']['epsilon']
        if isinstance(config, (float, int)):
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

    def _set_default_params(self):
        assert self.config['algorithm'] in ['DQN', 'PPO'], \
            'Unsupported RL algorithm: ' + self.config['algorithm']
        if self.config['algorithm'] == 'DQN':
            default_params = Runner.DQN_DEFAULT_PARAMS
        elif self.config['algorithm'] == 'PPO':
            default_params = Runner.PPO_DEFAULT_PARAMS
        for k, v in default_params.items():
            if k not in self.config['algorithm_params']:
                self.config['algorithm_params'][k] = v

    def createAgent(self):
        if INPUT_MODEL is not None:
            with CustomObjectScope({'GomokuConv' : GomokuConv}):
                self.model = load_model(INPUT_MODEL)
        else:
            self.model = self._createModel()

        self._set_default_params()
        if self.config['algorithm'] in ['DQN',]:
            memory = SequentialMemory(limit = 50000, window_length = 1)
            test_policy = EpsGreedyQPolicy(eps = 0)
            processor = GomokuProcessor() \
                if self.config['model_type'] == 'gomoku' else None

            with CustomObjectScope({'GomokuConv' : GomokuConv}):
                params = self.config['algorithm_params']
                self.agent = DQNAgent(
                    model = self.model,
                    nb_actions = self.env_cls.NUM_ACTIONS,
                    memory = memory,
                    nb_steps_warmup = 50,
                    target_model_update = params['target_model_update'],
                    gamma = self.config['gamma'],
                    policy = self._getTrainPolicy(),
                    test_policy = test_policy,
                    processor = processor,
                    enable_double_dqn = params['double_dqn'])
                self.agent.compile(self.config['optimizer'], metrics = ['mae'])
            self.env = self.env_cls(**self.config['env_ctor_params'])
        else:
            with CustomObjectScope({'GomokuConv' : GomokuConv}):
                params = self.config['algorithm_params']
                learner = PPOLearner(
                    self.model,
                    trajectory_length = params['trajectory_length'],
                    clipping_epsilon = params['clipping_epsilon'],
                    gamma = self.config['gamma'],
                    lam = params['lambda'],
                    vfloss_coeff = params['vfloss_coeff'],
                    entropy_coeff = params['entropy_coeff'],
                    fit_epochs = params['fit_epochs']
                    )
                self.agent = A2C(learner = learner,
                                 num_actors = params['num_actors'],)
                self.agent.compile(self.config['optimizer'])
            # TODO incorporate env index
            self.env = lambda _: self.env_cls(**self.config['env_ctor_params'])

    def fit(self, num_steps = None):
        assert self.agent is not None, "createAgent() should be run before fit"
        if num_steps is None:
            num_steps = self.config['num_steps']

        print('Starting training with configuration:')
        print(self.print_config())
        print('Real number of steps is: {}'.format(num_steps))

        try:
            loss_per_update = self.config['algorithm_params']['fit_epochs']
            num_updates = np.ceil(num_steps /
                                  self.config['algorithm_params']['num_actors'])
            # +1 for the last call (if nb_steps isn't evenly divisible by
            # num_actors)
            num_updates += 1
            if self.config['algorithm'] == 'PPO':
                # PPO has 3 losses
                loss_per_update *= 3
        except KeyError:
            loss_per_update = 1
            num_updates = num_steps
        callbacks = [
            TrainingStatisticsLogger(OUTPUT_DATA_FILE,
                                     self.config['measurement_name'],
                                     num_updates, self.env,
                                     loss_per_update = loss_per_update),
            ]

        start_time = time.monotonic()
        if isinstance(self.agent, A2C):
            # No verbose training support yet
            self.agent.fit(self.env, num_steps, callbacks = callbacks)
        else:
            self.env.reset()
            self.agent.fit(self.env, num_steps, verbose = VERBOSE_TRAINING,
                           callbacks = callbacks)
        duration = time.monotonic() - start_time

        print('Training completed. It took {} seconds, {} per step.'
              .format(duration, duration / num_steps))

    def test(self, num_episodes = 1):
        assert self.agent is not None, "createAgent() should be run before test"
        if isinstance(self.agent, A2C):
            # Visualisation and verbosity are not yet supported
            history = self.agent.test(self.env, nb_episodes = num_episodes,
                                      visualize = True,
                                      nb_max_episode_steps = 1000)
        else:
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
    runner = Runner(GomokuEnvironment)
    runner.config['epsilon'] = (0.3, 0., 500000)
    runner.config['model_type'] = 'gomoku'
    runner.config['env_ctor_params'] = {
        'opponents' : [
            'easy',
            'medium',
            'hard',
            ],
        'opponent_distribution' : [
            0.99,
            0.01,
            0,
            ],
        }
    runner.createAgent()
    runner.fit(num_epoch)
    m, v = runner.test()
    print('Test result: {} (+/- {})'.format(m, v))
    runner.model.save('trained_model.hdf5')


if __name__ == "__main__":
    main()
