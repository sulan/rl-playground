import numpy as np
np.random.seed(1)

from train import Runner
from dm_env import DumbMars1DEnvironment

STARTING_HEIGHT = 10
NUM_STEPS = 10000

ALGORITHM_PARAMS = {
    'DQN': {
        'double_dqn': True,
        'epsilon': 0.3,
    },
    'PPO': {
        'fit_epochs': 1,
        'num_actors': 32,
        'lambda': 0,
        'trajectory_length': 1,
    }
    }

def run(algorithm, starting_height, random_start = False, runner = None):
    print('Algorithm: {}, Starting from {}, {} start'.format(
        algorithm, starting_height, 'random' if random_start else 'fix'))
    measurement_name = '{}{}-{}-{}'.format(
        starting_height, 'random' if random_start else 'fixed', algorithm,
        'new' if runner is None else 'cont.')
    if not runner:
        runner = Runner(DumbMars1DEnvironment)
        runner.config['algorithm'] = algorithm
    else:
        assert algorithm == runner.config['algorithm']
    runner.config['env_ctor_params']['height'] = starting_height
    runner.config['env_ctor_params']['random_start'] = random_start
    runner.config['algorithm_params'] = dict(ALGORITHM_PARAMS[algorithm])
    runner.config['measurement_name'] = measurement_name
    runner.createAgent()
    runner.fit(NUM_STEPS)
    mean, variance = runner.test(num_episodes = 5)
    print('Test result: {} (+/- {})'.format(mean, variance))
    return runner


def main():
    algorithms = ['DQN', 'PPO']
    print('#######################################################################')
    print('#                    Simple, fixed starting height                    #')
    print('#######################################################################')
    for height in [10, 20, 30, 35]:
        for algorithm in algorithms:
            run(algorithm, height, False, None)
    print('#######################################################################')
    print('#             Generalisation from 30 to 35 (random start)             #')
    print('#######################################################################')
    for algorithm in algorithms:
        runner = run(algorithm, 30, True, None)
        run(algorithm, 35, True, runner)

if __name__ == "__main__":
    main()
