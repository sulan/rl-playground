import numpy as np
np.random.seed(1)

from keras.optimizers import Adam

from train import Runner
from dm_env import DumbMars1DEnvironment

STARTING_HEIGHTS = [10, 20, 30, 35]
N = 10
NUM_STEPS = 10000

test_no = None

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
    measurement_name = '{}{}-{}-{}-{}'.format(
        starting_height, 'random' if random_start else 'fixed', algorithm,
        'new' if runner is None else 'cont.',
        test_no)
    if not runner:
        runner = Runner(DumbMars1DEnvironment)
        runner.config['algorithm'] = algorithm
    else:
        assert algorithm == runner.config['algorithm']
    runner.config['env_ctor_params']['height'] = starting_height
    runner.config['env_ctor_params']['random_start'] = random_start
    runner.config['algorithm_params'] = dict(ALGORITHM_PARAMS[algorithm])
    runner.config['measurement_name'] = measurement_name
    runner.config['optimizer'] = Adam(lr = 0.0002)
    runner.createAgent()
    runner.fit(NUM_STEPS)
    mean, variance = runner.test(num_episodes = 5)
    print('Test result: {} (+/- {})'.format(mean, variance))
    return runner


def test_once():
    algorithms = ['DQN', 'PPO']
    print('#######################################################################')
    print('#                    Simple, fixed starting height                    #')
    print('#######################################################################')
    for height in STARTING_HEIGHTS:
        for algorithm in algorithms:
            run(algorithm, height, False, None)
    print('#######################################################################')
    print('#             Generalisation from 30 to 35 (random start)             #')
    print('#######################################################################')
    for algorithm in algorithms:
        runner = run(algorithm, 30, True, None)
        run(algorithm, 35, True, runner)

def main():
    global test_no
    for test_no in range(N):
        print('''\
#######################################################################
###########                  Test no.: {:>2}                   ###########
#######################################################################'''\
        .format(test_no))
        test_once()

#  Visualisation {{{1 #

def get_num_tests(keys, measurement_name):
    return len([None for k in keys if k.startswith(measurement_name)])

#  Generalisation {{{2 #
def visualise_generalisation(f):
    import matplotlib.pyplot as plt
    # Generalisation loss
    fig, axs = plt.subplots(2, 2)
    for measurement_template, i in (
            ('35fixed-{}-new-', 0),
            ('35random-{}-cont.-', 1),):
        # DQN
        legend = 'loss'
        measurement_name = measurement_template.format('DQN')
        for t in range(get_num_tests(f.keys(),measurement_name)):
            axs[i, 0].plot(f[measurement_name + str(t)]['loss'],
                           '-', color = (0, 0, 1, 0.1),
                           label = legend)
            legend = '_'
        axs[i, 0].grid(True)
        axs[i, 0].set_title(['DQN baseline',
                             'DQN generalisation (random)'][i])
        axs[i, 0].legend()
        # PPO
        legend_prefix = ''
        measurement_name = measurement_template.format('PPO')
        for t in range(get_num_tests(f.keys(), measurement_name)):
            ppo_loss = f[measurement_name + str(t)]['loss']
            axs[i, 1].plot(ppo_loss[0::3], '-', color = (0, 0, 1, 0.1),
                           label = legend_prefix + 'clip')
            axs[i, 1].plot(ppo_loss[1::3], '-', color = (0, 1, 0, 0.1),
                           label = legend_prefix + 'vf')
            axs[i, 1].plot(ppo_loss[2::3], '-', color = (1, 0, 0, 0.1),
                           label = legend_prefix + 'entropy')
            legend_prefix = '_'
        axs[i, 1].grid(True)
        axs[i, 1].set_title(['PPO baseline',
                             'PPO generalisation (random)'][i])
        axs[i, 1].legend()
    fig.tight_layout()
    plt.show()
    # Generalisation reward prediction error
    fig, axs = plt.subplots(2, 2)
    for measurement_template, i in (
            ('35fixed-{}-new-', 0),
            ('35random-{}-cont.-', 1),):
        # DQN fixed
        legend = 'reward prediction error'
        measurement_name = measurement_template.format('DQN')
        for t in range(get_num_tests(f.keys(),measurement_name)):
            g = f[measurement_name + str(t)]
            num_episodes = g['num_episodes'][0]
            rp_error = g['reward_prediction'][:num_episodes] - \
                g['episode_rewards'][:num_episodes]
            axs[i, 0].plot(rp_error, 'r.', label = legend)
            legend = '_'
        axs[i, 0].grid(True)
        axs[i, 0].set_title(['DQN baseline',
                             'DQN generalisation (random start)'][i])
        axs[i, 0].legend()
        # PPO fixed
        legend = 'reward prediction error'
        measurement_name = measurement_template.format('PPO')
        for t in range(get_num_tests(f.keys(), measurement_name)):
            g = f[measurement_name + str(t)]
            num_episodes = g['num_episodes'][0]
            rp_error = g['reward_prediction'][:num_episodes] - \
                g['episode_rewards'][:num_episodes]
            episode_ends = g['episode_ends'][:num_episodes]
            axs[i, 1].plot(
                episode_ends, rp_error, 'r.',
                label = legend)
            legend = '_'
        axs[i, 1].grid(True)
        axs[i, 1].set_title(['PPO baseline',
                             'PPO generalisation (random start)'][i])
    fig.tight_layout()
    plt.show()
#  2}}} #

def visualise_fixed(f):
    import matplotlib.pyplot as plt
    # FIXME determine this dynamically
    TEST_REWARD_INTERVAL = 100
    # Loss
    fig, axs = plt.subplots(len(STARTING_HEIGHTS), 2)
    for i, starting_height in enumerate(STARTING_HEIGHTS):
        # DQN
        legend = 'loss'
        measurement_name = '{}fixed-DQN-new-'.format(starting_height)
        for t in range(get_num_tests(f.keys(), measurement_name)):
            axs[i, 0].plot(f[measurement_name + str(t)]['loss'],
                           '-', color = (0, 0, 1, 0.1),
                           label = legend)
            legend = '_'
        axs[i, 0].set_ylim([-1, 10])
        axs[i, 0].grid(True)
        axs[i, 0].set_title('DQN started from ' + str(starting_height))
        axs[i, 0].legend()
        # PPO
        legend_prefix = ''
        measurement_name = '{}fixed-PPO-new-'.format(starting_height)
        for t in range(get_num_tests(f.keys(), measurement_name)):
            ppo_loss = f[measurement_name + str(t)]['loss']
            axs[i, 1].plot(ppo_loss[0::3], '-', color = (0, 0, 1, 0.1),
                           label = legend_prefix + 'clip')
            axs[i, 1].plot(ppo_loss[1::3], '-', color = (0, 1, 0, 0.1),
                           label = legend_prefix + 'vf')
            axs[i, 1].plot(ppo_loss[2::3], '-', color = (1, 0, 0, 0.1),
                           label = legend_prefix + 'entropy')
            legend_prefix = '_'
        axs[i, 1].set_ylim([-3, 20])
        axs[i, 1].grid(True)
        axs[i, 1].set_title('PPO started from ' + str(starting_height))
        axs[i, 1].legend()
    fig.tight_layout()
    plt.show()
    # Metrics
    fig, axs = plt.subplots(len(STARTING_HEIGHTS), 2)
    for i, starting_height in enumerate(STARTING_HEIGHTS):
        # DQN
        legend_prefix = ''
        measurement_name = '{}fixed-DQN-new-'.format(starting_height)
        for t in range(get_num_tests(f.keys(), measurement_name)):
            g = f[measurement_name + str(t)]
            num_episodes = g['num_episodes'][0]
            episode_ends = g['episode_ends'][:num_episodes]
            episode_rewards = g['episode_rewards'][:num_episodes]
            # xx = episode_ends + \
            #     np.random.random(episode_ends.shape) * 0.5 - 0.25
            # axs[i, 0].plot(xx, episode_rewards, '.',
            #          label = legend_prefix + 'episode rewards')
            reward_prediction = g['reward_prediction'][:num_episodes]
            xx = episode_ends + \
                np.random.random(episode_ends.shape) * 0.5 - 0.25
            # axs[i, 0].plot(xx, reward_prediction, 'x',
            #          label = legend_prefix + 'reward prediction')
            axs[i, 0].plot(xx, reward_prediction - episode_rewards, 'b.',
                           label = legend_prefix + 'reward prediction error')
            test_episodes = g['test_episode_rewards'][
                :num_episodes // TEST_REWARD_INTERVAL]
            xx = episode_ends[TEST_REWARD_INTERVAL - 1::TEST_REWARD_INTERVAL]
            xx = xx + np.random.random(*xx.shape) * 0.5 - 0.25
            axs[i, 0].plot(xx, test_episodes,
                     'ro', label = legend_prefix + 'test rewards')
            legend_prefix = '_'
        axs[i, 0].grid(True)
        axs[i, 0].set_title('DQN started from ' + str(starting_height))
        axs[i, 0].legend()
        legend_prefix = ''
        measurement_name = '{}fixed-PPO-new-'.format(starting_height)
        for t in range(get_num_tests(f.keys(), measurement_name)):
            g = f[measurement_name + str(t)]
            num_episodes = g['num_episodes'][0]
            episode_ends = g['episode_ends'][:num_episodes]
            episode_rewards = g['episode_rewards'][:num_episodes]
            # xx = episode_ends + \
            #     np.random.random(episode_ends.shape) * 0.5 - 0.25
            # axs[i, 1].plot(xx, episode_rewards, '.',
            #          label = legend_prefix + 'episode rewards')
            reward_prediction = g['reward_prediction'][:num_episodes]
            xx = episode_ends + \
                np.random.random(episode_ends.shape) * 0.5 - 0.25
            # axs[i, 1].plot(xx, reward_prediction, 'x',
            #          label = legend_prefix + 'reward prediction')
            axs[i, 1].plot(xx, reward_prediction - episode_rewards, 'b.',
                           label = legend_prefix + 'reward prediction error')
            test_episodes = g['test_episode_rewards'][
                :num_episodes // TEST_REWARD_INTERVAL]
            xx = episode_ends[TEST_REWARD_INTERVAL - 1::TEST_REWARD_INTERVAL]
            xx = xx + np.random.random(*xx.shape) * 0.5 - 0.25
            axs[i, 1].plot(xx, test_episodes,
                     'ro', label = legend_prefix + 'test rewards')
            legend_prefix = '_'
        axs[i, 1].grid(True)
        axs[i, 1].set_title('DQN started from ' + str(starting_height))
        axs[i, 1].legend()
    fig.tight_layout()
    plt.show()




def visualise(fname):
    """
    Visualise the experiment.
    """
    import h5py
    with h5py.File(fname, 'r') as f:
        # visualise_generalisation(f)
        visualise_fixed(f)
#  1}}} #

if __name__ == "__main__":
    main()

# vim:set et sw=4 ts=4 fdm=marker:
