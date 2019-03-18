from __future__ import print_function,division,with_statement,nested_scopes
import collections
import numpy as np
import gym_excerpt as gyme
from rl.core import Env

MAX_ACCELERATION = 4

from config_parser import ConfigParser
CONFIG = ConfigParser('./config.json')

# Reward system that rewards only goal, punishes the steps, everything else is 0
BINARY_REWARD = CONFIG.getOption('binary_reward', False)
# What to base the reward on: number of steps or fuel
REWARD_BASE = CONFIG.getOption('reward_base', 'num_steps')
assert REWARD_BASE in ['num_steps', 'fuel', 'goal'], REWARD_BASE
# Reward for the goal in the case of binary reward
GOAL_REWARD = CONFIG.getOption('goal_reward', 10)
# Maximum change in rotation in one step (in degrees)
MAX_TURN = CONFIG.getOption('max_turn', 2)
# Maximum horizontal distance both ways
MAX_HORIZONTAL_DISTANCE = CONFIG.getOption('max_horizontal_distance', 500)
MAX_ROTATION = 90

class DumbMarsEnvironment (Env):
    '''
    max reward = 1; min = -infty (theoretically)
    '''

    metadata = {
        'render.modes': ['human']
    }

    GRAVITY = CONFIG.getOption('gravity', -2)

    def __init__ (self, landing_strip_size):
        self.state = None
        self.lastReward = None
        self.random_start = False
        self.landing_strip_size = landing_strip_size

        self._seed()

        # self.visited_states = collections.Counter()

    def _seed (self, seed = None):
        self.np_random, seed = gyme.np_random(seed)
        return [seed]

    def determine_reward(self, x, y, vx, vy, a):
        assert REWARD_BASE in ['num_steps', 'fuel', 'goal'], REWARD_BASE
        if y > 0.5 and y <= self.ceil \
                and -MAX_HORIZONTAL_DISTANCE <= x <= +MAX_HORIZONTAL_DISTANCE:
            return 0 if REWARD_BASE == 'goal' else \
                    -0.1 if REWARD_BASE == 'num_steps' else -abs(a) / 10
        elif y > self.ceil:
            return 0 if BINARY_REWARD else 0 - vy*1
        elif np.abs(x) > MAX_HORIZONTAL_DISTANCE:
            return 0 if BINARY_REWARD else 0 - np.abs(vx) * 1
        else:
            # TODO consider rotation, as well
            if BINARY_REWARD:
                punishment = 0
            else:
                punishment = 1 * (vy + 2)
                if not np.abs(x) <= self.landing_strip_size:
                    punishment += -(np.abs(x) - self.landing_strip_size / 2)
            reward = GOAL_REWARD if BINARY_REWARD else 1
            return reward \
                    if vy >= -2 and np.abs(x) <= self.landing_strip_size / 2 \
                    else punishment

    def _step (self,action):
        raise NotImplementedError

    def _render (self, mode='human',close=False):
        print(self.state)
        #return str(self.state)+'\n'
        return str(self.state)


    # We need to define these to be compliant with the keras-rl API.
    # Remove them when subclassing gym.Env.
    # Possibly could extract it into an abstract base.

    def step (self,action):
        return self._step(action)

    def reset (self):
        return self._reset()

    def render (self,mode='human',close=False):
        modes = self.metadata.get('render.modes',[])
        if len(modes) == 0:
            raise gyme.UnsupportedMode(\
                    '{} does not support rendering (requested mode: {})'.\
                    format(self, mode))
        elif mode not in modes:
            raise gyme.UnsupportedMode(\
                    'Unsupported rendering mode: {}. (Supported modes for {}: {})'.\
                    format(mode, self, modes))
        return self._render(mode=mode,close=close)

    def close (self):
        pass

    def seed (self,seed=None):
        return self._seed(seed)

    def configure (self, *args, **kwargs):
        pass

class DumbMars1DEnvironment(DumbMarsEnvironment):
    NUM_ACTIONS = 3
    NUM_SENSORS = (3,)

    def __init__ (self, height, random_start = False):
        super(DumbMars1DEnvironment, self).__init__(1)
        self.action_space = gyme.Discrete(self.NUM_ACTIONS)
        self.height = height
        self.ceil = 2*height
        low = np.array([0.5,-self.ceil, -2])
        high= np.array([self.ceil,self.ceil, 2])
        self.observation_space = gyme.Box(low=low,high=high)
        self.random_start = random_start

    def _reset (self):
        if self.random_start:
            starting_height = np.floor(np.random.rand() * self.height) + 1
        else:
            starting_height = self.height
        self.state = np.array([starting_height,0,0])
        # self.visited_states[tuple(self.state)] += 1
        self.lastReward = None
        return np.array(self.state)

    def _step (self, action):
        assert self.action_space.contains(action), "invalid action {}"\
                .format(action)
        state = self.state
        y,v,a = state
        jerk = (action - 1) * 2

        a += jerk
        a = min(MAX_ACCELERATION,max(-MAX_ACCELERATION,a))

        y += v + 0.5*(a + self.GRAVITY)
        v += (a + self.GRAVITY)
        y = round(y)

        done = y < 0.5 or y > self.ceil
        reward = self.determine_reward(0, y, 0, v, a)

        self.state = np.array([y,v,a])
        self.lastReward = reward
        # self.visited_states[tuple(self.state)] += 1

        return np.array(self.state), reward, done, {}

class DumbMars2DEnvironment(DumbMarsEnvironment):
    # Possible actions:
    #   - increase, keep, decrease power (action % 3)
    #   - change rotation by (action / 3 - MAX_TURN) degrees
    NUM_ACTIONS = 3 * (2 * MAX_TURN + 1)
    # horizontal pos, height, hSpeed, vSpeed, rotation, acceleration
    NUM_SENSORS = (6,)
    # The direction of the acceleration is the vertical axis of the ship, ie.
    # it is rotated from the true vertical by rotation

    def __init__ (self,starting_x, starting_y, landing_strip_size):
        super(DumbMars2DEnvironment, self).__init__(landing_strip_size)
        self.action_space = gyme.Discrete(self.NUM_ACTIONS)
        self.starting_y = starting_y
        self.ceil = 2 * starting_y
        self.starting_x = starting_x
        low = np.array([-MAX_HORIZONTAL_DISTANCE, -self.ceil,
                        -MAX_HORIZONTAL_DISTANCE, -self.ceil,
                        -MAX_ROTATION, -MAX_ACCELERATION])
        high= np.array([+MAX_HORIZONTAL_DISTANCE, +self.ceil,
                        +MAX_HORIZONTAL_DISTANCE, +self.ceil,
                        +MAX_ROTATION, +MAX_ACCELERATION])
        self.observation_space = gyme.Box(low=low,high=high)

    def _reset (self):
        # No random start yet
        if self.random_start:
            print('Warning: random start is not yet supported in the 2D' +
                  ' Martian env.')
        #     starting_height = np.floor(np.random.rand() * self.height) + 1
        # else:
        #     starting_height = self.height
        self.state = np.array([self.starting_x, self.starting_y,
                               0, 0, 0, 0])
        # self.visited_states[tuple(self.state)] += 1
        self.lastReward = None
        return np.array(self.state)

    def _step (self, action):
        assert self.action_space.contains(action), "invalid action {}"\
                .format(action)
        state = self.state
        x, y, vx, vy, r, a = state
        jerk = (action % 3 - 1) * 2
        turn = action // 3 - MAX_TURN

        a += jerk
        a = min(MAX_ACCELERATION,max(-MAX_ACCELERATION,a))
        r += turn
        r = min(MAX_ROTATION, max(-MAX_ROTATION, r))

        r_radian = r / 180 * np.pi
        y += vy + 0.5*(a * np.cos(r_radian) + self.GRAVITY)
        vy += (a * np.cos(r_radian) + self.GRAVITY)
        x += vx + 0.5 * a * np.sin(r_radian)
        vx += a * np.sin(r_radian)

        done = y < 0.5 or y > self.ceil or x < -MAX_HORIZONTAL_DISTANCE \
                or x > MAX_HORIZONTAL_DISTANCE
        reward = self.determine_reward(x, y, vx, vy, a)

        self.state = np.array([x,y,vx,vy,r,a])
        self.lastReward = reward

        # No visited state for now (it will be too much output)
        # self.visited_states[tuple(self.state)] += 1

        return np.array(self.state), reward, done, {}
