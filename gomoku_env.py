import numpy as np
import gym_excerpt as gyme
from rl.core import Env

import gomoku.board as board
import gomoku.player as player

from config_parser import ConfigParser
CONFIG = ConfigParser('./config.json')

# Size of the board: [height, width]
BOARD_SIZE = CONFIG.getOption('board_size', [15, 15])

class PseudoGUI:
    """
    GUI adapter for the board, needed for the engine

    The engine accesses the state through a reference to the gui (that contains
    the board state).
    """

    def __init__(self, board):
        self.board = board
        self.in_game = True

    def renew_board(self):
        pass

    def highlight_lastmove(self):
        pass

class GomokuEnvironment(Env):
    """
    RL Environment implementing a Gomoku game against a (nonlearning) AI
    opponent

    The learning agent plays white.
    """

    metadata = {
        'render.modes': ['human']
        }

    # The actions are the cells, in row major order
    NUM_ACTIONS = BOARD_SIZE[0] * BOARD_SIZE[1]
    # The state is represented by two matrices: the first has 1s for the white
    # positions, the second for the black positions
    NUM_SENSORS = (2, BOARD_SIZE[0], BOARD_SIZE[1])

    def __init__(self):
        self.state = None
        self.lastReward = None

        self.board = board.Board(BOARD_SIZE[0], BOARD_SIZE[1])
        self.opponent = player.a_easy.Easy(board.black)
        self.gui = PseudoGUI(self.board)

        self._seed()

    def _get_state(self):
        return np.stack([self.board.board == board.white,
                         self.board.board == board.black])

    def _reset(self):
        self.board.reset()
        self.lastReward = None
        return self._get_state()

    def _step(self, action):
        assert 0 <= action < self.NUM_ACTIONS, \
                'Invalid action: {}'.format(action)

        reward, done = None, False
        board_height = BOARD_SIZE[0]
        i, j = action // board_height, action % board_height

        # White's turn
        try:
            self.board[i, j] = board.white
        except board.InvalidMoveError:
            reward = -1
            done = True
        if self.board.full():
            reward = 0
            done = True
        else:
            winner, _ = self.board.winner()
            if winner is not None:
                assert winner == board.white, 'The last move lost the game'
                reward = 1
                done = True
        if done:
            return self._get_state(), reward, done, {}

        # Black's turn
        self.opponent.make_move(self.gui)
        if self.board.full():
            reward = 0
            done = True
        else:
            winner, _ = self.board.winner()
            if winner is not None:
                assert winner == board.black, 'The last move lost the game'
                reward = -1
                done = True
            else:
                reward = 0
                done = False

        return self._get_state(), reward, done, {}

    def _seed(self, seed = None):
        self.np_random, seed = gyme.np_random(seed)
        return [seed]

    def _render(self, mode = 'human', close = False):
        print(self.board.board)

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
