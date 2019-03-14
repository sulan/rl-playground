from . import Player
from ..board import black, white, empty, InvalidMoveError
from .gomoku_conv import GomokuConv

import numpy as np

from keras.models import load_model
import keras.backend as K

from .config_parser import ConfigParser
CONFIG = ConfigParser('./config.json')

# Size of the board: [height, width]
BOARD_SIZE = CONFIG.getOption('board_size', [16, 16])
MODEL_FILE = CONFIG.getOption('input_model', None)

class KaikiPlayer(Player):
    name = 'Kaiki'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert MODEL_FILE is not None
        self.model = load_model(MODEL_FILE,
                custom_objects = {'GomokuConv' : GomokuConv})

    def _make_move(self, gui):
        state = np.stack([gui.board.board == self.color,
                          gui.board.board == -self.color])
        state.shape = (1, 1, 2,) + tuple(BOARD_SIZE)
        action = self.model.predict(state)
        action = np.argmax(action[0])
        i, j = action // BOARD_SIZE[0], action % BOARD_SIZE[0]
        try:
            gui.board[i, j] = self.color
        except InvalidMoveError:
            print('Warning: Kaiki disregarded the rules.')
            self.random_move(gui)

    def __del__(self):
        K.clear_session()
