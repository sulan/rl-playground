from . import Player
from ..board import black, white, empty, InvalidMoveError

import numpy as np

from keras.models import load_model

from .config_parser import ConfigParser
CONFIG = ConfigParser('./config.json')

# Size of the board: [height, width]
BOARD_SIZE = CONFIG.getOption('board_size', [15, 15])
MODEL_FILE = CONFIG.input_model

class KaikiPlayer(Player):
    name = 'Kaiki'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = load_model(MODEL_FILE)

    def _make_move(self, gui):
        state = np.stack([gui.board.board == self.color,
                          gui.board.board == -self.color])
        # TODO check shape
        state.shape = (1, 1, 2,) + BOARD_SIZE
        action = self.model.predict(state)
        action = action[0]
        i, j = action // BOARD_SIZE[0], action % BOARD_SIZE[0]
        try:
            gui.board[i, j] = self.color
        except InvalidMoveError:
            print('Warning: Kaiki disregarded the rules.')
            self.random_move(self, gui)
