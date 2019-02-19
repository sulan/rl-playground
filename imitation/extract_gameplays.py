import sys
import numpy as np
import h5py

import gomoku.board as gboard
import gomoku.player as player
from gomoku.player import PseudoGUI

from config_parser import ConfigParser
CONFIG = ConfigParser('./config.json')

# Size of the board: [height, width]
BOARD_SIZE = CONFIG.getOption('board_size', [16, 16])
BOARD_SIZE = tuple(BOARD_SIZE)
assert len(BOARD_SIZE) == 2
# Player distribution: easy, medium, hard
PLAYER_DISTRIBUTION = CONFIG.getOption('player_distribution', [0, 0.5, 0.5])
assert len(PLAYER_DISTRIBUTION) == 3
# Output file name
OUTPUT_NAME = CONFIG.getOption('output_name', 'dataset.hdf5')

class MyBoard(gboard.Board):
    """
    Board subclass that remembers the last action.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_action = None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.last_action = np.array(key)

def match(players):
    current_player_id = 0
    board = MyBoard(*BOARD_SIZE)
    gui = PseudoGUI(board)
    states = []
    actions = []
    print([type(p) for p in players])
    while True:
        current_player = players[current_player_id]
        state = [board.board == gboard.white, board.board == gboard.black]
        if current_player_id != 0:
            state = state[::-1]
        state = np.stack(state)
        current_player.make_move(gui)
        action = board.last_action
        states.append(state)
        actions.append(action)
        if board.full():
            break
        winner, _ = board.winner()
        if winner is not None:
            break
        current_player_id = 1 - current_player_id
    return states, actions

def get_random_player(color):
    player_ctors = [
        player.a_easy.Easy,
        player.b_medium.Medium,
        player.c_hard.Hard,
        ]
    return np.random.choice(player_ctors,
                            p = PLAYER_DISTRIBUTION)(color)

def generate_some_plays(num):
    states, actions = [], []
    for _ in range(num):
        s, a = match([get_random_player(gboard.white),
                      get_random_player(gboard.black)])
        states += s
        actions += a
    return states, actions

def main():
    max_num_episodes = int(sys.argv[1])
    f = h5py.File(OUTPUT_NAME, 'w')
    length = f.create_dataset('length', shape = (1,), dtype = 'i8')
    length[0] = 0
    max_length = 8
    states = f.create_dataset('states', shape = (max_length, 2) + BOARD_SIZE,
                              maxshape = (None, 2) + BOARD_SIZE, dtype = 'i1')
    actions = f.create_dataset('actions', shape = (max_length, 2),
                               maxshape = (None, 2), dtype = 'i1')
    num_episodes = 0
    while num_episodes < max_num_episodes:
        num_new_episodes = min(100, max_num_episodes - num_episodes)
        new_states, new_actions = generate_some_plays(num_new_episodes)
        num_episodes += num_new_episodes
        new_l = len(new_states)
        if length[0] + new_l > max_length:
            while length[0] + new_l > max_length:
                max_length *= 2
            states.resize(max_length, axis = 0)
            actions.resize(max_length, axis = 0)
        states[length[0]:length[0] + new_l, :, :, :] = \
                np.array(new_states).astype(np.int8)
        actions[length[0]:length[0] + new_l, :] = np.array(new_actions)
        length[0] = length[0] + new_l
    f.flush()
    f.close()

if __name__ == "__main__":
    main()
