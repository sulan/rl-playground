import sys
import numpy as np
import random
import h5py

import gomoku.lib.board as gboard
import gomoku.lib.player as player
from gomoku.lib.player import PseudoGUI

from rl.policy import GreedyQPolicy

from super import convert_action_to_board

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
# Whether to start from the middle of the board
START_FROM_MIDDLE = CONFIG.getOption('start_from_middle', False)


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


def extract_ai_gameplays():
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


def random_state_transform(state, action, board):
    r = random.randrange(8)
    if r < 4:
        state = np.rot90(state, k = r, axes = (1, 2))
        action = np.rot90(action, k = r, axes = (0, 1))
        board = np.rot90(board, k = r, axes = (0, 1))
    else:
        state = np.flip(np.rot90(state, k = r - 4, axes = (1, 2)), axis = 1)
        action = np.flip(np.rot90(action, k = r - 4, axes = (0, 1)), axis = 0)
        board = np.flip(np.rot90(board, k = r - 4, axes = (0, 1)), axis = 0)
    return state, action, board


def self_play_episode(model, policy = None):
    """
    Let Kaiki an episode against itself
    """
    step_middle = START_FROM_MIDDLE
    if policy is None:
        policy = GreedyQPolicy()
    current_colour = gboard.white
    board = MyBoard(*BOARD_SIZE)
    if step_middle:
        pass
        ## TODO
        #board[BOARD_SIZE[0]//2, BOARD_SIZE[1]//2] = current_colour
        #current_colour *= -1
    expert = player.c_hard.Hard(current_colour)
    expert_board = MyBoard(*BOARD_SIZE)
    expert_gui = PseudoGUI(expert_board)
    states = []
    actions = []
    expert_actions = []
    while True:
        state = [board.board == current_colour,
                 board.board == -current_colour]
        state = np.stack(state)

        # Make a move
        q = model.predict(state.reshape(1, 2, *BOARD_SIZE))
        q.shape = (-1,)
        action_ind = policy.select_action(q)
        action_board = convert_action_to_board(
            action_ind.reshape(1,1), BOARD_SIZE)
        action_board.shape = tuple(BOARD_SIZE)
        action = np.unravel_index(action_ind, BOARD_SIZE)
        state, action_board, expert_board.board = random_state_transform(
            state, action_board, board.board)
        states.append(state)
        actions.append(action_board)

        # Let's see the expert's opinion
        expert.color = current_colour
        expert.make_move(expert_gui)
        expert_action = expert_board.last_action
        expert_action = convert_action_to_board(
            expert_action.reshape(1,2), BOARD_SIZE)
        expert_action.shape = tuple(BOARD_SIZE)
        expert_actions.append(expert_action)

        # May not be a valid move
        try:
            board[action] = current_colour
        except gboard.InvalidMoveError:
            break
        if board.full():
            break
        winner, _ = board.winner()
        if winner is not None:
            break
        current_colour *= -1
    return states, actions, expert_actions

if __name__ == "__main__":
    extract_ai_gameplays()
