"""
Some simple labyrinth environments written using PyColab
"""
import numpy as np
import gym_excerpt as gyme

from pycolab import ascii_art
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as plab_sprites

from config_parser import ConfigParser
CONFIG = ConfigParser('./config.json')

DIE_ON_IMPACT = CONFIG.getOption('die_on_impact', False)

# The layout of the mazes.
# Legend:
#   'P': player starting position
#   'G': goal state(s)
#   '#': walls
MAZES_ART = [
    ['#########################',
     '#                       #',
     '#                       #',
     '#                       #',
     '#                       #',
     '#                       #',
     '#                       #',
     '#P     G                #',
     '#########################',],
    ]

class LabyrinthEnv(gyme.RLEnv):
    """
    Simple labyrinth (gridworld) environment where the goal is to find a
    specific state denoted by "G". The entire grid is observed.

    There is an option to save the replay: this is the sequences of the actions
    the agent took in each episode.
    """

    MAZE_ART = MAZES_ART[0]
    # Possible actions are: move north/east/south/west
    NUM_ACTIONS = 4
    # Observe the entire layout (with the agent's own position)
    # 0th layer: walls (1 means wall)
    # 1th layer: player
    # 2nd layer: goal state(s)
    NUM_SENSORS = (3, len(MAZE_ART), len(MAZE_ART[0]))

    metadata = {
        'render.modes': [],}

    def __init__(self, save_replay = False):
        self.engine = None
        self._replay = [] if save_replay else None

    def _reset(self):
        self.engine = make_game(self.MAZE_ART)
        observation, _, _ = self.engine.its_showtime()
        observation = self._to_observation(observation.board)
        if self.replay is not None:
            self.replay.append([])
        assert observation.shape == self.NUM_SENSORS
        return observation

    def _step(self, action):
        observation, reward, discount = self.engine.play(action)
        if self.replay is not None:
            assert len(self.replay) > 0, 'There are no started sequences in ' \
                'the replay; did you call `clear_replay` in the middle of an' \
                ' episode?'
            self.replay[-1].append(action)
        done = discount == 0.0
        info = {}
        return self._to_observation(observation.board), reward, done, info

    @staticmethod
    def _to_observation(board):
        """
        Converts the pycolab board representation to the one specified at
        NUM_SENSORS
        """
        walls = board == ord('#')
        player = board == ord('P')
        goal = board == ord('G')
        return np.array([walls, player, goal])

    @property
    def replay(self):
        return self._replay

    def clear_replay(self):
        if self._replay is not None:
            self._replay = []

class PlayerSprite(plab_sprites.MazeWalker):
    """
    The sprite taking care of the player logic
    """
    def __init__(self, corner, position, character):
        super().__init__(corner, position, character, impassable = '#')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is None:
            return
        moves = [
            self._north,
            self._east,
            self._south,
            self._west,
            self._stay, # Noop for human play
            ]
        if actions < 5:
            result = moves[actions](board, the_plot)
            if result is not None and DIE_ON_IMPACT:
                # Assuming there are no goals on walls, so this really will be
                # a -1
                the_plot.add_reward(-1)
                the_plot.terminate_episode()
        else:
            # Quit (for human play)
            the_plot.terminate_episode()

class ReplaySprite(PlayerSprite):
    """
    A sprite that simply repeats the actions given to it.

    Note that this works only in a deterministic environment/game.
    """

    def __init__(self, corner, position, character, replay):
        super().__init__(corner, position, character)
        self._replay = replay
        self._current_replay = 0

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is None:
            # This is when `Engine.its_showtime` is called
            return
        if self._current_replay >= len(self._replay):
            the_plot.log('Episode not finished')
            the_plot.terminate_episode()
            return
        actions = self._replay[self._current_replay]
        self._current_replay += 1
        super().update(actions, board, layers, backdrop, things, the_plot)

class GoalDrape(plab_things.Drape):
    """
    The drape taking care of the logic of reachig a goal state
    """
    def update(self, actions, board, layers, backdrop, things, the_plot):
        player_position = things['P'].position
        if self.curtain[player_position]:
            the_plot.add_reward(1)
            the_plot.terminate_episode()
        else:
            the_plot.add_reward(0)

def make_game(maze_art):
    return ascii_art.ascii_art_to_game(
        maze_art,
        what_lies_beneath = ' ',
        sprites = {
            'P': PlayerSprite,},
        drapes = {
            'G': GoalDrape,},
        update_schedule = ['P', 'G'],
        z_order = 'GP')

def create_ui(delay):
    import curses
    from pycolab import human_ui
    return human_ui.CursesUi(
        keys_to_actions = {
            curses.KEY_UP: 0,
            curses.KEY_RIGHT: 1,
            curses.KEY_DOWN: 2,
            curses.KEY_LEFT: 3,
            -1: 4,
            'q': 5,
            'Q': 5,},
        colour_fg = {
            ' ': (0,0,0),
            '#': (999,999,999),
            'P': (0,0,999),
            'G': (0,999,999),
            },
        delay = delay)

def replay_game(maze_id, replay):
    engine = ascii_art.ascii_art_to_game(
        MAZES_ART[maze_id],
        what_lies_beneath = ' ',
        sprites = {
            'P': ascii_art.Partial(ReplaySprite, replay = replay),},
        drapes = {
            'G': GoalDrape,},
        update_schedule = ['P', 'G'],
        z_order = 'GP')
    ui = create_ui(100)
    ui.play(engine)


def play_game(maze_id):
    engine = make_game(MAZES_ART[maze_id])
    ui = create_ui(None)
    ui.play(engine)

if __name__ == "__main__":
    play_game(0)
