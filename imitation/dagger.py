import sys
import numpy as np

from keras.utils import CustomObjectScope
import tqdm

from extract_gameplays import self_play_episode, BOARD_SIZE, CONFIG
from super import loss, converter2
from kaiki_model import create_kaiki_model, GomokuConv

# Number of episodes self-play in each iteration
NUM_EPISODES_PER_ITERATION = CONFIG.getOption('num_episodes_per_iteration', 100)
# Number of fit epochs to perform in each iteration
NUM_EPOCHS_PER_ITERATION = CONFIG.getOption('num_epochs_per_iteration', 1000)
# The name of the output file with the model
OUTPUT_FILE = CONFIG.getOption('dagger_output_file', 'model.hdf5')


def dagger(num_iterations):
    with CustomObjectScope({'GomokuConv':GomokuConv}):
        model = create_kaiki_model((2, BOARD_SIZE[0], BOARD_SIZE[1]))
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['mse'])

    progressbar = tqdm.tqdm(range(num_iterations), desc = 'Progress: ')
    for _ in progressbar:
        # Self-play
        states = []
        expert_actions = []
        for _ in range(NUM_EPISODES_PER_ITERATION):
            state, _, expert_action= self_play_episode(model)
            states.append(state)
            expert_actions.append(expert_action)
        states = np.array(states)
        actions = np.array(actions)

        # Learn
        states, actions = converter2(states, actions)
        model.fit(states, actions, batch_size = 2048,
                  epochs = NUM_EPOCHS_PER_ITERATION,
                  callbacks = [], verbose = 0, validation_split = 0.8)

    model.save(OUTPUT_FILE)

if __name__ == "__main__":
    dagger(int(sys.argv[1]))
