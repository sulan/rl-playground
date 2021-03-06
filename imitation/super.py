import json, time, sys
import numpy as np
import h5py
import pickle

from kaiki_model import create_kaiki_model

from keras.callbacks import ModelCheckpoint


def converter2(states, actions):

    new_states=np.rot90(states,k=0,axes=(2,3))
    new_states=np.concatenate([new_states,np.rot90(states,k=1,axes=(2,3))],axis=0)
    new_states=np.concatenate([new_states,np.rot90(states,k=2,axes=(2,3))],axis=0)
    new_states=np.concatenate([new_states,np.rot90(states,k=3,axes=(2,3))],axis=0)
    new_states=np.concatenate([new_states,np.flip(np.rot90(states,k=0,axes=(2,3)),3)],axis=0)
    new_states=np.concatenate([new_states,np.flip(np.rot90(states,k=1,axes=(2,3)),3)],axis=0)
    new_states=np.concatenate([new_states,np.flip(np.rot90(states,k=2,axes=(2,3)),3)],axis=0)
    new_states=np.concatenate([new_states,np.flip(np.rot90(states,k=3,axes=(2,3)),3)],axis=0)

    new_actions=np.rot90(actions,k=0,axes=(1,2))
    new_actions=np.concatenate([new_actions,np.rot90(actions,k=1,axes=(1,2))],axis=0)
    new_actions=np.concatenate([new_actions,np.rot90(actions,k=2,axes=(1,2))],axis=0)
    new_actions=np.concatenate([new_actions,np.rot90(actions,k=3,axes=(1,2))],axis=0)
    new_actions=np.concatenate([new_actions,np.flip(np.rot90(actions,k=0,axes=(1,2)),2)],axis=0)
    new_actions=np.concatenate([new_actions,np.flip(np.rot90(actions,k=1,axes=(1,2)),2)],axis=0)
    new_actions=np.concatenate([new_actions,np.flip(np.rot90(actions,k=2,axes=(1,2)),2)],axis=0)
    new_actions=np.concatenate([new_actions,np.flip(np.rot90(actions,k=3,axes=(1,2)),2)],axis=0)

    return new_states, new_actions

def convert_action_to_board(actions, board_shape):
    """
    Converts action indices to a board representation similar to the output of
    Kaiki

    # Arguments:
    action (array-like): A matrix with shape (N, k). If k == 1, then it
        interprets the rows as row-major indices into the board, if k == 2,
        then it interprets the rows as (row, column) coordinates into the board
    board_shape (pair of integers): The shape of the board.

    # Returns:
        A stack of matrices representing the board for the different actions in
        the rows of the input, with 1 (white) at the specified actions.
    """
    assert len(actions.shape) == 2
    N = actions.shape[0]
    output = np.zeros((N,) + board_shape)
    if actions.shape[1] == 1:
        actions = np.unravel_index(actions, board_shape)
        output[range(N), actions[0], actions[1]] = 1
        return output
    assert actions.shape[1] == 2
    output[range(N), actions[:, 0], actions[:, 1]] = 1
    return output

def converter(states, actions):

    board_shape = states[:, 0, :, :].shape

    board = np.zeros(board_shape)-1

    board[range(board_shape[0]), actions[:, 0], actions[:, 1]] = 1

    return states, board

def loss(y_true, y_pred):

    mse = (y_true-y_pred)**2

    custom_loss = np.sum((((y_true+1)*127)+1)*mse)

    return custom_loss


def main():

    f = h5py.File("new_dataset.hdf5", 'r')
    g = h5py.File("dataset.hdf5", 'r')

    length_f = f['length'][0]
    length_g = g['length'][0]

    model = create_kaiki_model((2,None,None))

    callback = ModelCheckpoint("model6", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    #model = load_kaiki_model(INPUT_MODEL, states.shape[2], states.shape[3], False)

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['mse'])


    states_f, actions_f = converter2(f['states'][:length_f,0], f['actions'][:length_f,0])
    states_g, actions_g = converter(g['states'][:length_g], g['actions'][:length_g])
    states, actions = np.concatenate([states_f, states_f, states_f, states_f, states_g]), np.concatenate([actions_f, actions_f, actions_f, actions_f, actions_g])

    model.fit(x=states, y=actions, batch_size=256, epochs=2, callbacks=[callback], verbose=1, validation_split=0.0)
    print(model.evaluate(x=states[:2048], y=actions[:2048], verbose=0))
    #pickle.dump(runner.agent.memory, open("agent.memory", "wb"))
    #a = pickle.load(open("agent.memory", "rb"))


if __name__ == "__main__":
    main()
