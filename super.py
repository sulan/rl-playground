import json, time, sys
import numpy as np
import h5py
import pickle

from kaiki_model import create_kaiki_model


def converter(states, actions):

    board_shape = states[:, 0, :, :].shape

    board = np.zeros(board_shape)-1

    board[range(board_shape[0]), actions[:, 0], actions[:, 1]] = (board_shape[1]*board_shape[2])-1

    return np.reshape(states, newshape=(states.shape[0],) + (1,) + states.shape[1:]), np.reshape(board, newshape=(board.shape[0],) + (1,) + board.shape[1:])


def main():

    f = h5py.File("data.hdf5", 'r')

    length = f['length'][0]
    print(f['states'][0].shape)
    model = create_kaiki_model(f['states'][0].shape)

    model.compile(
        optimizer='adam',
        loss='mean_squared_error')

    #a,b = f['states'][:length], f['actions'][:length]

    #print(a,b) np.reshape(f['states'][:length], newshape=(1,)+f['states'][:length].shape), np.reshape(f['actions'][:length], newshape=(1,)+f['actions'][:length].shape)

    states, actions = converter(f['states'][:length], f['actions'][:length])

    model.fit(x=states, y=actions, epochs=100, verbose=1, validation_split=10.0)
    print(model.evaluate(x=states, y=actions, verbose=0))
    #model.predict()
    #pickle.dump(runner.agent.memory, open("agent.memory", "wb"))
    #a = pickle.load(open("agent.memory", "rb"))

    #print(a.observations.data[0:101])


if __name__ == "__main__":
    main()
