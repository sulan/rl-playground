import json, time, sys
import numpy as np
import h5py
import pickle

from keras.utils import custom_object_scope

from kaiki_model import create_kaiki_model
from gomoku_conv import GomokuConv

from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def converter(states, actions):

    board_shape = states[:, 0, :, :].shape

    board = np.zeros(board_shape)-1

    board[range(board_shape[0]), actions[:, 0], actions[:, 1]] = 1#(board_shape[1] * board_shape[2])-1

    return np.reshape(states, newshape=(states.shape[0],) + (1,) + states.shape[1:]), np.reshape(board, newshape=(board.shape[0],) + (1,) + board.shape[1:])


def loss(y_true, y_pred):

    mse = (y_true-y_pred)**2

    custom_loss = np.sum(((((y_true+1)/2)*254)+1)*mse)

    return custom_loss


def main():

    f = h5py.File("dataset.hdf5", 'r')

    length = f['length'][0]

    model = create_kaiki_model(f['states'][0].shape)

    callback = ModelCheckpoint("model3", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['mse'])

    #model = load_model("model2", custom_objects={'GomokuConv':GomokuConv})

    #a,b = f['states'][:length], f['actions'][:length]

    #print(a,b) np.reshape(f['states'][:length], newshape=(1,)+f['states'][:length].shape), np.reshape(f['actions'][:length], newshape=(1,)+f['actions'][:length].shape)

    states, actions = converter(f['states'][:length], f['actions'][:length])

    model.fit(x=states, y=actions, batch_size=32, epochs=1, callbacks=[callback], verbose=1, validation_split=0.1)
    print(model.evaluate(x=states[:1024], y=actions[:1024], verbose=0))
    print(model.predict(states[1000:1001]))
    print(actions[1000:1001])
    #pickle.dump(runner.agent.memory, open("agent.memory", "wb"))
    #a = pickle.load(open("agent.memory", "rb"))

    #print(a.observations.data[0:101])


if __name__ == "__main__":
    main()
