import json, time, sys
import numpy as np
import h5py
import pickle

from keras.utils import custom_object_scope

from kaiki_model import create_kaiki_model
from gomoku_conv import GomokuConv

from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def converter2(states, actions):

    #states = states
    for i in range(len(actions)):
        actions[i] = ((actions[i]+1)*2/np.max(actions[i]+1))-1

    new_states=np.rot90(states,k=0,axes=(3,4))
    new_states=np.concatenate([new_states,np.rot90(states,k=1,axes=(3,4))],axis=0)
    new_states=np.concatenate([new_states,np.rot90(states,k=2,axes=(3,4))],axis=0)
    new_states=np.concatenate([new_states,np.rot90(states,k=3,axes=(3,4))],axis=0)
    new_states=np.concatenate([new_states,np.flip(np.rot90(states,k=0,axes=(3,4)),3)],axis=0)
    new_states=np.concatenate([new_states,np.flip(np.rot90(states,k=1,axes=(3,4)),3)],axis=0)
    new_states=np.concatenate([new_states,np.flip(np.rot90(states,k=2,axes=(3,4)),3)],axis=0)
    new_states=np.concatenate([new_states,np.flip(np.rot90(states,k=3,axes=(3,4)),3)],axis=0)

    new_actions=np.rot90(actions,k=0,axes=(2,3))
    new_actions=np.concatenate([new_actions,np.rot90(actions,k=1,axes=(2,3))],axis=0)
    new_actions=np.concatenate([new_actions,np.rot90(actions,k=2,axes=(2,3))],axis=0)
    new_actions=np.concatenate([new_actions,np.rot90(actions,k=3,axes=(2,3))],axis=0)
    new_actions=np.concatenate([new_actions,np.flip(np.rot90(actions,k=0,axes=(2,3)),2)],axis=0)
    new_actions=np.concatenate([new_actions,np.flip(np.rot90(actions,k=1,axes=(2,3)),2)],axis=0)
    new_actions=np.concatenate([new_actions,np.flip(np.rot90(actions,k=2,axes=(2,3)),2)],axis=0)
    new_actions=np.concatenate([new_actions,np.flip(np.rot90(actions,k=3,axes=(2,3)),2)],axis=0)

    #new_states = np.array(new_states)
    #new_actions = np.array(new_actions)
    #print(new_states)

    return new_states, new_actions


def converter(states, actions):

    board_shape = states[:, 0, :, :].shape

    board = np.zeros(board_shape)-1

    board[range(board_shape[0]), actions[:, 0], actions[:, 1]] = 1#(board_shape[1] * board_shape[2])-1

    return np.reshape(states, newshape=(states.shape[0],) + (1,) + states.shape[1:]), np.reshape(board, newshape=(board.shape[0],) + (1,) + board.shape[1:])


def loss(y_true, y_pred):

    mse = (y_true-y_pred)**2

    custom_loss = np.sum((((y_true+1)*127)+1)*mse)

    return custom_loss


def main():

    f = h5py.File("new_dataset.hdf5", 'r')

    length = f['length'][0]

    #print(f['actions'][4])

    model = create_kaiki_model(f['states'][0,0,:].shape)

    callback = ModelCheckpoint("model5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['mse'])

    #model = load_model("model2", custom_objects={'GomokuConv':GomokuConv})

    #a,b = f['states'][:length], f['actions'][:length]

    #print(a,b) np.reshape(f['states'][:length], newshape=(1,)+f['states'][:length].shape), np.reshape(f['actions'][:length], newshape=(1,)+f['actions'][:length].shape)

    #print(f['actions'][100:101])
    #return
    states, actions = converter2(f['states'][:length], f['actions'][:length])

    #print(states.shape)

    model.fit(x=states, y=actions, batch_size=2048, epochs=1000, callbacks=[callback], verbose=1, validation_split=0.0)
    print(model.evaluate(x=states[:2048], y=actions[:2048], verbose=0))
    #print(model.predict(states[1000:1001]))
    #print(actions[1000:1001])
    print(model.predict(states[10:11]))
    print(actions[10:11])
    #pickle.dump(runner.agent.memory, open("agent.memory", "wb"))
    #a = pickle.load(open("agent.memory", "rb"))

    #print(a.observations.data[0:101])


if __name__ == "__main__":
    main()
