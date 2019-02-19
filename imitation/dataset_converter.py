import numpy as np
import h5py


def converter(states, actions):

    board_shape = states[:, 0, :, :].shape

    board = np.zeros(board_shape)-1

    board[range(board_shape[0]), actions[:, 0], actions[:, 1]] = 1#(board_shape[1] * board_shape[2])-1

    return np.reshape(states, newshape=(states.shape[0],) + (1,) + states.shape[1:]), np.reshape(board, newshape=(board.shape[0],) + (1,) + board.shape[1:])


def main():

    f = h5py.File("dataset.hdf5", 'r')

    length = f['length'][0]

    states, actions = converter(f['states'][:length], f['actions'][:length])

    new_states = []#np.array((1, 1, 2, 16, 16))
    new_actions = []#np.zeros((1, 1, 1, 16, 16))

    for i in range(length):
        if i == 1000:
            print(i)
            break
        #if i == 300:
            #print(len(new_states))
            #if len(new_states)>256:
            #    print("asd")
        current = states[i]
        current0=np.rot90(current,0)
        #current1=np.rot90(current,1)
        #current2=np.rot90(current,2)
        #current3=np.rot90(current,3)
        #current4=np.flip(np.rot90(current,0))
        #current5=np.flip(np.rot90(current,1))
        #current6=np.flip(np.rot90(current,2))
        #current7=np.flip(np.rot90(current,3))
        leng=len(new_states)
        for j in range(leng):
            other = new_states[j]
            if np.sum(current0-other) == 0:
                new_actions[j]+=np.rot90(actions[i],0)
                current = None
                break
            #if np.sum(current1-other) == 0:
            #    new_actions[j]+=np.rot90(actions[i],-1)
            #    current = None
            #    break
            #if np.sum(current2-other) == 0:
            #    new_actions[j]+=np.rot90(actions[i],-2)
            #    current = None
            #    break
            #if np.sum(current3-other) == 0:
            #    new_actions[j]+=np.rot90(actions[i],-3)
            #    current = None
            #    break
            #if np.sum(current4-other) == 0:
            #    new_actions[j]+=np.flip(np.rot90(actions[i],0))
            #    current = None
            #    break
            #if np.sum(current5-other) == 0:
            #    new_actions[j]+=np.flip(np.rot90(actions[i],-1))
            #    current = None
            #    break
            #if np.sum(current6-other) == 0:
            #    new_actions[j]+=np.flip(np.rot90(actions[i],-2))
            #    current = None
            #    break
            #if np.sum(current7-other) == 0:
            #    new_actions[j]+=np.flip(np.rot90(actions[i],-3))
            #    current = None
            #    break
        if current is not None:
            new_states+=[current]
            new_actions+=[actions[i]]

    for i in range(len(new_actions)):
        new_actions[i] /= -np.min(new_actions[i])
        #if np.max(new_actions[i]) < 1:
        #    print(i,new_actions[i])

    new_states = np.array(new_states)
    new_actions = np.array(new_actions)

    g = h5py.File("new_data.hdf5", 'w')

    #print([type(new_states), type(new_actions), type(length)])

    length = g.create_dataset('length', shape=(1,), dtype='i8')

    length[0] = len(new_states)

    states = g.create_dataset('states', shape=(length[0], 1, 2, 16, 16),
                              maxshape=(None, 1, 2, 16, 16), dtype='i1')
    actions = g.create_dataset('actions', shape=(length[0], 1, 16, 16),
                               maxshape=(None, 1, 16, 16), dtype='f2')

    states[:length[0], :, :, :] = np.array(new_states).astype(np.int8)
    actions[:length[0], :, :, :] = np.array(new_actions).astype(np.float16)

    print(actions[4])

if __name__ == "__main__":
    main()
