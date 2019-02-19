# Visualisation with nvim-ipy

#  Import {{{1 #
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pickle
from scipy.signal import medfilt
#%matplotlib qt5
#  1}}} #

file_name = 'train.out.hdf5'
#file_name = '/tmp/results1/train.out.hdf5'
#file_name = '/tmp/l341b/meas/test-episode-test_20181116-2007/train.out.hdf5'
#  Datasets {{{1 #
f = h5py.File(file_name, 'r')
g = f['default']
loss = g['loss']
ne = g['num_episodes'][0]
print('Num episodes', ne)
rp = g['reward_prediction'][:ne]
er = g['episode_rewards'][:ne]
ee = g['episode_ends'][:ne]
episode_lengths = ee[1:] - ee[:-1]
episode_lengths = np.r_[ee[0], episode_lengths]
test_rewards = g['test_episode_rewards'][:ne // 100]
eps_max_loss = np.zeros(ne)
for i,(s,e) in enumerate(zip(np.r_[0,ee[:-1]], ee)):
    eps_max_loss[i] = np.max(loss[s:e])
#  1}}} #

#  Datasets2 {{{1 # 
g2 = f['default']
loss2 = g2['loss']
ne2 = g2['num_episodes'][0]
print('Num episodes', ne2)
rp2 = g2['reward_prediction'][:ne2]
er2 = g2['episode_rewards'][:ne2]
ee2 = g2['episode_ends'][:ne2]
episode_lengths2 = ee2[1:] - ee2[:-1]
episode_lengths2 = np.r_[ee2[0], episode_lengths2]
test_rewards2 = g2['test_episode_rewards'][:ne2 // 100]
eps_max_loss2 = np.zeros(ne2)
for i,(s,e) in enumerate(zip(np.r_[0,ee2[:-1]], ee2)):
    eps_max_loss2[i] = np.max(loss2[s:e])
#  1}}} #

#  Plot {{{ # 
plt.figure()
plt.plot(loss)
plt.xlabel('#step')
plt.title('loss')
plt.figure()
plt.plot(er)
plt.plot(rp)
plt.plot(eps_max_loss)
plt.legend(['episode_rewards', 'reward_prediction','loss'])
plt.xlabel('#episode')
plt.figure()
plt.plot(test_rewards)
plt.xlabel('#step/100')
plt.title('Test episode reward')
plt.figure()
plt.plot(episode_lengths)
plt.title('Episode length')
plt.xlabel('#episode')
plt.show()
#  }}} Plot # 

#  Agent Memory {{{ #
#mem_name = '/tmp/results1/agent.memory'
#with open(mem_name, 'rb') as f:
#    mem = pickle.load(f)
#  }}} Agent Memory #

#  GC {{{ # 
import gc
gc.collect()
#  }}} GC # 
# vim:set et sw=4 ts=4 fdm=marker:
