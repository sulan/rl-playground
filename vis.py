# Visualisation with nvim-ipy

#  Import {{{1 #
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import medfilt
%matplotlib qt5
#  1}}} #

file_name = 'train.out.hdf5'
file_name = '/tmp/results0_wo_gom/train.out.hdf5'
file_name = '/tmp/l341b/meas/ppo-metrics-test_20190304-1627/train.out.hdf5'
A2C = True
TEST_INTERVAL = 100
#  Datasets {{{1 #
f = h5py.File(file_name, 'r')
g = f['default']
loss = g['loss']
ne = g['num_episodes'][0]
print('Num episodes', ne)
rp = g['reward_prediction'][:ne]
er = g['episode_rewards'][:ne]
ee = g['episode_ends'][:ne]
test_rewards = g['test_episode_rewards'][:ne // TEST_INTERVAL]
if not A2C:
    episode_lengths = ee[1:] - ee[:-1]
    episode_lengths = np.r_[ee[0], episode_lengths]
    eps_mean_loss = np.zeros(ne)
    for i,(s,e) in enumerate(zip(np.r_[0,ee[:-1]], ee)):
        eps_max_loss[i] = np.mean(loss[s:e])
else:
    sorted_inds = np.argsort(ee)
    er = er[sorted_inds]
    rp = rp[sorted_inds]
    sorted_inds_interval = np.argsort(ee[TEST_INTERVAL::TEST_INTERVAL])
    assert sorted_inds_interval.shape == test_rewards.shape
    test_rewards = test_rewards[sorted_inds_interval]
    ee = ee[sorted_inds]
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

#  Plot A2C {{{ #
plt.figure()
plt.plot(loss)
plt.xlabel('#step')
plt.title('loss')
plt.figure()
plt.plot(loss)
plt.xlabel('#step')
plt.plot(ee + np.random.random(ee.shape) * 0.5 - 0.25, er, '.-')
plt.plot(ee + np.random.random(ee.shape) * 0.5 - 0.25, rp, 'x-')
plt.legend(['loss', 'episode_rewards', 'reward_prediction'])
#  }}} Plot A2C # 

#  GC {{{ # 
import gc
gc.collect()
#  }}} GC # 
# vim:set et sw=4 ts=4 fdm=marker:
