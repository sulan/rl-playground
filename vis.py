# Visualisation with nvim-ipy

#  Import {{{1 #
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import medfilt
#%matplotlib qt5
#  1}}} #


def moving_average(a, n=300):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

file_name = 'train.out.hdf5'
file_name = '/tmp/train.out.hdf5'
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
        eps_mean_loss[i] = np.mean(loss[s:e])
else:
    sorted_inds = np.argsort(ee)
    er = er[sorted_inds]
    rp = rp[sorted_inds]
    sorted_inds_interval = np.argsort(ee[TEST_INTERVAL::TEST_INTERVAL])
    assert sorted_inds_interval.shape == test_rewards.shape
    test_rewards = test_rewards[sorted_inds_interval]
    ee = ee[sorted_inds]
#  1}}} #

print('Number of games: ', ne)
print('Winrate: ', np.average((er+1)/2))

#  Plot {{{ # 
plt.figure()
plt.plot(loss)
plt.xlabel('#step')
plt.title('loss')
plt.figure()
plt.plot(er)
plt.plot(moving_average(er, 3000), color='red')
plt.plot(rp)
plt.plot(eps_mean_loss)
plt.legend(['episode_rewards', 'ep_rewards MA', 'reward_prediction', 'loss'])
plt.xlabel('#episode')
plt.figure()
plt.plot(test_rewards)
plt.plot(moving_average(test_rewards, 30), color='red')
plt.xlabel('#step/100')
plt.title('Test episode reward')
plt.figure()
plt.plot(episode_lengths)
plt.plot(moving_average(episode_lengths, 3000), color='red')
plt.title('Episode length')
plt.xlabel('#episode')
plt.show()
#  }}} Plot # 

#  Plot PPO {{{ #
plt.figure()
plt.plot(loss[0::3])
plt.plot(loss[1::3])
plt.plot(loss[2::3])
plt.legend(['clip', 'vf', 'entropy'])
plt.xlabel('#updates')
plt.title('loss')
plt.grid(True)
plt.figure()
plt.plot(loss[0::3])
plt.plot(loss[1::3])
plt.plot(loss[2::3])
plt.xlabel('#updates')
plt.plot(ee + np.random.random(ee.shape) * 0.5 - 0.25, er, '.')
plt.plot(ee + np.random.random(ee.shape) * 0.5 - 0.25, rp, 'x')
plt.legend(['clip_loss', 'vf_loss', 'entropy_loss', 'episode_rewards', 'reward_prediction'])
plt.grid(True)
#  }}} Plot PPO #

#%reset
#  GC {{{ # 
import gc
gc.collect()
#  }}} GC # 
# vim:set et sw=4 ts=4 fdm=marker:
