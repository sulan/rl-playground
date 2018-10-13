"""
File: gym_excerpt.py
Author: sulan
Description: This is an excerpt from the OpenAI Gym package: all code here was copied
             from the git repo (commit 2a0a2a3) and modifications were made only to
             remove unnecessary dependencies and to resolve problems arising from the
             omission of namespaces.
"""

import numpy as np
import rl
from rl.core import Space,Env

import hashlib
import os
import struct
import sys

#######################################################################
#                            error module                             #
#######################################################################

class Error(Exception):
    pass

class UnsupportedMode(Exception):
    """Raised when the user requests a rendering mode not supported by the
    environment.
    """
    pass


#######################################################################
#                            spaces module                            #
#######################################################################


space_np_random = np.random.RandomState()

def seed(seed=None):
    """Seed the common numpy.random.RandomState used in spaces
    CF
    https://github.com/openai/gym/commit/58e6aa95e5af2c738557431f812abb81c505a7cf#commitcomment-17669277
    for some details about why we seed the spaces separately from the
    envs, but tl;dr is that it's pretty uncommon for them to be used
    within an actual algorithm, and the code becomes simpler to just
    use this common numpy.random.RandomState.
    """
    space_np_random.seed(seed)

# This numpy.random.RandomState gets used in all spaces for their
# 'sample' method. It's not really expected that people will be using
# these in their algorithms.
seed(0)

class Discrete(Space):
    """
    {0,1,...,n-1}
    Example usage:
    self.observation_space = spaces.Discrete(2)
    """
    def __init__(self, n):
        self.n = n
    def sample(self):
        return space_np_random.randint(self.n)
    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n
    def __repr__(self):
        return "Discrete(%d)" % self.n
    def __eq__(self, other):
        return self.n == other.n

class Box(Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low, high, shape=None):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)
    def sample(self):
        return space_np_random.uniform(low=self.low, high=self.high, size=self.low.shape)
    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()
    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    @property
    def shape(self):
        return self.low.shape
    def __repr__(self):
        return "Box" + str(self.shape)
    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)


#######################################################################
#                           seeding module                            #
#######################################################################

if sys.version_info < (3,):
    integer_types = (int, long)
else:
    integer_types = (int,)


def np_random(seed=None):
    if seed is not None and not (isinstance(seed, integer_types) and 0 <= seed):
        raise Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    seed = _seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed

def hash_seed(seed=None, max_bytes=8):
    """Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:
    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928
    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)
    Args:
        seed (Optional[int]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    if seed is None:
        seed = _seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(hash[:max_bytes])

def _seed(a=None, max_bytes=8):
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.
    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, integer_types):
        a = a % 2**(8 * max_bytes)
    else:
        raise Error('Invalid type for seed: {} ({})'.format(type(a), a))

    return a

# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b'\0' * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def _int_list_from_bigint(bigint):
    # Special case 0
    if bigint < 0:
        raise Error('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints
