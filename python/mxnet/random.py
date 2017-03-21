# coding: utf-8
# pylint: disable=no-member, protected-access, unused-import, no-name-in-module
"""Random Number interface of mxnet."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call
from ._ndarray_internal import _sample_uniform as uniform
from ._ndarray_internal import _sample_normal as normal

def seed(seed_state):
    """Seed the random number generators in mxnet.

    This seed will affect behavior of functions in this module,
    as well as results from executors that contains Random number
    such as Dropout operators.

    Parameters
    ----------
    seed_state : int
        The random number seed to set to all devices.

    Notes
    -----
    The random number generator of mxnet is by default device specific.
    This means if you set the same seed, the random number sequence
    generated from GPU0 can be different from CPU.
    """
    if not isinstance(seed_state, int):
        raise ValueError('sd must be int')
    seed_state = ctypes.c_int(int(seed_state))
    check_call(_LIB.MXRandomSeed(seed_state))
