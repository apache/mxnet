# coding: utf-8
# pylint: disable=no-member, protected-access, unused-import, no-name-in-module
"""Random number interface of MXNet."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call
from ._ndarray_internal import _sample_uniform as uniform
from ._ndarray_internal import _sample_normal as normal
from ._ndarray_internal import _sample_gamma as gamma
from ._ndarray_internal import _sample_exponential as exponential
from ._ndarray_internal import _sample_poisson as poisson
from ._ndarray_internal import _sample_negbinomial as negative_binomial
from ._ndarray_internal import _sample_gennegbinomial as generalized_negative_binomial

def seed(seed_state):
    """Seeds the random number generators in MXNet.

    This seed will affect behavior of functions in this module.
    It also affects the results from executors that contain random numbers
    such as dropout operators.

    Parameters
    ----------
    seed_state : int
        The random number seed to set to all devices.

    Notes
    -----
    The random number generator of MXNet is, by default, device-specific.
    This means that if you set the same seed, the random number sequence
    generated from GPU0 can be different from CPU.
    """
    if not isinstance(seed_state, int):
        raise ValueError('sd must be int')
    seed_state = ctypes.c_int(int(seed_state))
    check_call(_LIB.MXRandomSeed(seed_state))
