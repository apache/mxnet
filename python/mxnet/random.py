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

    This affects the behavior of modules in MXNet that uses random number generators,
    like the dropout operator and `NDArray`'s random sampling operators.

    Parameters
    ----------
    seed_state : int
        The random number seed to set to all devices.

    Notes
    -----
    Random number generators in MXNet are device specific. Therefore, random numbers
    generated from two devices can be different even if they are seeded using the same seed.

    Example
    -------
    >>> print(mx.nd.random_normal(shape=(2,2)).asnumpy())
    [[ 1.36481571 -0.62203991]
     [-1.4962182  -0.08511394]]
    >>> print(mx.nd.random_normal(shape=(2,2)).asnumpy())
    [[ 1.09544981 -0.20014545]
     [-0.20808885  0.2527658 ]]
    >>>
    >>> mx.random.seed(128)
    >>> print(mx.nd.random_normal(shape=(2,2)).asnumpy())
    [[ 0.47400656 -0.75213492]
     [ 0.20251541  0.95352972]]
    >>> mx.random.seed(128)
    >>> print(mx.nd.random_normal(shape=(2,2)).asnumpy())
    [[ 0.47400656 -0.75213492]
     [ 0.20251541  0.95352972]]
    """
    if not isinstance(seed_state, int):
        raise ValueError('sd must be int')
    seed_state = ctypes.c_int(int(seed_state))
    check_call(_LIB.MXRandomSeed(seed_state))
