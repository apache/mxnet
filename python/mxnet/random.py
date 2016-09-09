# coding: utf-8
# pylint: disable=no-member, protected-access
"""Random Number interface of mxnet."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call
from .ndarray import empty
from . import _ndarray_internal as _internal


def uniform(low, high, shape=None, ctx=None, out=None):
    """Generate uniform distribution in [low, high) with shape.

    Parameters
    ----------
    low : float
        The lower bound of distribution.
    high : float
        The upper bound of distribution.
    shape : tuple, optional
        Output shape of the NDArray generated.
    ctx : Context, optional
        Context of output NDArray, will use default context if not specified.
    out : NDArray, optional
        Output place holder

    Returns
    -------
    out : NDArray
        The result NDArray with generated result.
    """
    if out is not None:
        if shape is not None or ctx is not None:
            raise ValueError('shape and ctx is not needed when out is specified')
    else:
        if shape is None:
            raise ValueError('shape is required when out is not specified')
        if isinstance(shape, int):
            shape = (shape,)
        out = empty(shape, ctx)
    return _internal._sample_uniform(low=low, high=high, shape=out.shape, out=out)


def normal(loc, scale, shape=None, ctx=None, out=None):
    """Generate normal(Gaussian) distribution N(mean, stdvar^2) with shape.

    Parameters
    ----------
    loc : float
        The mean of the normal distribution.
    scale : float
        The standard deviation of normal distribution.
    shape : tuple, optional
        Output shape of the NDArray generated.
    ctx : Context, optional
        Context of output NDArray, will use default context if not specified.
    out : NDArray, optional
        Output place holder

    Returns
    -------
    out : NDArray
        The result NDArray with generated result.
    """
    if out is not None:
        if shape is not None or ctx is not None:
            raise ValueError('shape and ctx is not needed when out is specified')
    else:
        if shape is None:
            raise ValueError('shape is required when out is not specified')
        if isinstance(shape, int):
            shape = (shape,)
        out = empty(shape, ctx)
    return _internal._sample_normal(loc=loc, scale=scale, shape=out.shape, out=out)


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
