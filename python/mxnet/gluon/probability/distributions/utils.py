# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import
"""Distribution utilities"""
__all__ = ['prob2logit', 'logit2prob', 'cached_property', 'sample_n_shape_converter',
           'constraint_check', 'digamma', 'gammaln', 'erfinv', 'erf']

from functools import update_wrapper
from numbers import Number
import numpy as onp
try:
    import scipy.special as sc
except ImportError:
    sc = None
from .... import np, npx


def constraint_check():
    """Unified check_constraint interface for both scalar and tensor
    """
    def _check(condition, err_msg):
        if isinstance(condition, bool):
            if not condition:
                raise ValueError(err_msg)
            return 1.0
        return npx.constraint_check(condition, err_msg)
    return _check


def digamma():
    """Unified digamma interface for both scalar and tensor
    """
    def compute(value):
        """Return digamma(value)
        """
        if isinstance(value, Number):
            if sc is not None:
                return sc.digamma(value, dtype='float32')
            else:
                raise ValueError('Numbers are not supported as input if scipy is not installed')
        return npx.digamma(value)
    return compute


def gammaln():
    """Unified gammaln interface for both scalar and tensor
    """
    def compute(value):
        """Return log(gamma(value))
        """
        if isinstance(value, Number):
            if sc is not None:
                return sc.gammaln(value, dtype='float32')
            else:
                raise ValueError('Numbers are not supported as input if scipy is not installed')
        return npx.gammaln(value)
    return compute


def erf():
    """Unified erf interface for both scalar and tensor
    """
    def compute(value):
        if isinstance(value, Number):
            if sc is not None:
                return sc.erf(value)
            else:
                raise ValueError('Numbers are not supported as input if scipy is not installed')
        return npx.erf(value)
    return compute


def erfinv():
    """Unified erfinv interface for both scalar and tensor
    """
    def compute(value):
        if isinstance(value, Number):
            if sc is not None:
                return sc.erfinv(value)
            else:
                raise ValueError('Numbers are not supported as input if scipy is not installed')
        return npx.erfinv(value)
    return compute


def sample_n_shape_converter(size):
    """Convert `size` to the proper format for performing sample_n.
    """
    if size is None:
        return size
    if size == ():
        size = None
    else:
        if isinstance(size, int):
            size = (size,)
        size = (-2,) + size
    return size


def sum_right_most(x, ndim):
    """Sum along the right most `ndim` dimensions of `x`,

    Parameters
    ----------
    x : Tensor
        Input tensor.
    ndim : Int
        Number of dimensions to be summed.

    Returns
    -------
    Tensor
    """
    if ndim == 0:
        return x
    axes = list(range(-ndim, 0))
    return x.sum(axes)


def _clip_prob(prob):
    eps = onp.finfo('float32').eps
    return np.clip(prob, eps, 1 - eps)


def _clip_float_eps(value):
    eps = onp.finfo('float32').eps
    return np.maximum(value, eps)


def prob2logit(prob, binary=True):
    r"""Convert probability to logit form.
    For the binary case, the logit stands for log(p / (1 - p)).
    Whereas for the multinomial case, the logit denotes log(p).
    """
    _clipped_prob = _clip_prob(prob)
    if binary:
        return np.log(_clipped_prob) - np.log1p(-_clipped_prob)
    # The clipped prob would cause numerical error in the categorical case,
    # no idea about the reason behind.
    return np.log(_clipped_prob)


def logit2prob(logit, binary=True):
    r"""Convert logit into probability form.
    For the binary case, `sigmoid()` is applied on the logit tensor.
    Whereas for the multinomial case, `softmax` is applied along the last
    dimension of the logit tensor.
    """
    if binary:
        return npx.sigmoid(logit)
    return npx.softmax(logit)


class _CachedProperty(object):
    r"""Use as a decorator for loading class attribute, but caches the value."""

    def __init__(self, func):
        self._func = func
        update_wrapper(self, self._func)

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        value = self._func(instance)
        setattr(instance, self._func.__name__, value)
        return value


cached_property = _CachedProperty
