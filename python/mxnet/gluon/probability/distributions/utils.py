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
__all__ = ['getF', 'prob2logit', 'logit2prob', 'cached_property', 'sample_n_shape_converter',
           'constraint_check', 'digamma', 'gammaln', 'erfinv', 'erf']

from functools import update_wrapper
from numbers import Number
import numpy as onp
try:
    import scipy.special as sc
except ImportError:
    sc = None
from .... import symbol as sym
from .... import ndarray as nd


def constraint_check(F):
    """Unified check_constraint interface for both scalar and tensor
    """
    def _check(condition, err_msg):
        if isinstance(condition, bool):
            if not condition:
                raise ValueError(err_msg)
            return 1.0
        return F.npx.constraint_check(condition, err_msg)
    return _check


def digamma(F):
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
        return F.npx.digamma(value)
    return compute


def gammaln(F):
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
        return F.npx.gammaln(value)
    return compute


def erf(F):
    """Unified erf interface for both scalar and tensor
    """
    def compute(value):
        if isinstance(value, Number):
            if sc is not None:
                return sc.erf(value)
            else:
                raise ValueError('Numbers are not supported as input if scipy is not installed')
        return F.npx.erf(value)
    return compute


def erfinv(F):
    """Unified erfinv interface for both scalar and tensor
    """
    def compute(value):
        if isinstance(value, Number):
            if sc is not None:
                return sc.erfinv(value)
            else:
                raise ValueError('Numbers are not supported as input if scipy is not installed')
        return F.npx.erfinv(value)
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


def getF(*params):
    """Get running mode from parameters,
    return mx.ndarray if inputs are python scalar.

    Returns
    -------
    ndarray or _Symbol
        the running mode inferred from `*params`
    """
    mode_flag = 0
    for param in params:
        if isinstance(param, nd.NDArray):
            if mode_flag < 0:
                raise TypeError("Expect parameters to have consistent running mode," +
                                " got {}".format([type(p) for p in params]))
            mode_flag = 1
        elif isinstance(param, sym.Symbol):
            if mode_flag > 0:
                raise TypeError("Expect parameters to have consistent running mode," +
                                " got {}".format([type(p) for p in params]))
            mode_flag = -1
    # In case of scalar params, we choose to use the imperative mode.
    if mode_flag < 0:
        return sym
    return nd


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


def _clip_prob(prob, F):
    eps = onp.finfo('float32').eps
    return F.np.clip(prob, eps, 1 - eps)


def _clip_float_eps(value, F):
    eps = onp.finfo('float32').eps
    return F.np.maximum(value, eps)


def prob2logit(prob, binary=True, F=None):
    r"""Convert probability to logit form.
    For the binary case, the logit stands for log(p / (1 - p)).
    Whereas for the multinomial case, the logit denotes log(p).
    """
    if F is None:
        F = getF(prob)
    _clipped_prob = _clip_prob(prob, F)
    if binary:
        return F.np.log(_clipped_prob) - F.np.log1p(-_clipped_prob)
    # The clipped prob would cause numerical error in the categorical case,
    # no idea about the reason behind.
    return F.np.log(_clipped_prob)


def logit2prob(logit, binary=True, F=None):
    r"""Convert logit into probability form.
    For the binary case, `sigmoid()` is applied on the logit tensor.
    Whereas for the multinomial case, `softmax` is applied along the last
    dimension of the logit tensor.
    """
    if F is None:
        F = getF(logit)
    if binary:
        return F.npx.sigmoid(logit)
    return F.npx.softmax(logit)


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
