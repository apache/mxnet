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
__all__ = ['getF', 'prob2logit', 'logit2prob', 'cached_property', 'sample_n_shape_converter']

from functools import update_wrapper
from .... import nd, sym, np

def sample_n_shape_converter(size):
    """Convert `size` to the proper format for performing sample_n.
    """
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
    # TODO: Raise exception when params types are not consistent, i.e. mixed ndarray and symbols.
    for param in params:
        if isinstance(param, np.ndarray):
            return nd
        elif isinstance(param, sym.numpy._Symbol):
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
    axes = list(range(-ndim, 0))
    return x.sum(axes)


def _clip_prob(prob, F):
    import numpy as onp
    eps = onp.finfo('float32').eps
    return F.np.clip(prob, eps, 1 - eps)


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
