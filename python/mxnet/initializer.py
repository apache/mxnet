# coding: utf-8
"""Initialization helper for mxnet"""
from __future__ import absolute_import

import numpy as np
from .base import string_types
from .ndarray import NDArray
from . import random

class Initializer(object):
    """Base class for Initializer."""

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, NDArray):
            raise TypeError('arr must be NDArray')
        if name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)
    # pylint: disable=no-self-use, missing-docstring
    def _init_zero(self, _, arr):
        arr[:] = 0.0

    def _init_bias(self, _, arr):
        arr[:] = 0.0

    def _init_gamma(self, _, arr):
        arr[:] = 1.0

    def _init_beta(self, _, arr):
        arr[:] = 0.0

    def _init_weight(self, name, arr):
        """Abstruct method to Initialize weight"""
        raise NotImplementedError("Must override it")

    def _init_default(self, name, _):
        raise ValueError('Unknown initialization pattern for %s' % name)
    # pylint: enable=no-self-use, missing-docstring

class Uniform(Initializer):
    """Initialize the weight with uniform [-scale, scale]

    Parameters
    ----------
    scale : float, optional
        The scale of uniform distribution
    """
    def __init__(self, scale=0.07):
        self.scale = scale

    def _init_weight(self, _, arr):
        random.uniform(-self.scale, self.scale, out=arr)


class Normal(Initializer):
    """Initialize the weight with normal(0, sigma)

    Parameters
    ----------
    sigma : float, optional
        Standard deviation for gaussian distribution.
    """
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def _init_weight(self, _, arr):
        random.normal(0, self.sigma, out=arr)


class Xavier(Initializer):
    """Initialize the weight with Xavier initialization scheme."""

    def _init_weight(self, _, arr):
        shape = arr.shape
        fan_in, fan_out = np.prod(shape[1:]), shape[0]
        scale = np.sqrt(3. / (fan_in + fan_out))
        random.uniform(-scale, scale, out=arr)

