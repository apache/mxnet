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
"""Base distribution class"""
__all__ = ['Distribution']
import mxnet as mx
from mxnet import np, npx


def _getF(*params):
    r"""
    Get running mode from parameters,
    return mx.ndarray if inputs are python scalar.
    """
    for param in params:
        if isinstance(param, np.ndarray):
            return mx.ndarray
        elif isinstance(param, mx.symbol.numpy._Symbol):
            return mx.symbol.numpy._Symbol
    return mx.ndarray


class Distribution(object):
    r"""Base class for distribution.
    
    Parameters
    ----------
    F : mx.ndarray or mx.symbol.numpy._Symbol
        Variable that stores the running mode.
    """          

    # Variable indicating whether the sampling method has
    # pathwise gradient.
    has_grad = False

    def __init__(self, F=None):
        self.F = F

    def log_prob(self, x):
        r"""
        Returns the log of the probability density/mass function evaluated at `x`.
        """
        raise NotImplementedError()

    def prob(self, x):
        r"""
        Returns the probability density/mass function evaluated at `x`.
        """
        raise NotImplementedError

    def sample(self, shape):
        r"""
        Generates a `shape` shaped sample.
        """
        raise NotImplementedError

    def sample_n(self, n):
        r"""
        Generate samples of (n + parameter_shape) from the distribution.
        """
        raise NotImplementedError

    @property
    def mean(self):
        r"""
        Return the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def variance(self):
        r"""
        Return the variance of the distribution.
        """
        return NotImplementedError
