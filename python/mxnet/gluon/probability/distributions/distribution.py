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
        self._kl_dict = {}
        self.F = F

    def log_prob(self, value):
        r"""
        Returns the log of the probability density/mass function evaluated at `value`.
        """
        raise NotImplementedError()

    def prob(self, value):
        r"""
        Returns the probability density/mass function evaluated at `value`.
        """
        raise NotImplementedError

    def cdf(self, value):
        r"""
        Return the cumulative density/mass function evaluated at `value`.
        """
        raise NotImplementedError

    def icdf(self, value):
        r"""
        Return the inverse cumulative density/mass function evaluated at `value`.
        """
        raise NotImplementedError

    def sample(self, size=None):
        r"""
        Generates a `shape` shaped sample.
        """
        raise NotImplementedError

    def sample_n(self, n):
        r"""
        Generate samples of (n + parameter_shape) from the distribution.
        """
        raise NotImplementedError

    def broadcast_to(self, batch_shape):
        """
        Returns a new distribution instance with parameters expanded
        to `batch_shape`. This method calls `numpy.broadcast_to` on
        the parameters.

        Parameters
        ----------
        batch_shape : Tuple
            The batch shape of the desired distribution.

        """
        raise NotImplementedError

    def enumerate_support(self):
        r"""
        Returns a tensor that contains all values supported
        by a discrete distribution.
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

    @property
    def support(self):
        """
        Return a function representing the distribution's support.
        """
        # TODO: return a constraint object
        return NotImplementedError

