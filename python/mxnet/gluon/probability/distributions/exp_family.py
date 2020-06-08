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
"""Exponential family class"""
__all__ = ['ExponentialFamily']

from .distribution import Distribution


class ExponentialFamily(Distribution):
    r"""
    ExponentialFamily inherits from Distribution. ExponentialFamily is a base
    class for distributions whose density function has the form:
    p_F(x;\theta) = exp(
        <t(x), \theta> -
        F(\theta) +
        k(x)
    ) where
    t(x): sufficient statistics
    \theta: natural parameters
    F(\theta): log_normalizer
    k(x): carrier measure
    """

    @property
    def _natural_params(self):
        r"""
        Return a tuple that stores natural parameters of the distribution.
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        r"""
        Return the log_normalizer F(\theta) based the natural parameters.
        """
        raise NotImplementedError

    def _mean_carrier_measure(self, x):
        r"""
        Return the mean of carrier measure k(x) based on input x,
        this method is required for calculating the entropy.
        """
        raise NotImplementedError

    def entropy(self):
        r"""
        Return the entropy of a distribution.
        The entropy of distributions in exponential families
        could be computed by:
        H(P) = F(\theta) - <\theta, F(\theta)'> - E_p[k(x)]
        """
        raise NotImplementedError
