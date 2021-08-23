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
"""Beta Distribution."""
__all__ = ['Beta']

from .exp_family import ExponentialFamily
from .constraint import UnitInterval, Positive
from .utils import sample_n_shape_converter, gammaln, digamma, _clip_prob
from .... import np


class Beta(ExponentialFamily):
    r"""Create a Beta distribution object.

    Parameters
    ----------
    alpha : Tensor or scalar
       The first shape parameter
    beta : Tensor or scalar
        The second shape parameter
    """
    # pylint: disable=abstract-method

    has_grad = False
    support = UnitInterval()
    arg_constraints = {'alpha': Positive(),
                       'beta': Positive()}

    def __init__(self, alpha, beta, validate_args=None):
        self.alpha = alpha
        self.beta = beta
        super(Beta, self).__init__(
            event_dim=0, validate_args=validate_args)

    def sample(self, size=None):
        X = np.random.gamma(self.alpha, 1, size=size)
        Y = np.random.gamma(self.beta, 1, size=size)
        out = X / (X + Y)
        return _clip_prob(out)

    def sample_n(self, size=None):
        return self.sample(sample_n_shape_converter(size))

    @property
    def mean(self):
        a = self.alpha
        b = self.beta
        return a / (a + b)

    @property
    def variance(self):
        a = self.alpha
        b = self.beta
        return (a * b /
                ((a + b) ** 2 * (a + b + 1)))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        lgamma = gammaln()
        log = np.log
        log1p = np.log1p
        a = self.alpha
        b = self.beta
        lgamma_term = lgamma(a + b) - lgamma(a) - lgamma(b)
        return (a - 1) * log(value) + (b - 1) * log1p(-value) + lgamma_term

    def entropy(self):
        lgamma = gammaln()
        dgamma = digamma()
        a = self.alpha
        b = self.beta
        lgamma_term = lgamma(a + b) - lgamma(a) - lgamma(b)
        return (-lgamma_term - (a - 1) * dgamma(a) - (b - 1) * dgamma(b) +
                (a + b - 2) * dgamma(a + b))
