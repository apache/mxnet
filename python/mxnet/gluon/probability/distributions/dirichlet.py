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
"""Dirichlet Distribution."""
__all__ = ['Dirichlet']

from .exp_family import ExponentialFamily
from .constraint import Positive, Simplex
from .utils import gammaln, digamma, sample_n_shape_converter, _clip_float_eps
from .... import np


class Dirichlet(ExponentialFamily):
    r"""Create a Dirichlet distribution object.

    Parameters
    ----------
    alpha : Tensor or scalar
       Shape parameter of the distribution
    """
    # pylint: disable=abstract-method

    has_grad = False
    support = Simplex()
    arg_constraints = {'alpha': Positive()}

    def __init__(self, alpha, validate_args=None):
        self.alpha = alpha
        super(Dirichlet, self).__init__(
            event_dim=1, validate_args=validate_args)

    def sample(self, size=None):
        if size is None:
            size = ()
            alpha = self.alpha
        else:
            if isinstance(size, int):
                alpha = np.broadcast_to(self.alpha, (size,) + (-2,))
            else:
                alpha = np.broadcast_to(self.alpha, size + (-2,))
        gamma_samples = np.random.gamma(alpha, 1)
        s = gamma_samples.sum(-1, keepdims=True)
        return _clip_float_eps(gamma_samples / s)

    def sample_n(self, size=None):
        alpha = self.alpha
        if size is None:
            return self.sample()
        gamma_samples = np.random.gamma(
            alpha, 1, sample_n_shape_converter(size))
        s = gamma_samples.sum(-1, keepdims=True)
        return _clip_float_eps(gamma_samples / s)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        lgamma = gammaln()
        alpha = self.alpha
        return (np.log(value) * (alpha - 1.0)).sum(-1) +\
            lgamma(alpha.sum(-1)) - lgamma(alpha).sum(-1)

    @property
    def mean(self):
        alpha = self.alpha
        return alpha / alpha.sum(-1, keepdims=True)

    @property
    def variance(self):
        a = self.alpha
        s = a.sum(-1, keepdims=True)
        return a * (s - a) / ((s + 1) * s ** 2)

    def entropy(self):
        lgamma = gammaln()
        dgamma = digamma()
        a0 = self.alpha.sum(-1)
        log_B_alpha = lgamma(self.alpha).sum(-1) - lgamma(a0)
        return (log_B_alpha + (self.alpha - 1).sum(-1) * dgamma(a0) -
                ((self.alpha - 1) * dgamma(self.alpha)).sum(-1))
