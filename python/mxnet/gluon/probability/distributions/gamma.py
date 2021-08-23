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
"""Gamma Distribution."""
__all__ = ['Gamma']

from .exp_family import ExponentialFamily
from .constraint import Real, Positive
from .utils import sample_n_shape_converter, gammaln, digamma
from .... import np



class Gamma(ExponentialFamily):
    r"""Create a Gamma distribution object.

    Parameters
    ----------
    shape : Tensor or scalar
        shape parameter of the distribution, often represented by `k` or `\alpha`
    scale : Tensor or scalar, default 1
        scale parameter of the distribution, often represented by `\theta`,
        `\theta` = 1 / `\beta`, where `\beta` stands for the rate parameter.
    """
    # pylint: disable=abstract-method

    # TODO: Implement implicit reparameterization gradient for Gamma.
    has_grad = False
    support = Real()
    arg_constraints = {'shape': Positive(), 'scale': Positive()}

    def __init__(self, shape, scale=1.0, validate_args=None):
        self.shape = shape
        self.scale = scale
        super(Gamma, self).__init__(
            event_dim=0, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        log_fn = np.log
        lgamma = gammaln()
        # alpha (concentration)
        a = self.shape
        # beta (rate)
        b = 1 / self.scale
        return a * log_fn(b) + (a - 1) * log_fn(value) - b * value - lgamma(a)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        new_instance.shape = np.broadcast_to(self.shape, batch_shape)
        new_instance.scale = np.broadcast_to(self.scale, batch_shape)
        super(Gamma, new_instance).__init__(event_dim=self.event_dim,
                                            validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def sample(self, size=None):
        return np.random.gamma(self.shape, 1, size) * self.scale

    def sample_n(self, size=None):
        return np.random.gamma(self.shape, 1, sample_n_shape_converter(size)) * self.scale

    @property
    def mean(self):
        return self.shape * self.scale

    @property
    def variance(self):
        return self.shape * (self.scale ** 2)

    def entropy(self):
        lgamma = gammaln()
        dgamma = digamma()
        return (self.shape + np.log(self.scale) + lgamma(self.shape) +
                (1 - self.shape) * dgamma(self.shape))

    @property
    def _natural_params(self):
        return (self.shape - 1, -1 / self.scale)
