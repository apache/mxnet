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
"""Exponential Distribution."""
__all__ = ['Exponential']

from .exp_family import ExponentialFamily
from .constraint import Positive
from .utils import getF, sample_n_shape_converter, cached_property


class Exponential(ExponentialFamily):
    r"""Create a Exponential distribution object parameterized by `scale`.

    Parameters
    ----------
    scale : Tensor or scalar
       Scale of the distribution. (scale = 1 /rate)
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
    """
    # pylint: disable=abstract-method

    has_grad = True
    support = Positive()
    arg_constraints = {'scale': Positive()}

    def __init__(self, scale=1.0, F=None, validate_args=None):
        _F = F if F is not None else getF(scale)
        self.scale = scale
        super(Exponential, self).__init__(
            F=_F, event_dim=0, validate_args=validate_args)

    @cached_property
    def rate(self):
        return 1 / self.scale

    @property
    def mean(self):
        return self.scale

    @property
    def variance(self):
        return self.scale ** 2

    @property
    def stddev(self):
        return self.scale

    def sample(self, size=None):
        return self.F.np.random.exponential(self.scale, size=size)

    def sample_n(self, size=None):
        return self.F.np.random.exponential(self.scale,
                                            size=sample_n_shape_converter(size))

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        new_instance.scale = F.np.broadcast_to(self.scale, batch_shape)
        super(Exponential, new_instance).__init__(F=F,
                                                  event_dim=self.event_dim,
                                                  validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        return F.np.log(self.rate) - self.rate * value

    def cdf(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        return 1 - F.np.exp(-self.rate * value)

    def icdf(self, value):
        F = self.F
        return - self.scale * F.np.log(1 - value)

    def entropy(self):
        F = self.F
        return 1.0 + F.np.log(self.scale)

    @property
    def _natural_params(self):
        return (-self.rate,)

    def _log_normalizer(self, x):
        # pylint: disable=arguments-differ
        F = self.F
        return -F.np.log(-x)
