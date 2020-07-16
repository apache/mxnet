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
"""Gumbel Distribution."""
__all__ = ['Gumbel']

import math
from numpy import euler_gamma # Euler-Mascheroni constant
from .distribution import Distribution
from .constraint import Real, Positive
from .utils import getF, sample_n_shape_converter


class Gumbel(Distribution):
    r"""Create a Gumble distribution object

    Parameters
    ----------
    loc : Tensor or scalar, default 0
        Location parameter of the distribution.
    scale : Tensor or scalar, default 1
        Scale parameter of the distribution
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    has_grad = True
    support = Real()
    arg_constraints = {'loc': Real(),
                       'scale': Positive()}

    def __init__(self, loc, scale=1, F=None, validate_args=None):
        _F = F if F is not None else getF(loc, scale)
        self.loc = loc
        self.scale = scale
        super(Gumbel, self).__init__(
            F=_F, event_dim=0, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        # Standardized sample
        y = (self.loc - value) / self.scale
        return (y - F.np.exp(y)) - F.np.log(self.scale)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        new_instance.loc = F.np.broadcast_to(self.loc, batch_shape)
        new_instance.scale = F.np.broadcast_to(self.scale, batch_shape)
        super(Gumbel, new_instance).__init__(F=F,
                                             event_dim=self.event_dim,
                                             validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def cdf(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        y = (value - self.loc) / self.scale
        exp_fn = F.np.exp
        return exp_fn(-exp_fn(-y))

    def icdf(self, value):
        F = self.F
        log_fn = F.np.log
        return self.loc + self.scale * (-log_fn(-log_fn(value)))

    def sample(self, size=None):
        return self.F.np.random.gumbel(self.loc, self.scale, size)

    def sample_n(self, size=None):
        return self.F.np.random.gumbel(self.loc, self.scale, sample_n_shape_converter(size))

    @property
    def mean(self):
        return self.loc + self.scale * euler_gamma

    @property
    def stddev(self):
        return (math.pi / math.sqrt(6)) * self.scale

    @property
    def variance(self):
        return self.stddev ** 2

    def entropy(self):
        F = self.F
        return F.np.log(self.scale) + (1 + euler_gamma)
