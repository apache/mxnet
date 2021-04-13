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
"""Cauchy distribution"""

__all__ = ['Cauchy']

from numbers import Number
from numpy import nan, pi
from .constraint import Real
from .distribution import Distribution
from .utils import getF, sample_n_shape_converter


class Cauchy(Distribution):
    r"""Create a relaxed Cauchy distribution object.

    Parameters
    ----------
    loc : Tensor or scalar, default 0
        mode or median of the distribution
    scale : Tensor or scalar, default 1
        half width at half maximum
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    has_grad = True
    support = Real()
    arg_constraints = {'loc': Real(), 'scale': Real()}

    def __init__(self, loc=0.0, scale=1.0, F=None, validate_args=None):
        _F = F if F is not None else getF(loc, scale)
        self.loc = loc
        self.scale = scale
        super(Cauchy, self).__init__(
            F=_F, event_dim=0, validate_args=validate_args)

    @property
    def mean(self):
        return nan

    @property
    def variance(self):
        return nan

    def sample(self, size=None):
        # TODO: Implement sampling op in the backend.
        F = self.F
        # `np.zeros_like` does not support scalar at this moment.
        if (isinstance(self.loc, Number), isinstance(self.scale, Number)) == (True, True):
            u = F.np.random.uniform(size=size)
        else:
            u = F.np.random.uniform(F.np.zeros_like(
                self.loc + self.scale), size=size)
        return self.icdf(u)

    def sample_n(self, size=None):
        return self.sample(sample_n_shape_converter(size))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        log = self.F.np.log
        return (-log(pi) - log(self.scale) -
                log(1 + ((value - self.loc) / self.scale) ** 2))

    def cdf(self, value):
        if self._validate_args:
            self._validate_samples(value)
        return self.F.np.arctan((value - self.loc) / self.scale) / pi + 0.5

    def icdf(self, value):
        return self.F.np.tan(pi * (value - 0.5)) * self.scale + self.loc

    def entropy(self):
        log = self.F.np.log
        return log(4 * pi) + log(self.scale)
