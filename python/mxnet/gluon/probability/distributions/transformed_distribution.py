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
"""Transformed distribution"""
__all__ = ['TransformedDistribution']

from ..transformation import Transformation
from .distribution import Distribution
from .utils import sum_right_most


class TransformedDistribution(Distribution):
    def __init__(self, base_dist, transforms):
        self._base_dist = base_dist
        if isinstance(transforms, Transformation):
            transforms = [transforms,]
        self._transforms = transforms
        _F = base_dist.F
        # Overwrite the F in transform
        for t in self._transforms:
            t.F = _F
        event_dim = max([self._base_dist.event_dim] +
                             [t.event_dim for t in self._transforms])
        super(TransformedDistribution, self).__init__(_F, event_dim=event_dim)

    def sample(self, size=None):
        x = self._base_dist.sample(size)
        for t in self._transforms:
            x = t(x)
        return x

    def log_prob(self, value):
        """
        Compute log-likelihood of `value` with `log_det_jacobian` and
        log-likelihood of the base distribution according to the following conclusion:

        Given that Y = T(X),
        log(p(y)) = log(p(x)) - log(|dy/dx|)
        """
        log_prob = 0.0
        y = value # T_n(T_{n-1}(...T_1(x)))
        # Reverse `_transforms` to transform to the base distribution.
        for t in reversed(self._transforms):
            x = t.inv(y)
            log_prob = log_prob - sum_right_most(t.log_det_jacobian(x, y),
                                                 self.event_dim - t.event_dim)
            y = x
        log_prob = log_prob + sum_right_most(self._base_dist.log_prob(y),
                                             self.event_dim - self._base_dist.event_dim)
        return log_prob

    def cdf(self, value):
        """
        Compute the cumulative distribution function(CDF) p(Y < `value`)
        """
        sign = self.F.np.ones_like(value)
        for t in reversed(self._transforms):
            value = t.inv(value)
            sign = sign * t.sign
        value = self._base_dist.cdf(value)
        return sign * (value - 0.5) + 0.5

    def icdf(self, value):
        # FIXME: implement the inverse cdf for transformed distribution.
        sign = self.F.np.ones_like(value)
        for t in self._transforms:
            sign = sign * t.sign
        value = sign * (value - 0.5) + 0.5 # value or (1 - value)
        samples_base = self._base_dist.icdf(value)
        for t in self._transforms:
            samples_base = t(samples_base)
        return samples_base
