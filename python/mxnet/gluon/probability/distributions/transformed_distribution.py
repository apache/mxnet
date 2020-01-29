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
        super(TransformedDistribution, self).__init__(_F)

    def sample(self, size=None):
        x = self._base_dist.sample(size)
        for t in self._transforms:
            x = t(x)
        return x

    def log_prob(self, value):
        """
        Compute log-likehood of `value` with `log_det_jacobian` and
        log-likehood of the base distribution according to the following conclusion:

        Given that Y = T(X),
        log(p(y)) = log(p(x)) - log(|dy/dx|)
        """
        log_prob = 0.0
        # y = T_n(T_{n-1}(...T_1(x))),
        y = value
        # Reverse `_transforms` to transform to the base distribution.
        for t in reversed(self._transforms):
            x = t.inv(y)
            # FIXME: handle multivariate cases.
            log_prob = log_prob - t.log_det_jacobian(x, y)
            y = x

        # FIXME: handle multivariate cases.
        log_prob = log_prob + self._base_dist.log_prob(y)
        return log_prob
        

