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
"""Half-cauchy Distribution"""
__all__ = ["HalfCauchy"]

import math
from numpy import inf
from .transformed_distribution import TransformedDistribution
from ..transformation import AbsTransform
from .cauchy import Cauchy
from .constraint import Positive


class HalfCauchy(TransformedDistribution):
    r"""Create a half cauchy object, where
        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Parameters
    ----------
    scale : Tensor or scalar, default 1
        Scale of the full Cauchy distribution.
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    has_grad = True
    support = Positive()
    arg_constraints = {'scale': Positive()}

    def __init__(self, scale=1.0, F=None, validate_args=None):
        base_dist = Cauchy(0, scale, F)
        self.scale = scale
        super(HalfCauchy, self).__init__(
            base_dist, AbsTransform(), validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        log_prob = self._base_dist.log_prob(value) + math.log(2)
        log_prob = self.F.np.where(value < 0, -inf, log_prob)
        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_samples(value)
        return 2 * self._base_dist.cdf(value) - 1

    def icdf(self, value):
        return self._base_dist.icdf((value + 1) / 2)

    def entropy(self):
        return self._base_dist.entropy() - math.log(2)

    @property
    def mean(self):
        return self.scale * math.sqrt(2 / math.pi)

    @property
    def variance(self):
        pow_fn = self.F.np.power
        return pow_fn(self.scale, 2) * (1 - 2 / math.pi)
