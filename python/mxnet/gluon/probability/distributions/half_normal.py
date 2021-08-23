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
"""Half-normal Distribution"""
__all__ = ["HalfNormal"]

import math
from numpy import inf
from .transformed_distribution import TransformedDistribution
from ..transformation import AbsTransform
from .normal import Normal
from .constraint import Positive
from .... import np


class HalfNormal(TransformedDistribution):
    r"""Create a half normal object, where
        X ~ Normal(0, scale)
        Y = |X| ~ HalfNormal(scale)

    Parameters
    ----------
    scale : Tensor or scalar, default 1
        Scale of the full Normal distribution.
    """
    # pylint: disable=abstract-method

    has_grad = True
    support = Positive()
    arg_constraints = {'scale': Positive()}

    def __init__(self, scale=1.0, validate_args=None):
        base_dist = Normal(0, scale)
        self.scale = scale
        super(HalfNormal, self).__init__(
            base_dist, AbsTransform(), validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        log_prob = self._base_dist.log_prob(value) + math.log(2)
        log_prob = np.where(value < 0, -inf, log_prob)
        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_samples(value)
        return 2 * self._base_dist.cdf(value) - 1

    def icdf(self, value):
        return self._base_dist.icdf((value + 1) / 2)

    @property
    def loc(self):
        return self._base_dist.loc

    @property
    def mean(self):
        return self.scale * math.sqrt(2 / math.pi)

    @property
    def variance(self):
        return np.power(self.scale, 2) * (1 - 2 / math.pi)
