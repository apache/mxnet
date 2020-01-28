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
from .transformed_distribution import TransformedDistribution
from ..transformation import AbsTransform
from .normal import Normal
from .constraint import Positive
from .utils import getF
import math

class HalfNormal(TransformedDistribution):
    has_grad = True

    def __init__(self, scale=1.0, F=None):
        base_dist = Normal(0, scale, F)
        super(HalfNormal, self).__init__(base_dist, AbsTransform())

    def log_prob(self, value):
        value = self.support.check(value)
        log_prob = self._base_dist.log_prob(value) + math.log(2)
        return log_prob

    def cdf(self, value):
        # FIXME
        pass

    def icdf(self, value):
        # FIXME
        pass

    @property
    def support(self):
        return Positive(self.F)

    @property
    def scale(self):
        return self._base_dist._scale

    @property
    def mean(self):
        return self.scale * math.sqrt(2 / math.pi)

    @property
    def variance(self):
        pow_fn = self.F.np.power
        return pow_fn(self.scale, 2) * (1 - 2 / math.pi)

    