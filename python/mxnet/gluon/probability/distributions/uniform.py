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
"""Uniform distribution"""
__all__ = ['Uniform']

from .distribution import Distribution
from .constraint import Real, Interval
from .utils import getF, sample_n_shape_converter

class Uniform(Distribution):
    # Reparameterization gradient for Uniform is currently not implemented
    # in the backend at this moment.
    has_grad = False
    arg_constraints = {'low': Real(), 'high': Real()}

    def __init__(self, low=0.0, high=1.0, F=None, validate_args=None):
        _F = F if F is not None else getF(low, high)
        self.low = low
        self.high = high
        super(Uniform, self).__init__(F=_F, event_dim=0, validate_args=validate_args)

    def log_prob(self, value):
        F = self.F
        lower_bound = (self.low < value).astype('float')
        upper_bound = (self.high > value).astype('float')
        # 0 if value \in [low, high], -inf otherwise.
        out_of_support_value = F.np.log(lower_bound * upper_bound)
        return out_of_support_value - F.np.log * (self.high - self.low)
    
    def sample(self, size=None):
        F = self.F
        return F.np.random.uniform(self.low, self.high, size=size)

    def sample_n(self, size=None):
        F = self.F
        return F.np.random.uniform(self.low, self.high,
                                   size=sample_n_shape_converter(size))

    @property
    def support(self):
        return Interval(self.low, self.high)
    
    def cdf(self, value):
        x = (value - self.low) / (self.high - self.low)
        return x.clip(0, 1)

    def icdf(self, value):
        return value * (self.high - self.low) + self.low

    def entropy(self):
        return self.F.np.log(self.high - self.low)