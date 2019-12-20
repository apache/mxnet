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
"""Normal distribution"""
__all__ = ['Normal']
from mxnet import np, npx
from .exp_family import ExponentialFamily
from .distribution import _getF


class Normal(ExponentialFamily):

    def __init__(self, loc=0.0, scale=1.0, F=None):
        self.F = F if F is not None else _getF([loc, scale])
        super(Normal, self).__init__(F=F)
        self._loc = loc
        self._scale = scale
        self.F = F

    def sample(self, shape):
        return self.F.np.random.normal(self._loc,
                                       self._scale,
                                       shape)

    def sample_n(self, n):
        return self.F.npx.random.normal_n(self._loc,
                                          self._scale,
                                          n)

    def log_prob(self, value):
        F = self.F
        var = self._scale ** 2
        log_scale = F.np.log(self._scale)
        log_prob = -((value - self._loc) ** 2) / (2 * var)
        log_prob = log_prob - log_scale
        log_prob = log_prob + F.np.log(F.np.sqrt(2 * F.np.pi))
        return log_prob
