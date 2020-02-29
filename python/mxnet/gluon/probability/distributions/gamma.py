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
"""Gamma Distribution."""
__all__ = ['Gamma']

from .exp_family import ExponentialFamily
from .constraint import Real, Positive
from .utils import getF, sample_n_shape_converter

class Gamma(ExponentialFamily):
    # TODO: Implement implicit reparameterization gradient for Gamma.
    has_grad = False
    support = Real()
    arg_constraints = {'shape': Positive(), 'scale': Positive()}

    def __init__(self, shape, scale=1.0, F=None, validate_args=None):
        _F = F if F is not None else getF(shape, scale)
        self.shape = shape
        self.scale = scale
        super(Gamma, self).__init__(F=_F, event_dim=0, validate_args=validate_args)

    def log_prob(self, value): 
        F = self.F
        log_fn = F.np.log
        lgamma = F.npx.gammaln
        # alpha (concentration)
        a = self.shape
        # beta (rate)
        b = 1 / self.scale
        return a * log_fn(b) + (a - 1) * log_fn(value) - b * value - lgamma(a)

    def sample(self, size=None):
        return self.F.np.random.gamma(self.shape, self.scale, size)

    def sample_n(self, size=None):
        return self.F.np.random.gamma(self.shape, self.scale, sample_n_shape_converter(size))

    def entropy(self):
        # TODO: require computing derivative of gammaln(shape)
        raise NotImplementedError

    @property
    def _natural_params(self):
        return (self.shape - 1, -1 / self.scale)