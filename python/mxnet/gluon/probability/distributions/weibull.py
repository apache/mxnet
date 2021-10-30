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
"""Weibull Distribution."""
__all__ = ['Weibull']

# Euler-Mascheroni constant
from numpy import euler_gamma
from .transformed_distribution import TransformedDistribution
from .exponential import Exponential
from .constraint import Positive
from ..transformation import PowerTransform, AffineTransform
from .utils import sample_n_shape_converter, gammaln
from .... import np, npx


class Weibull(TransformedDistribution):
    r"""Create a two parameter Weibull distribution object.

    Parameters
    ----------
    concentration : Tensor or scalar
        Concentration/shape parameter of the distribution.
    scale : Tensor or scalar, default 1
        scale parameter of the distribution.
    """
    # pylint: disable=abstract-method
    has_grad = True
    support = Positive()
    arg_constraints = {'scale': Positive(),
                       'concentration': Positive()}

    def __init__(self, concentration, scale=1.0, validate_args=None):
        self.concentration = concentration
        self.scale = scale
        base_dist = Exponential()
        super(Weibull, self).__init__(base_dist, [PowerTransform(1 / self.concentration),
                                                  AffineTransform(0, self.scale)])

    def sample(self, size=None):
        return self.scale * np.random.weibull(self.concentration, size)

    def sample_n(self, size=None):
        return self.scale * np.random.weibull(self.concentration,
                                              sample_n_shape_converter(size))

    @property
    def mean(self):
        return self.scale * np.exp(npx.gammaln(1 + 1 / self.concentration))

    @property
    def variance(self):
        exp = np.exp
        lgamma = gammaln()
        term1 = exp(lgamma(1 + 2 / self.concentration))
        term2 = exp(2 * lgamma(1 + 1 / self.concentration))
        return (self.scale ** 2) * (term1 - term2)

    def entropy(self):
        return (euler_gamma * (1 - 1 / self.concentration) +
                np.log(self.scale / self.concentration) + 1)
