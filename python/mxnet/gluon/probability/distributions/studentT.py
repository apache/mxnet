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
"""Student T distribution"""
__all__ = ['StudentT']

from numpy import nan, inf, pi
from .distribution import Distribution
from .constraint import Real, Positive
from .chi2 import Chi2
from .utils import gammaln, digamma, sample_n_shape_converter
from .... import np


class StudentT(Distribution):
    r"""Create a studentT distribution object, often known as t distribution.

    Parameters
    ----------
    df : Tensor or scalar
        degree of freedom.
    loc : Tensor or scalar, default 0
        mean of the distribution.
    scale : Tensor or scalar, default 1
        scale of the distribution
    """
    # pylint: disable=abstract-method

    support = Real()
    arg_constraints = {'df': Positive(), 'loc': Real(), 'scale': Real()}

    def __init__(self, df, loc=0.0, scale=1.0, validate_args=None):
        self.df = df
        self.loc = loc
        self.scale = scale
        self._chi2 = Chi2(self.df)
        super(StudentT, self).__init__(
            event_dim=0, validate_args=validate_args)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        new_instance.loc = np.broadcast_to(self.loc, batch_shape)
        new_instance.scale = np.broadcast_to(self.scale, batch_shape)
        new_instance.df = np.broadcast_to(self.df, batch_shape)
        new_instance._chi2 = self._chi2.broadcast_to(batch_shape)
        super(StudentT, new_instance).__init__(
            event_dim=0, validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    @property
    def mean(self):
        # mean is only defined for df > 1
        m = np.where(self.df <= 1, nan, self.loc)
        return m

    @property
    def variance(self):
        df = self.df
        v = self.scale ** 2 * self.df / (self.df - 2)
        v = np.where(df <= 2, inf, v)
        v = np.where(df <= 1, nan, v)
        return v

    def sample(self, size=None):
        X = np.random.normal(size=size)
        Z = self._chi2.sample(size)
        Y = X * np.sqrt(self.df / Z)
        return self.loc + Y * self.scale

    def sample_n(self, size=None):
        return self.sample(sample_n_shape_converter(size))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        lgamma = gammaln()
        df = self.df
        value = (value - self.loc) / self.scale
        return (
            lgamma((df + 1) / 2) - lgamma(df / 2) -
            np.log(self.scale) - 0.5 * np.log(df * pi)
            - 0.5 * (df + 1) * np.log1p(value ** 2 / df)
        )

    def entropy(self):
        lgamma = gammaln()
        dgamma = digamma()
        log_fn = np.log
        lbeta = lgamma(0.5 * self.df) + lgamma(0.5) - \
            lgamma(0.5 * (self.df + 1))
        return (log_fn(self.scale) +
                0.5 * (self.df + 1) *
                (dgamma(0.5 * (self.df + 1)) - dgamma(0.5 * self.df)) +
                0.5 * log_fn(self.df) + lbeta)
