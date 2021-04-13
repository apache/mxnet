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
"""Snedecor's F Distribution."""
__all__ = ['FisherSnedecor']

from numpy import nan
from .distribution import Distribution
from .gamma import Gamma
from .constraint import Positive
from .utils import getF, gammaln


class FisherSnedecor(Distribution):
    r"""Create a FisherSnedecor distribution object, often known as F distribution.

    Parameters
    ----------
    df1 : Tensor or scalar
        degree of freedom parameter 1
    scale : Tensor or scalar
        degree of freedom parameter 2
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    support = Positive()
    arg_constraints = {'df1': Positive(), 'df2': Positive()}

    def __init__(self, df1, df2, F=None, validate_args=None):
        _F = F if F is not None else getF(df1, df2)
        self.df1 = df1
        self.df2 = df2
        self._gamma1 = Gamma(0.5 * self.df1, 1 / self.df1)
        self._gamma2 = Gamma(0.5 * self.df2, 1 / self.df2)
        super(FisherSnedecor, self).__init__(
            F=_F, event_dim=0, validate_args=validate_args)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        new_instance.df1 = F.np.broadcast_to(self.df1, batch_shape)
        new_instance.df2 = F.np.broadcast_to(self.df2, batch_shape)
        new_instance._gamma1 = self._gamma1.broadcast_to(batch_shape)
        new_instance._gamma2 = self._gamma2.broadcast_to(batch_shape)
        super(FisherSnedecor, new_instance).__init__(F=F,
                                                     event_dim=0, validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    @property
    def mean(self):
        # mean is only defined for df2 > 2
        df2 = self.F.np.where(self.df2 <= 2, nan, self.df2)
        return df2 / (df2 - 2)

    @property
    def variance(self):
        # variance is only define for df2 > 4
        df2 = self.F.np.where(self.df2 <= 4, nan, self.df2)
        df1 = self.df1
        numerator = 2 * df2 ** 2 * (df1 + df2 - 2)
        denominator = df1 * (df2 - 2) ** 2 * (df2 - 4)
        return numerator / denominator

    def sample(self, size=None):
        X1 = self._gamma1.sample(size)
        X2 = self._gamma2.sample(size)
        return X1 / X2

    def sample_n(self, size=None):
        X1 = self._gamma1.sample_n(size)
        X2 = self._gamma2.sample_n(size)
        return X1 / X2

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        lgamma = gammaln(F)
        log = F.np.log
        ct1 = self.df1 / 2
        ct2 = self.df2 / 2
        ct3 = self.df1 / self.df2
        t1 = lgamma(ct1 + ct2) - lgamma(ct1) - \
            lgamma(ct2)  # Beta(df1/2, df2/2)
        t2 = log(ct3) * ct1 + (ct1 - 1) * log(value)
        t3 = (ct1 + ct2) * log(ct3 * value + 1)
        return t1 + t2 - t3
