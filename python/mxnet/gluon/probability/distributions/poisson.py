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
"""Poisson distribution."""
__all__ = ['Poisson']

from numbers import Number
from .exp_family import ExponentialFamily
from .constraint import Positive, NonNegativeInteger
from .utils import getF, gammaln


class Poisson(ExponentialFamily):
    r"""Create a Poisson distribution object.

    Parameters
    ----------
    rate : Tensor or scalar, default 1
        rate parameter of the distribution.
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    arg_constraints = {'rate': Positive()}
    support = NonNegativeInteger()

    def __init__(self, rate=1.0, F=None, validate_args=None):
        _F = F if F is not None else getF(rate)
        self.rate = rate
        super(Poisson, self).__init__(
            F=_F, event_dim=0, validate_args=validate_args)

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        new_instance.rate = F.np.broadcast_to(self.rate, batch_shape)
        super(Poisson, new_instance).__init__(F=F,
                                              event_dim=self.event_dim,
                                              validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def sample(self, size=None):
        F = self.F
        lam = self.rate
        if size is None:
            size = ()
        if isinstance(lam, Number):
            # Scalar case
            return F.npx.scalar_poisson(lam, size)
        else:
            # Tensor case
            shape_tensor = F.np.ones(size)
            # shape = () currently not supported
            return F.npx.tensor_poisson(lam * shape_tensor)

    def sample_n(self, size=None):
        F = self.F
        lam = self.rate
        if isinstance(lam, Number):
            # Scalar case
            if size is None:
                size = ()
            return F.npx.scalar_poisson(lam, size)
        else:
            return F.np.moveaxis(F.npx.tensor_poisson(lam, size), -1, 0)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        lgamma = gammaln(F)
        rate = self.rate
        return value * F.np.log(rate) - rate - lgamma(value + 1)

    @property
    def _natural_params(self):
        F = self.F
        return (F.np.log(self.rate),)

    def _log_normalizer(self, x):
        # pylint: disable=arguments-differ
        F = self.F
        return F.np.exp(x)
