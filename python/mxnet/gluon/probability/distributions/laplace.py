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
"""Laplace distribution"""
__all__ = ['Laplace']

from .constraint import Real, Positive
from .distribution import Distribution
from .utils import getF, sample_n_shape_converter


class Laplace(Distribution):
    r"""Create a laplace distribution object.

    Parameters
    ----------
    loc : Tensor or scalar, default 0
        mean of the distribution.
    scale : Tensor or scalar, default 1
        scale of the distribution
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.

    """
    # pylint: disable=abstract-method

    has_grad = False
    support = Real()
    arg_constraints = {'loc': Real(), 'scale': Positive()}

    def __init__(self, loc=0.0, scale=1.0, F=None, validate_args=None):
        _F = F if F is not None else getF(loc, scale)
        self.loc = loc
        self.scale = scale
        super(Laplace, self).__init__(
            F=_F, event_dim=0, validate_args=validate_args)

    def log_prob(self, value):
        """Compute the log likelihood of `value`.

        Parameters
        ----------
        value : Tensor
            Input data.

        Returns
        -------
        Tensor
            Log likelihood of the input.
        """
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        return -F.np.log(2 * self.scale) - F.np.abs(value - self.loc) / self.scale

    def sample(self, size=None):
        r"""Generate samples of `size` from the normal distribution
        parameterized by `self._loc` and `self._scale`

        Parameters
        ----------
        size : Tuple, Scalar, or None
            Size of samples to be generated. If size=None, the output shape
            will be `broadcast(loc, scale).shape`

        Returns
        -------
        Tensor
            Samples from Normal distribution.
        """
        return self.F.np.random.laplace(self.loc, self.scale, size)

    def sample_n(self, size=None):
        r"""Generate samples of (batch_size + broadcast(loc, scale).shape)
        from the normal distribution parameterized by `self._loc` and `self._scale`

        Parameters
        ----------
        size : Tuple, Scalar, or None
            Size of independent batch to be generated from the distribution.

        Returns
        -------
        Tensor
            Samples from Normal distribution.
        """
        return self.F.np.random.laplace(self.loc, self.scale, sample_n_shape_converter(size))

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        new_instance.loc = F.np.broadcast_to(self.loc, batch_shape)
        new_instance.scale = F.np.broadcast_to(self.scale, batch_shape)
        super(Laplace, new_instance).__init__(F=F,
                                              event_dim=self.event_dim,
                                              validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def cdf(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        value = value - self.loc
        return 0.5 - 0.5 * F.np.sign(value) * F.np.expm1(-F.np.abs(value) / self.scale)

    def icdf(self, value):
        F = self.F
        value = value - 0.5
        return self.loc - self.scale * F.np.sign(value) * F.np.log1p(-2 * F.np.abs(value))

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return (2 ** 0.5) * self.scale

    @property
    def variance(self):
        return 2 * (self.scale ** 2)

    def entropy(self):
        F = self.F
        return 1 + F.np.log(2 * self.scale)
