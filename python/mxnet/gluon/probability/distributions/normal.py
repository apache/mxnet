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

import math
from .constraint import Real, Positive
from .exp_family import ExponentialFamily
from .utils import getF, erf, erfinv


class Normal(ExponentialFamily):
    r"""Create a Normal distribution object.

    Parameters
    ----------
    loc : Tensor or scalar, default 0
        mean of the distribution.
    scale : Tensor or scalar, default 1
        standard deviation of the distribution
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    has_grad = True
    support = Real()
    arg_constraints = {'loc': Real(), 'scale': Positive()}

    def __init__(self, loc=0.0, scale=1.0, F=None, validate_args=None):
        _F = F if F is not None else getF(loc, scale)
        self.loc = loc
        self.scale = scale
        super(Normal, self).__init__(
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
        log_scale = F.np.log(self.scale)
        log_prob = -((value - self.loc) ** 2) / (2 * self.variance)
        log_prob = log_prob - log_scale
        log_prob = log_prob - F.np.log(F.np.sqrt(2 * math.pi))
        return log_prob

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
        return self.F.np.random.normal(self.loc, self.scale, size)

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
        return self.F.npx.random.normal_n(self.loc, self.scale, size)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        new_instance.loc = F.np.broadcast_to(self.loc, batch_shape)
        new_instance.scale = F.np.broadcast_to(self.scale, batch_shape)
        super(Normal, new_instance).__init__(F=F,
                                             event_dim=self.event_dim,
                                             validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def cdf(self, value):
        if self._validate_args:
            self._validate_samples(value)
        erf_func = erf(self.F)
        standarized_samples = ((value - self.loc) /
                               (math.sqrt(2) * self.scale))
        erf_term = erf_func(standarized_samples)
        return 0.5 * (1 + erf_term)

    def icdf(self, value):
        erfinv_func = erfinv(self.F)
        return self.loc + self.scale * erfinv_func(2 * value - 1) * math.sqrt(2)

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.scale ** 2

    def entropy(self):
        F = self.F
        return 0.5 + 0.5 * math.log(2 * math.pi) + F.np.log(self.scale)

    @property
    def _natural_params(self):
        r"""Return the natural parameters of normal distribution,
        which are (\frac{\mu}{\sigma^2}, -0.5 / (\sigma^2))

        Returns
        -------
        Tuple
            Natural parameters of normal distribution.
        """
        return (self.loc / (self.scale ** 2),
                -0.5 * self.F.np.reciprocal(self.scale ** 2))

    def _log_normalizer(self, x, y):
        # pylint: disable=arguments-differ
        F = self.F
        return -0.25 * F.np.pow(x, 2) / y + 0.5 * F.np.log(-math.pi / y)
