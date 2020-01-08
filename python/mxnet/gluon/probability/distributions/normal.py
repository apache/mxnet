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
from .exp_family import ExponentialFamily
from .utils import getF


class Normal(ExponentialFamily):
    r"""Create a normal distribution object.

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

    has_grad = True

    def __init__(self, loc=0.0, scale=1.0, F=None):
        self.F = F if F is not None else getF([loc, scale])
        super(Normal, self).__init__(F=F)
        self._loc = loc
        self._scale = scale
        self.F = F

    def log_prob(self, value):
        """Compute the log likehood of `value`.

        Parameters
        ----------
        value : Tensor
            Input data.

        Returns
        -------
        Tensor
            Log likehood of the input.
        """
        F = self.F
        var = self._scale ** 2
        log_scale = F.np.log(self._scale)
        log_prob = -((value - self._loc) ** 2) / (2 * var)
        log_prob = log_prob - log_scale
        log_prob = log_prob + F.np.log(F.np.sqrt(2 * F.np.pi))
        return log_prob

    def sample(self, size=None):
        r"""Generate samples of `size` from the normal distribution
        parameterized by `self.loc` and `self.scale`

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
        return self.F.np.random.normal(self.loc,
                                       self.scale,
                                       size)

    def sample_n(self, batch_size=None):
        r"""Generate samples of (batch_size + broadcast(loc, scale).shape)
        from the normal distribution parameterized by `self.loc` and `self.scale`

        Parameters
        ----------
        batch_size : Tuple, Scalar, or None
            Size of independent batch to be generated from the distribution.

        Returns
        -------
        Tensor
            Samples from Normal distribution.
        """
        return self.F.npx.random.normal_n(self.loc,
                                          self.scale,
                                          batch_size)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        new_instance.loc = F.np.broadcast_to(self.loc, batch_shape)
        new_instance.scale = F.np.broadcast_to(self.scale, batch_shape)
        super(Normal, new_instance).__init__(F=F)
        return new_instance

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

    @property
    def _log_normalizer(self, x, y):
        """Return the log_normalizer term of normal distribution in exponential family term.

        Parameters
        ----------
        x : Tensor
            The first natural parameter.
        y : Tensor
            The second natural parameter.

        Returns
        -------
        Tensor
            the log_normalizer term
        """
        F = self.F
        return -0.25 * F.np.pow(2) / y + 0.5 * F.np.log(-F.np.pi / y)

