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
"""Bernoulli class."""
__all__ = ['Bernoulli']
from .exp_family import ExponentialFamily
from .utils import prob2logit, logit2prob, getF


class Bernoulli(ExponentialFamily):
    r"""Create a bernoulli distribution object.

    Parameters
    ----------
    prob : Tensor or scalar, default None
        Probability of sampling `1`.
    logit : Tensor or scalar, default None
        The log-odds of sampling `1`.
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """

    def __init__(self, prob=None, logit=None, F=None):
        _F = F if F is not None else getF([prob, logit])
        super(Bernoulli, self).__init__(F=_F)

        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        self._prob = prob
        self._logit = logit

    @property
    def prob(self):
        """Get the probability of sampling `1`.
        
        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return self._prob if self._prob is not None else logit2prob(self._logit, True, self.F)

    @property
    def logit(self):
        """Get the log-odds of sampling `1`.
        
        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return self._logit if self._logit is not None else prob2logit(self._prob, True, self.F)

    @property
    def mean(self):
        return self.prob

    @property
    def variance(self):
        return self.prob * (1 - self.prob)

    def log_prob(self, value):
        # FIXME!
        # constraint_check = self.F.npx.constraint_check(self.support(value))
        # value = constraint_check * value

        F = self.F
        if self._prob is None:
            logit = self.logit
            return logit * (value - 1) - F.np.log(F.np.exp(-logit) + 1)
        else:
            # Parameterized by probability
            eps = 1e-12
            return (self.F.np.log(self._prob + eps) * value
                    + self.F.np.log1p(-self._prob + eps) * (1 - value))

    def sample(self, size=None):
        return self.F.npx.random.bernoulli(self._prob, self._logit, size)

    @property
    def support(self):
        # TODO: replace bitwise_or with logical_or, return a constraint object.
        return lambda x: self.F.np.bitwise_or((x == 0.0), (x == 1.0))

    @property
    def _natural_params(self):
        return (self.logit,)

    @property
    def _log_normalizer(self, x):
        return self.F.np.log(1 + self.F.np.exp(x))
