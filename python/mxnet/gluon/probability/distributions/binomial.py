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
"""Binomial distribution class."""
__all__ = ['Binomial']

from .distribution import Distribution
from .utils import prob2logit, logit2prob, getF, cached_property, sample_n_shape_converter
from .utils import gammaln
from .constraint import Interval, Real, NonNegativeInteger


class Binomial(Distribution):
    r"""Create a binomial distribution object.

    Parameters
    ----------
    n : scalar
        Non-negative interger of Bernoulli trials to stop.
    prob : Tensor or scalar, default None
        Probability of sampling `1`.
    logit : Tensor or scalar, default None
        The log-odds of sampling `1`.
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    support = NonNegativeInteger()
    arg_constraints = {'prob': Interval(0, 1),
                       'logit': Real()}

    def __init__(self, n=1, prob=None, logit=None, F=None, validate_args=None):
        if (n < 0) or (n % 1 != 0):
            raise ValueError(
                "Expect `n` to be non-negative integer, received n={}".format(n))
        _F = F if F is not None else getF(n, prob, logit)
        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        if prob is not None:
            self.prob = prob
        else:
            self.logit = logit
        self.n = n
        super(Binomial, self).__init__(
            F=_F, event_dim=0, validate_args=validate_args)

    @cached_property
    def prob(self):
        """Get the probability of sampling `1`.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        # pylint: disable=method-hidden
        return logit2prob(self.logit, True, self.F)

    @cached_property
    def logit(self):
        """Get the log-odds of sampling `1`.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        # pylint: disable=method-hidden
        return prob2logit(self.prob, True, self.F)

    @property
    def mean(self):
        return self.n * self.prob

    @property
    def variance(self):
        p = self.prob
        return self.n * p * (1 - p)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        if 'prob' in self.__dict__:
            new_instance.prob = F.np.broadcast_to(self.prob, batch_shape)
        else:
            new_instance.logit = F.np.broadcast_to(self.logit, batch_shape)
        new_instance.n = self.n
        super(Binomial, new_instance).__init__(F=F,
                                               event_dim=self.event_dim,
                                               validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        lgamma = gammaln(F)
        binomal_coef = lgamma(self.n + 1) - lgamma(1 +
                                                   value) - lgamma(self.n - value + 1)
        # log(prob) may have numerical issue.
        unnormalized_log_prob = (value * F.np.log(self.prob) +
                                 (self.n - value) * F.np.log1p(-self.prob))
        return binomal_coef + unnormalized_log_prob

    def sample(self, size=None):
        F = self.F
        if size is not None:
            logit = F.np.broadcast_to(self.logit, size)
        else:
            logit = self.logit
        expanded_logit = F.np.repeat(
            F.np.expand_dims(logit, -1), int(self.n), -1)
        return F.npx.random.bernoulli(logit=expanded_logit).sum(-1)

    def sample_n(self, size=None):
        F = self.F
        logit = self.logit
        expanded_logit = F.np.repeat(
            F.np.expand_dims(logit, -1), int(self.n), -1)
        return F.npx.random.bernoulli(
            logit=expanded_logit,
            size=sample_n_shape_converter(size)
        ).sum(-1)
