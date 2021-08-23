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
"""Negative binomial distribution class."""
__all__ = ['NegativeBinomial']

from .distribution import Distribution
from .poisson import Poisson
from .gamma import Gamma
from .utils import prob2logit, logit2prob, cached_property
from .utils import gammaln
from .constraint import GreaterThanEq, Interval, Real, NonNegativeInteger
from .... import np


class NegativeBinomial(Distribution):
    r"""Create a negative binomial distribution object.

    Parameters
    ----------
    n : Tensor or scalar
        Non-negative number of negative Bernoulli trials to stop.
    prob : Tensor or scalar, default None
        Probability of sampling `1`.
    logit : Tensor or scalar, default None
        The log-odds of sampling `1`.
    """
    # pylint: disable=abstract-method

    support = NonNegativeInteger()
    arg_constraints = {'n': GreaterThanEq(0),
                       'prob': Interval(0, 1),
                       'logit': Real()}

    def __init__(self, n, prob=None, logit=None, validate_args=None):
        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        if prob is not None:
            self.prob = prob
        else:
            self.logit = logit
        self.n = n
        super(NegativeBinomial, self).__init__(
            event_dim=0, validate_args=validate_args)

    @cached_property
    def prob(self):
        """Get the probability of sampling `1`.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        # pylint: disable=method-hidden
        return logit2prob(self.logit, True)

    @cached_property
    def logit(self):
        """Get the log-odds of sampling `1`.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        # pylint: disable=method-hidden
        return prob2logit(self.prob, True)

    @property
    def mean(self):
        return self.n * np.exp(self.logit)

    @property
    def variance(self):
        prob = self.prob
        return self.n * prob / (1 - prob) ** 2

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        if 'prob' in self.__dict__:
            new_instance.prob = np.broadcast_to(self.prob, batch_shape)
        else:
            new_instance.logit = np.broadcast_to(self.logit, batch_shape)
        new_instance.n = np.broadcast_to(self.n, batch_shape)
        super(NegativeBinomial, new_instance).__init__(event_dim=self.event_dim,
                                                       validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        lgamma = gammaln()
        binomal_coef = lgamma(value + self.n) - \
            lgamma(1 + value) - lgamma(self.n)
        # log(prob) may have numerical issue.
        unnormalized_log_prob = self.n * \
            np.log(self.prob) + value * np.log1p(-self.prob)
        return binomal_coef + unnormalized_log_prob

    def sample(self, size=None):
        # Sample via Poisson-Gamma mixture
        rate = Gamma(shape=self.n, scale=np.exp(
            self.logit)).sample(size)
        return Poisson(rate).sample()

    def sample_n(self, size=None):
        # Sample via Poisson-Gamma mixture
        rate = Gamma(shape=self.n, scale=np.exp(
            self.logit)).sample_n(size)
        return Poisson(rate).sample()
