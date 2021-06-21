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
from .utils import prob2logit, logit2prob, cached_property, sample_n_shape_converter
from .constraint import Boolean, Interval, Real
from .... import np, npx


class Bernoulli(ExponentialFamily):
    r"""Create a bernoulli distribution object.

    Parameters
    ----------
    prob : Tensor or scalar, default None
        Probability of sampling `1`.
    logit : Tensor or scalar, default None
        The log-odds of sampling `1`.
    """
    # pylint: disable=abstract-method

    support = Boolean()
    arg_constraints = {'prob': Interval(0, 1),
                       'logit': Real()}

    def __init__(self, prob=None, logit=None, validate_args=None):
        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        if prob is not None:
            self.prob = prob
        else:
            self.logit = logit

        super(Bernoulli, self).__init__(
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
        return self.prob

    @property
    def variance(self):
        return self.prob * (1 - self.prob)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        if 'prob' in self.__dict__:
            new_instance.prob = np.broadcast_to(self.prob, batch_shape)
        else:
            new_instance.logit = np.broadcast_to(self.logit, batch_shape)
        super(Bernoulli, new_instance).__init__(F=F,
                                                event_dim=self.event_dim,
                                                validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        if self.prob is None:
            logit = self.logit
            return logit * (value - 1) - np.log(np.exp(-logit) + 1)
        else:
            # Parameterized by probability
            eps = 1e-12
            return (np.log(self.prob + eps) * value
                    + np.log1p(-self.prob + eps) * (1 - value))

    def sample(self, size=None):
        return npx.random.bernoulli(self.prob, self.logit, size)

    def sample_n(self, size=None):
        return npx.random.bernoulli(self.prob, self.logit, sample_n_shape_converter(size))

    @property
    def _natural_params(self):
        return (self.logit,)

    def _log_normalizer(self, x):
        # pylint: disable=arguments-differ
        return np.log(1 + np.exp(x))

    def entropy(self):
        logit = self.logit
        prob = self.prob
        return -(logit * (prob - 1) - np.log(np.exp(-logit) + 1))
