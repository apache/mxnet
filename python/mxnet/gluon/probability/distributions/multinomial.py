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
"""Multinomial Distribution"""
__all__ = ['Multinomial']

from numbers import Number
from .distribution import Distribution
from .one_hot_categorical import OneHotCategorical
from .utils import getF, cached_property, logit2prob, prob2logit, gammaln
from .constraint import Simplex, Real, IntegerInterval


class Multinomial(Distribution):
    r"""Create a multinomial distribution object.

    Parameters
    ----------
    num_events : int
        number of events.
    prob : Tensor
        probability of each event.
    logit : Tensor
        unnormalized probability of each event.
    total_count : int
        number of trials.
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    arg_constraints = {'prob': Simplex(), 'logit': Real()}

    def __init__(self, num_events,
                 prob=None, logit=None, total_count=1, F=None, validate_args=None):
        _F = F if F is not None else getF(prob, logit)
        if not isinstance(total_count, Number):
            raise ValueError("Expect `total_conut` to be scalar value")
        self.total_count = total_count
        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))
        if prob is not None:
            self.prob = prob
        else:
            self.logit = logit
        self._categorical = OneHotCategorical(
            num_events, prob, logit, F, validate_args)
        super(Multinomial, self).__init__(
            F=_F, event_dim=1, validate_args=validate_args)

    @property
    def mean(self):
        return self.prob * self.total_count

    @property
    def variance(self):
        return self.total_count * self.prob * (1 - self.prob)

    @cached_property
    def prob(self):
        # pylint: disable=method-hidden
        return logit2prob(self.logit, False, self.F)

    @cached_property
    def logit(self):
        # pylint: disable=method-hidden
        return prob2logit(self.prob, False, self.F)

    @property
    def support(self):
        return IntegerInterval(0, self.total_count)

    def sample(self, size=None):
        if size is not None:
            categorical = self._categorical.broadcast_to(size)
        else:
            categorical = self._categorical
        return categorical.sample_n(self.total_count).sum(0)

    def sample_n(self, size=None):
        if isinstance(size, Number):
            size = (size,)
        size = () if size is None else size
        return self._categorical.sample_n((self.total_count,) + size).sum(0)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        lgamma = gammaln(F)
        log_factorial_n = lgamma(value.sum(-1) + 1)
        log_factorial_x = lgamma(value + 1).sum(-1)
        log_power = (self.logit * value).sum(-1)
        return log_factorial_n - log_factorial_x + log_power

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        new_instance._categorical = self._categorical.broadcast_to(batch_shape)
        new_instance.num_events = self.num_events
        new_instance.total_conut = self.total_count
        super(Multinomial, new_instance).__init__(F=F,
                                                  event_dim=self.event_dim,
                                                  validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance
