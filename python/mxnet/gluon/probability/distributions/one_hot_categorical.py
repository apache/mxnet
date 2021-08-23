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
"""One-hot Categorical Distribution"""
__all__ = ['OneHotCategorical']

from .distribution import Distribution
from .categorical import Categorical
from .utils import cached_property
from .constraint import Simplex, Real
from .... import npx


class OneHotCategorical(Distribution):
    """Create a one-hot categorical distribution object.

    Parameters
    ----------
    num_events : Int
        Number of events.
    prob : Tensor
        Probabilities of each event.
    logit : Tensor
        The log-odds of each event
    """
    # pylint: disable=abstract-method

    arg_constraints = {'prob': Simplex(), 'logit': Real()}

    def __init__(self, num_events, prob=None, logit=None, validate_args=None):
        if (num_events > 0):
            num_events = int(num_events)
            self.num_events = num_events
        else:
            raise ValueError("`num_events` should be greater than zero. " +
                             "Received num_events={}".format(num_events))
        self._categorical = Categorical(
            num_events, prob, logit, validate_args)
        super(OneHotCategorical, self).__init__(
            event_dim=1, validate_args=validate_args)

    @cached_property
    def prob(self):
        return self._categorical.prob

    @cached_property
    def logit(self):
        return self._categorical.logit

    @property
    def mean(self):
        return self._categorical.prob

    @property
    def variance(self):
        prob = self.prob
        return prob * (1 - prob)

    def sample(self, size=None):
        indices = self._categorical.sample(size)
        return npx.one_hot(indices, self.num_events)

    def sample_n(self, size=None):
        indices = self._categorical.sample_n(size)
        return npx.one_hot(indices, self.num_events)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        logit = self.logit
        return (value * logit).sum(-1)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        new_instance._categorical = self._categorical.broadcast_to(batch_shape)
        new_instance.num_events = self.num_events
        super(OneHotCategorical, new_instance).__init__(event_dim=self.event_dim,
                                                        validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def enumerate_support(self):
        value = self._categorical.enumerate_support()
        return npx.one_hot(value, self.num_events)
