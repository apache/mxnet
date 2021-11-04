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
"""Categorical class."""
__all__ = ['Categorical']

from .distribution import Distribution
from .utils import prob2logit, logit2prob, cached_property, sample_n_shape_converter
from .constraint import Simplex, Real, IntegerInterval
from .... import np, npx


class Categorical(Distribution):
    """Create a categorical distribution object.

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

    has_enumerate_support = True
    arg_constraints = {'prob': Simplex(),
                       'logit': Real()}

    def __init__(self, num_events, prob=None, logit=None, validate_args=None):
        if (num_events > 0):
            num_events = int(num_events)
            self.num_events = num_events
        else:
            raise ValueError("`num_events` should be greater than zero. " +
                             "Received num_events={}".format(num_events))
        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        if prob is not None:
            self.prob = prob
        else:
            self.logit = logit

        super(Categorical, self).__init__(
            event_dim=0, validate_args=validate_args)

    @cached_property
    def prob(self):
        # pylint: disable=method-hidden
        """Get the probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return logit2prob(self.logit, False)

    @cached_property
    def logit(self):
        # pylint: disable=method-hidden
        """Get the log probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return prob2logit(self.prob, False)

    @property
    def support(self):
        return IntegerInterval(0, self.num_events)

    def log_prob(self, value):
        """Compute the log-likelihood of `value`

        Parameters
        ----------
        value : Tensor
            samples from Categorical distribution

        Returns
        -------
        Tensor
            log-likelihood of `value`
        """
        if self._validate_args:
            self._validate_samples(value)
        logit = self.logit
        indices = np.expand_dims(value, -1).astype('int')
        expanded_logit = logit * np.ones_like(logit + indices)  # pylint: disable=too-many-function-args
        return npx.pick(expanded_logit, indices).squeeze()

    def sample(self, size=None):
        """Sample from categorical distribution.
        Given logit/prob of size `(batch_size, num_events)`,
        `batch_size` samples will be drawn.
        If `size` is given, `np.broadcast(size, batch_size)` samples will be drawn.

        Parameters
        ----------
        size : int or tuple of ints

        Returns
        -------
        out : Tensor
            Samples from the categorical distribution.
        """
        if size is None:
            size = ()
            logit = self.logit
        else:
            if isinstance(size, int):
                logit = np.broadcast_to(self.logit, (size,) + (-2,))
            else:
                logit = np.broadcast_to(self.logit, size + (-2,))
        gumbel_samples = np.random.gumbel(logit)
        return np.argmax(gumbel_samples, axis=-1)

    def sample_n(self, size=None):
        size = sample_n_shape_converter(size)
        gumbel_samples = np.random.gumbel(self.logit, size=size)
        return np.argmax(gumbel_samples, axis=-1)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        new_instance.prob = np.broadcast_to(self.prob, batch_shape + (-2,))
        new_instance.logit = np.broadcast_to(self.logit, batch_shape + (-2,))
        new_instance.num_events = self.num_events
        super(Categorical, new_instance).__init__(event_dim=self.event_dim,
                                                  validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def enumerate_support(self):
        num_events = self.num_events
        value = npx.arange_like(self.logit) % num_events
        return np.moveaxis(value, -1, 0)
