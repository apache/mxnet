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
"""Geometric distribution class."""
__all__ = ['Geometric']

from numbers import Number
from .distribution import Distribution
from .utils import prob2logit, logit2prob, cached_property, sample_n_shape_converter
from .constraint import NonNegativeInteger, Interval, Real
from .... import np


class Geometric(Distribution):
    r"""Create a geometric distribution object.

    Parameters
    ----------
    prob : Tensor or scalar, default None
        Probability of sampling `1`.
    logit : Tensor or scalar, default None
        The log-odds of sampling `1`.
    """
    # pylint: disable=abstract-method

    support = NonNegativeInteger()
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
        super(Geometric, self).__init__(
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
        return 1 / self.prob - 1

    @property
    def variance(self):
        return (1 / self.prob - 1) / self.prob

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        if 'prob' in self.__dict__:
            new_instance.prob = np.broadcast_to(self.prob, batch_shape)
        else:
            new_instance.logit = np.broadcast_to(self.logit, batch_shape)
        super(Geometric, new_instance).__init__(event_dim=self.event_dim,
                                                validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        prob = self.prob
        return value * np.log1p(-prob) + np.log(prob)

    def sample(self, size=None):
        if isinstance(self.prob, Number):
            shape_tensor = np.zeros(())
        else:
            shape_tensor = np.zeros_like(self.prob)  # pylint: disable=too-many-function-args
        u = np.random.uniform(shape_tensor, size=size)
        samples = np.floor(
            np.log(u) / np.log1p(-self.prob)
        )
        return samples

    def sample_n(self, size=None):
        return self.sample(sample_n_shape_converter(size))

    def entropy(self):
        logit = self.logit
        prob = self.prob
        return -(logit * (prob - 1) - np.log1p(np.exp(-logit))) / prob
