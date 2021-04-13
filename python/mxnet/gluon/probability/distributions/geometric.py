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
from .utils import prob2logit, logit2prob, getF, cached_property, sample_n_shape_converter
from .constraint import NonNegativeInteger, Interval, Real


class Geometric(Distribution):
    r"""Create a geometric distribution object.

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
    # pylint: disable=abstract-method

    support = NonNegativeInteger()
    arg_constraints = {'prob': Interval(0, 1),
                       'logit': Real()}

    def __init__(self, prob=None, logit=None, F=None, validate_args=None):
        _F = F if F is not None else getF(prob, logit)
        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        if prob is not None:
            self.prob = prob
        else:
            self.logit = logit
        super(Geometric, self).__init__(
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
        return 1 / self.prob - 1

    @property
    def variance(self):
        return (1 / self.prob - 1) / self.prob

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        if 'prob' in self.__dict__:
            new_instance.prob = F.np.broadcast_to(self.prob, batch_shape)
        else:
            new_instance.logit = F.np.broadcast_to(self.logit, batch_shape)
        super(Geometric, new_instance).__init__(F=F,
                                                event_dim=self.event_dim,
                                                validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        F = self.F
        prob = self.prob
        return value * F.np.log1p(-prob) + F.np.log(prob)

    def sample(self, size=None):
        F = self.F
        if isinstance(self.prob, Number):
            shape_tensor = F.np.zeros(())
        else:
            shape_tensor = F.np.zeros_like(self.prob)
        u = F.np.random.uniform(shape_tensor, size=size)
        samples = F.np.floor(
            F.np.log(u) / F.np.log1p(-self.prob)
        )
        return samples

    def sample_n(self, size=None):
        return self.sample(sample_n_shape_converter(size))

    def entropy(self):
        F = self.F
        logit = self.logit
        prob = self.prob
        return -(logit * (prob - 1) - F.np.log1p(F.np.exp(-logit))) / prob
