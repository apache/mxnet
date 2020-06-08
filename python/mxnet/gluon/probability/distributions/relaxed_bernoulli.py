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
"""Relaxed Bernoulli class."""
__all__ = ['RelaxedBernoulli']

from .distribution import Distribution
from .transformed_distribution import TransformedDistribution
from ..transformation import SigmoidTransform
from .utils import prob2logit, logit2prob, getF, cached_property
from .constraint import OpenInterval, Real, Interval


class _LogitRelaxedBernoulli(Distribution):
    r"""Helper class for creating an unnormalized relaxed Bernoulli object.

    Parameters
    ----------
    T : scalar, default None
        Relaxation temperature
    prob : Tensor or scalar, default None
        Probability of sampling `1`.
    logit : Tensor or scalar, default None
        The log-odds of sampling `1`.
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    has_grad = True
    support = Real()
    arg_constraints = {'prob': Interval(0, 1),
                       'logit': Real()}

    def __init__(self, T, prob=None, logit=None, F=None, validate_args=None):
        _F = F if F is not None else getF(prob, logit)
        self.T = T
        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))
        if prob is not None:
            self.prob = prob
        else:
            self.logit = logit
        super(_LogitRelaxedBernoulli, self).__init__(
            F=_F, event_dim=0, validate_args=validate_args
        )

    @cached_property
    def prob(self):
        # pylint: disable=method-hidden
        return logit2prob(self.logit, True, self.F)

    @cached_property
    def logit(self):
        # pylint: disable=method-hidden
        return prob2logit(self.prob, True, self.F)

    def sample(self, size=None):
        F = self.F
        logit = self.logit
        return F.np.random.logistic(loc=logit, scale=1, size=size) / self.T

    def log_prob(self, value):
        F = self.F
        # log-likelihood of `value` from (Logistic(logit, 1) / T)
        diff = self.logit - self.T * value
        return F.np.log(self.T) + diff - 2 * F.np.log1p(F.np.exp(diff))


class RelaxedBernoulli(TransformedDistribution):
    r"""Create a relaxed Bernoulli distribution object.

    Parameters
    ----------
    T : scalar, default None
        Relaxation temperature
    prob : Tensor or scalar, default None
        Probability of sampling `1`.
    logit : Tensor or scalar, default None
        The log-odds of sampling `1`.
    F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    has_grad = True
    support = OpenInterval(0, 1)
    arg_constraints = {'prob': Interval(0, 1),
                       'logit': Real()}

    def __init__(self, T, prob=None, logit=None, F=None, validate_args=None):
        base_dist = _LogitRelaxedBernoulli(T, prob, logit, F, validate_args)
        super(RelaxedBernoulli, self).__init__(base_dist, SigmoidTransform())

    @property
    def T(self):
        return self._base_dist.T

    @property
    def prob(self):
        return self._base_dist.prob

    @property
    def logit(self):
        return self._base_dist.logit

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        if 'prob' in self.__dict__:
            new_instance.prob = F.np.broadcast_to(self.prob, batch_shape)
        else:
            new_instance.logit = F.np.broadcast_to(self.logit, batch_shape)
        super(RelaxedBernoulli, new_instance).__init__(F=F,
                                                       event_dim=self.event_dim,
                                                       validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance
