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
__all__ = ['RelaxedOneHotCategorical']

from math import lgamma
from .distribution import Distribution
from .transformed_distribution import TransformedDistribution
from ..transformation import ExpTransform
from .utils import prob2logit, logit2prob, getF, cached_property
from .constraint import Real, Simplex


class _LogRelaxedOneHotCategorical(Distribution):
    """Helper class for creating the log of a
    categorical distribution object.

    Parameters
    ----------
    T : scalar, default None
        Relaxation temperature
    num_events : Int
        Number of events.
    prob : Tensor
        Probabilities of each event.
    logit : Tensor
        The log-odds of each event
     F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    has_grad = True
    arg_constraints = {'prob': Simplex(),
                       'logit': Real()}

    def __init__(self, T, num_events, prob=None, logit=None, F=None, validate_args=None):
        self.T = T
        _F = F if F is not None else getF(prob, logit)
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

        super(_LogRelaxedOneHotCategorical, self).__init__(
            _F, event_dim=1, validate_args=validate_args)

    @cached_property
    def prob(self):
        """Get the probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        # pylint: disable=method-hidden
        return logit2prob(self.logit, False, self.F)

    @cached_property
    def logit(self):
        """Get the log probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        # pylint: disable=method-hidden
        return prob2logit(self.prob, False, self.F)

    def log_prob(self, value):
        """Compute the log-likelihood of `value`

        Parameters
        ----------
        value : Tensor
            samples from Relaxed Categorical distribution

        Returns
        -------
        Tensor
            log-likelihood of `value`
        """
        F = self.F
        K = self.num_events  # Python scalar
        log = F.np.log
        exp = F.np.exp
        logit = self.logit
        y = logit - value * self.T
        log_sum_exp = log(exp(y).sum(-1, keepdims=True) + 1e-20)
        log_scale = lgamma(K) - log(self.T) * (-(K - 1))
        return (y - log_sum_exp).sum(-1) + log_scale

    def sample(self, size=None):
        F = self.F
        if size is None:
            size = ()
            logit = self.logit
        else:
            if isinstance(size, int):
                logit = F.np.broadcast_to(self.logit, (size) + (-2,))
            else:
                logit = F.np.broadcast_to(self.logit, size + (-2,))
        scores = F.np.random.gumbel(logit) / self.T
        return F.np.log(F.npx.softmax(scores, axis=-1) + 1e-20)


class RelaxedOneHotCategorical(TransformedDistribution):
    """Create a relaxed one hot categorical distribution object.

    Parameters
    ----------
    T : scalar, default None
        Relaxation temperature
    num_events : Int
        Number of events.
    prob : Tensor
        Probabilities of each event.
    logit : Tensor
        The log-odds of each event
     F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """
    # pylint: disable=abstract-method

    has_grad = True
    arg_constraints = {'prob': Simplex(),
                       'logit': Real()}

    def __init__(self, T, num_events, prob=None, logit=None, F=None, validate_args=None):
        base_dist = _LogRelaxedOneHotCategorical(
            T, num_events, prob, logit, F, validate_args)
        super(RelaxedOneHotCategorical, self).__init__(
            base_dist, ExpTransform())

    @property
    def T(self):
        return self._base_dist.T

    @property
    def prob(self):
        return self._base_dist.prob

    @property
    def logit(self):
        return self._base_dist.logit
