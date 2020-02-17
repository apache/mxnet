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
from .utils import prob2logit, logit2prob, getF, cached_property
from .constraint import Simplex, Real


# FIXME: Finish implementation
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
     F : mx.ndarray or mx.symbol.numpy._Symbol or None
        Variable recording running mode, will be automatically
        inferred from parameters if declared None.
    """

    arg_constraints = {'prob': Simplex(),
                       'logit': Real()}

    def __init__(self, prob=None, logit=None, F=None, validate_args=None):
        _F = F if F is not None else getF([prob, logit])

        # if (num_events > 0):
        #     num_events = int(num_events)
        #     self.num_events = num_events
        # else:
        #     raise ValueError("`num_events` should be greater than zero. " +
        #                      "Received num_events={}".format(num_events))

        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        if prob is not None:
            self.prob = prob
        else:
            self.logit = logit

        super(Categorical, self).__init__(F=_F, event_dim=0, validate_args=validate_args)

    @cached_property
    def prob(self):
        """Get the probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return logit2prob(self.logit, False, self.F)

    @cached_property
    def logit(self):
        """Get the log probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return prob2logit(self.prob, False, self.F)

    def log_prob(self, value):
        logit = self.logit
        return (logit * value).sum(-1)

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
        F = self.F
        if size is None:
            size = ()
            logit = self.logit
        else:
            if not isinstance(size, tuple):
                logit = F.np.broadcast_to(self.logit, (size) + (-2,))
            else:
                logit = F.np.broadcast_to(self.logit, size + (-2,))
        gumbel_noise = F.np.random.gumbel(F.np.zeros_like(logit))
        return F.np.argmax(gumbel_noise + logit, axis=-1)
