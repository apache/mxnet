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
from .utils import prob2logit, logit2prob, getF


class Categorical(Distribution):
    """Create a categorical distribution object.

    Parameters
    ----------
    Distribution : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """

    def __init__(self, prob, logit=None, F=None):
        _F = F if F is not None else getF([prob, logit])
        super(Categorical, self).__init__(F=_F)

        if (prob is None) == (logit is None):
            raise ValueError(
                "Either `prob` or `logit` must be specified, but not both. " +
                "Received prob={}, logit={}".format(prob, logit))

        self._prob = prob
        self._logit = logit

    @property
    def prob(self):
        """Get the probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return self._prob if self._prob is not None else logit2prob(self._logit, False, self.F)

    @property
    def logit(self):
        """Get the log probability of sampling each class.

        Returns
        -------
        Tensor
            Parameter tensor.
        """
        return self._logit if self._logit is not None else prob2logit(self._prob, False, self.F)

    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass

    def log_prob(self, value):
        pass

    def sample(self, size):
        pass
