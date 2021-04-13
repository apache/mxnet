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
"""Independent class."""
__all__ = ['Independent']

from .distribution import Distribution
from .constraint import dependent_property
from .utils import sum_right_most


class Independent(Distribution):
    r"""
    Reinterprets some collection of independent, non-identical distributions as
    a single multivariate random variable (convert some `batch_dim` to `event_dim`).
    """
    # pylint: disable=abstract-method

    arg_constraints = {}

    def __init__(self, base_distribution, reinterpreted_batch_ndims, validate_args=None):
        event_dim = reinterpreted_batch_ndims + base_distribution.event_dim
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super(Independent, self).__init__(F=base_distribution.F,
                                          event_dim=event_dim,
                                          validate_args=validate_args)

    def broadcast_to(self, batch_shape):
        new_instance = self.__new__(type(self))
        F = self.F
        # we use -2 to copy the sizes of reinterpreted batch dimensions
        reinterpreted_axes = (-2,) * self.reinterpreted_batch_ndims
        new_instance.base_dist = self.base_dist.broadcast_to(
            batch_shape + reinterpreted_axes)
        new_instance.reinterpreted_batch_ndims = self.reinterpreted_batch_ndims
        super(Independent, new_instance).__init__(F=F, event_dim=self.event_dim,
                                                  validate_args=False)
        new_instance._validate_args = self._validate_args
        return new_instance

    @property
    def has_enumerate_support(self):
        if self.reinterpreted_batch_ndims > 0:
            return False
        return self.base_dist.has_enumerate_support

    @dependent_property
    def support(self):
        return self.base_dist.support

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, size=None):
        return self.base_dist.sample(size)

    def sample_n(self, size):
        return self.base_dist.sample_n(size)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        return sum_right_most(log_prob, self.reinterpreted_batch_ndims)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return sum_right_most(entropy, self.reinterpreted_batch_ndims)

    def enumerate_support(self):
        if self.reinterpreted_batch_ndims > 0:
            raise NotImplementedError(
                "Enumeration over cartesian product is not implemented")
        return self.base_dist.enumerate_support()
