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
"""Multivariate Normal Distribution"""
__all__ = ['MultivariateNormal']

import math
from .distribution import Distribution
from .constraint import Real, PositiveDefinite, LowerCholesky
from .utils import cached_property
from .... import np


class MultivariateNormal(Distribution):
    r"""Create a multivaraite Normal distribution object.

    Parameters
    ----------
    loc : Tensor
        mean of the distribution.
    cov : Tensor
        covariance matrix of the distribution
    precision : Tensor
        precision matrix of the distribution
    scale_tril : Tensor
        lower-triangular factor of the covariance
    """
    # pylint: disable=abstract-method

    has_grad = True
    support = Real()
    arg_constraints = {'loc': Real(),
                       'cov': PositiveDefinite(),
                       'precision': PositiveDefinite(),
                       'scale_tril': LowerCholesky()}

    def __init__(self, loc, cov=None, precision=None, scale_tril=None, validate_args=None):
        if (cov is not None) + (precision is not None) + (scale_tril is not None) != 1:
            raise ValueError("Exactly one onf `cov` or `precision` or " +
                             "`scale_tril` may be specified")
        self.loc = loc
        if cov is not None:
            self.cov = cov
        elif precision is not None:
            self.precision = precision
        else:
            self.scale_tril = scale_tril
        super(MultivariateNormal, self).__init__(
            event_dim=1, validate_args=validate_args)

    def _precision_to_scale_tril(self, P):
        """
        P = inv(L * L.T) = inv(L.T) * inv(L)
        flip(P) = flip(inv(L.T)) * flip(inv(L))
        flip(inv(L.T)) = Cholesky(flip(P))
        L = flip(Cholesky(flip(P))).T
        """
        L_flip_inv_T = np.linalg.cholesky(np.flip(P, (-1, -2)))
        L = np.linalg.inv(np.swapaxes(
            np.flip(L_flip_inv_T, (-1, -2)), -1, -2))
        return L

    @cached_property
    def scale_tril(self):
        # pylint: disable=method-hidden
        if 'cov' in self.__dict__:
            return np.linalg.cholesky(self.cov)
        return self._precision_to_scale_tril(self.precision)

    @cached_property
    def cov(self):
        # pylint: disable=method-hidden
        if 'scale_tril' in self.__dict__:
            scale_triu = np.swapaxes(self.scale_tril, -1, -2)
            return np.matmul(self.scale_tril, scale_triu)
        return np.linalg.inv(self.precision)

    @cached_property
    def precision(self):
        # pylint: disable=method-hidden
        if 'cov' in self.__dict__:
            return np.linalg.inv(self.cov)
        scale_tril_inv = np.linalg.inv(self.scale_tril)
        scale_triu_inv = np.swapaxes(scale_tril_inv, -1, -2)
        return np.matmul(scale_triu_inv, scale_tril_inv)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (self.scale_tril ** 2).sum(-1)

    def sample(self, size=None):
        # symbol does not support `np.broadcast`
        shape_tensor = self.loc + self.scale_tril.sum(-1)
        if size is not None:
            if isinstance(size, int):
                size = (size,)
            shape_tensor = np.broadcast_to(shape_tensor, size + (-2,))
        noise = np.random.normal(np.zeros_like(
            shape_tensor), np.ones_like(shape_tensor))
        samples = self.loc + \
            np.einsum('...jk,...j->...k', self.scale_tril, noise)
        return samples

    def sample_n(self, size=None):
        if size is None:
            return self.sample()
        # symbol does not support `np.broadcast`
        shape_tensor = self.loc + self.scale_tril[..., 0]
        if isinstance(size, int):
            size = (size,)
        noise = np.random.normal(np.zeros_like(shape_tensor), np.ones_like(shape_tensor),
                                 (-2,) + size)
        samples = self.loc + \
            np.einsum('...jk,...j->...k', self.scale_tril, noise)
        return samples

    def log_prob(self, value):
        if self._validate_args:
            self._validate_samples(value)
        diff = value - self.loc
        # diff.T * inv(\Sigma) * diff
        M = np.einsum(
            '...i,...i->...',
            diff,
            np.einsum('...jk,...j->...k', self.precision,
                      diff)  # Batch matrix vector multiply
        ) * -0.5
        #   (2 * \pi)^{-k/2} * det(\Sigma)^{-1/2}
        # = det(2 * \pi * L * L.T)^{-1/2}
        # = det(\sqrt(2 * \pi) * L)^{-1}
        half_log_det = np.log(
            np.diagonal(np.sqrt(2 * math.pi) *
                        self.scale_tril, axis1=-2, axis2=-1)
        ).sum(-1)
        return M - half_log_det

    def entropy(self):
        #   det(2 * \pi * e * \Sigma)
        # = det(\sqrt(2 * \pi * e) * L)^2
        return np.log(np.diagonal(
            np.sqrt(2 * math.pi * math.e) * self.scale_tril,
            axis1=-2, axis2=-1
        )).sum(-1)
