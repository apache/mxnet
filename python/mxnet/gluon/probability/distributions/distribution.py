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
"""Base distribution class."""
__all__ = ['Distribution']

from numbers import Number
from .utils import cached_property


class Distribution(object):
    r"""Base class for distribution.

    Parameters
    ----------
    F : mx.ndarray or mx.symbol.numpy._Symbol
        Variable that stores the running mode.
    event_dim : int, default None
        Variable indicating the dimension of the distribution's support.
    validate_args : bool, default None
        Whether to validate the distribution parameters
    """

    # Variable indicating whether the sampling method has
    # pathwise gradient.
    has_grad = False
    support = None
    has_enumerate_support = False
    arg_constraints = {}
    _validate_args = False

    @staticmethod
    def set_default_validate_args(value):
        if value not in [True, False]:
            raise ValueError
        Distribution._validate_args = value

    def __init__(self, F=None, event_dim=None, validate_args=None):
        self.F = F
        self.event_dim = event_dim
        if validate_args is not None:
            self._validate_args = validate_args
        if self._validate_args:
            for param, constraint in self.arg_constraints.items():
                if param not in self.__dict__ and isinstance(getattr(type(self), param),
                                                             cached_property):
                    # skip param that is decorated by cached_property
                    continue
                setattr(self, param, constraint.check(getattr(self, param)))
        super(Distribution, self).__init__()

    def log_prob(self, value):
        r"""
        Returns the log of the probability density/mass function evaluated at `value`.
        """
        raise NotImplementedError()

    def pdf(self, value):
        r"""
        Returns the probability density/mass function evaluated at `value`.
        """
        return self.F.np.exp(self.log_prob(value))

    def cdf(self, value):
        r"""
        Returns the cumulative density/mass function evaluated at `value`.
        """
        raise NotImplementedError

    def icdf(self, value):
        r"""
        Returns the inverse cumulative density/mass function evaluated at `value`.
        """
        raise NotImplementedError

    def sample(self, size=None):
        r"""
        Generates a `shape` shaped sample.
        """
        raise NotImplementedError

    def sample_n(self, size):
        r"""
        Generate samples of (n + parameter_shape) from the distribution.
        """
        raise NotImplementedError

    def broadcast_to(self, batch_shape):
        r"""
        Returns a new distribution instance with parameters expanded
        to `batch_shape`. This method calls `numpy.broadcast_to` on
        the parameters.

        Parameters
        ----------
        batch_shape : Tuple
            The batch shape of the desired distribution.

        """
        raise NotImplementedError

    def enumerate_support(self):
        r"""
        Returns a tensor that contains all values supported
        by a discrete distribution.
        """
        raise NotImplementedError

    @property
    def arg_constraints(self):
        """
        Returns a dictionary from parameter names to
        :class:`~mxnet.gluon.probability.distributions.constraint.Constraint` objects that
        should be satisfied by each parameter of this distribution. Args that
        are not ndarray/symbol need not appear in this dict.
        """
        # pylint: disable=function-redefined
        raise NotImplementedError

    @property
    def mean(self):
        r"""
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def variance(self):
        r"""
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()

    @property
    def support(self):
        r"""
        Returns a function representing the distribution's support.
        """
        # pylint: disable=function-redefined
        raise NotImplementedError

    def entropy(self):
        r"""
        Returns entropy of distribution.
        """
        raise NotImplementedError

    def perplexity(self):
        r"""
        Returns perplexity of distribution.
        """
        F = self.F
        return F.np.exp(self.entropy())

    def __repr__(self):
        mode = self.F
        args_string = ''
        if 'symbol' not in mode.__name__:
            for k, _ in self.arg_constraints.items():
                v = self.__dict__[k]
                if isinstance(v, Number):
                    shape_v = ()
                else:
                    shape_v = v.shape
                args_string += '{}: size {}'.format(k, shape_v) + ', '
        args_string += ', '.join(['F: {}'.format(mode.__name__),
                                  'event_dim: {}'.format(self.event_dim)])
        return self.__class__.__name__ + '(' + args_string + ')'

    def _validate_samples(self, value):
        """
        Validate samples for methods like `log_prob`, `cdf`.
        Check if `value` lies in `self.support`
        """
        return self.support.check(value)
