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
# pylint: disable=abstract-method
# pylint: disable=arguments-differ
"""Transformation Classes"""
__all__ = ["Transformation", "TransformBlock", "ComposeTransform", "ExpTransform",
           "AffineTransform", "PowerTransform", "AbsTransform", 'SigmoidTransform',
           'SoftmaxTransform']

import weakref
from ..distributions.utils import _clip_prob, cached_property, sum_right_most
from ...block import HybridBlock
from .... import ndarray as nd


class Transformation(object):
    r"""Abstract class for implementing invertible transformation
    with computable log  det jacobians

    Attributes
    ----------
    bijective : bool

    """
    bijective = False
    event_dim = 0

    def __init__(self, F=nd):
        self._inv = None
        self._F = F
        super(Transformation, self).__init__()

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value

    @property
    def sign(self):
        """
        Returns the sign of the determinant of the Jacobian.
        """
        raise NotImplementedError

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseTransformation(self)
            self._inv = weakref.ref(inv)
        return inv

    def __call__(self, x):
        return self._forward_compute(x)

    def _inv_call(self, y):
        return self._inverse_compute(y)

    def _forward_compute(self, x):
        raise NotImplementedError

    def _inverse_compute(self, x):
        raise NotImplementedError

    def log_det_jacobian(self, x, y):
        """
        Compute the value of log(|dy/dx|)
        """
        raise NotImplementedError


class _InverseTransformation(Transformation):
    """
    A private class representing the invert of `Transformation`,
    which should be accessed through `Transformation.inv` property.
    """

    def __init__(self, forward_transformation):
        super(_InverseTransformation, self).__init__()
        self._inv = forward_transformation

    @property
    def inv(self):
        return self._inv

    @property
    def sign(self):
        return self._inv.sign

    @property
    def event_dim(self):
        return self._inv.event_dim

    def __call__(self, x):
        return self._inv._inverse_compute(x)

    def log_det_jacobian(self, x, y):
        return -self._inv.log_det_jacobian(y, x)


class TransformBlock(Transformation, HybridBlock):
    """Transform with learnable parameters should inherit from this class
    rather than `Transformation`.
    For example: normalization flow.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ComposeTransform(Transformation):
    r"""
    Composes multiple transforms in a chain.
    """
    def __init__(self, parts):
        super(ComposeTransform, self).__init__()
        self._parts = parts

    def _forward_compute(self, x):
        for t in self._parts:
            x = t(x)
        return x

    @property
    def F(self):
        return self._parts[0].F

    @F.setter
    def F(self, value):
        for t in self._parts:
            t.F = value

    # @cached_property is, in essence, @property with lazy evaluation.
    # pylint: disable=invalid-overridden-method
    @cached_property
    def sign(self):
        sign = 1
        for p in self._parts:
            sign = sign * p.sign
        return sign

    @cached_property
    def event_dim(self):
        return max(p.event_dim for p in self._parts) if self._parts else 0

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = ComposeTransform([t.inv for t in reversed(self._parts)])
            self._inv = weakref.ref(inv)
            inv._inv = weakref.ref(self)
        return inv

    def log_det_jacobian(self, x, y):
        if not self._parts:
            return self.F.np.zeros_like(x)
        result = 0
        x_prime = None
        for t in self._parts[:-1]:
            x_prime = t(x)
            result = result + sum_right_most(t.log_det_jacobian(x, x_prime),
                                             self.event_dim - t.event_dim)
            x = x_prime
        t_last = self._parts[-1]
        result = result + sum_right_most(t_last.log_det_jacobian(x, y),
                                         self.event_dim - t_last.event_dim)

        return result


class ExpTransform(Transformation):
    r"""
    Perform the exponential transform: y = exp{x}.
    """
    bijective = True
    sign = 1

    def _forward_compute(self, x):
        return self.F.np.exp(x)

    def _inverse_compute(self, y):
        return self.F.np.log(y)

    def log_det_jacobian(self, x, y):
        return x


class AffineTransform(Transformation):
    r"""
    Perform *pointwise* affine transform: y = loc + scale * x.
    """
    bijective = True

    def __init__(self, loc, scale, event_dim=0):
        super(AffineTransform, self).__init__()
        self._loc = loc
        self._scale = scale
        self.event_dim = event_dim

    def _forward_compute(self, x):
        return self._loc + self._scale * x

    def _inverse_compute(self, y):
        return (y - self._loc) / self._scale

    def log_det_jacobian(self, x, y):
        abs_fn = self.F.np.abs
        log_fn = self.F.np.log
        ones_fn = self.F.np.ones_like
        # element-wise abs(log(dy/dx))
        value = ones_fn(x) * log_fn(abs_fn(self._scale))
        return sum_right_most(value, self.event_dim)

    @property
    def sign(self):
        return self.F.np.sign(self._scale)


class PowerTransform(Transformation):
    r"""
    Perform *pointwise* power transform: y = pow(x, exponent).
    """
    bijective = True
    sign = 1

    def __init__(self, exponent):
        super(PowerTransform, self).__init__()
        self._exponent = exponent

    def _forward_compute(self, x):
        return self.F.np.power(x, self._exponent)

    def _inverse_compute(self, y):
        return self.F.np.power(y, 1 / self._exponent)

    def log_det_jacobian(self, x, y):
        log_fn = self.F.np.log
        abs_fn = self.F.np.abs
        return log_fn(abs_fn(self._exponent * y / x))


class SigmoidTransform(Transformation):
    r"""
    Perform *pointwise* sigmoid transform: y = 1 / (1 + exp(-x)).
    """
    bijective = True
    sign = 1

    def _forward_compute(self, x):
        F = self.F
        return _clip_prob(F.npx.sigmoid(x), F)

    def _inverse_compute(self, y):
        F = self.F
        clipped_prob = _clip_prob(y, F)
        return F.np.log(clipped_prob) - F.np.log1p(-clipped_prob)

    def log_det_jacobian(self, x, y):
        F = self.F
        log = F.np.log
        exp = F.np.exp
        softplus_fn = lambda x: log(1 + exp(x))
        return -softplus_fn(-x) - softplus_fn(x)


class SoftmaxTransform(Transformation):
    event_dim = 1

    def _forward_compute(self, x):
        return self.F.npx.softmax(x, -1)

    def _inverse_compute(self, y):
        return self.F.log(y)


class AbsTransform(Transformation):
    def _forward_compute(self, x):
        return self.F.np.abs(x)

    def _inverse_compute(self, y):
        return y
