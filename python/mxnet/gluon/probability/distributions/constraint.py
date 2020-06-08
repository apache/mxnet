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
"""Base class and implementations of constraint"""
__all__ = ["Constraint", "Real", "Boolean",
           "Interval", "OpenInterval", "HalfOpenInterval", "UnitInterval",
           "IntegerInterval", "IntegerOpenInterval", "IntegerHalfOpenInterval",
           "GreaterThan", "GreaterThanEq", "IntegerGreaterThan", "IntegerGreaterThanEq",
           "LessThan", "LessThanEq", "IntegerLessThan", "IntegerLessThanEq",
           "Positive", "NonNegative", "PositiveInteger", "NonNegativeInteger",
           "Simplex", "LowerTriangular", "LowerCholesky", "PositiveDefinite",
           "Cat", "Stack"]

from .utils import getF, constraint_check
from .... import ndarray as nd


class Constraint(object):
    """Base class for constraints.

    A constraint object represents a region over which a variable
    is valid.
    """

    def check(self, value):
        """Check if `value` satisfies the constraint,
        return the origin value if valid,
        raise `ValueError` with given message otherwise.

        Parameters
        ----------
        value : Tensor
            Input tensor to be checked.
        """
        raise NotImplementedError


class _Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    """

    def check(self, value):
        raise ValueError('Cannot validate dependent constraint')


def is_dependent(constraint):
    return isinstance(constraint, _Dependent)


class _DependentProperty(property, _Dependent):
    """
    Decorator that extends @property to act like a `_Dependent` constraint when
    called on a class and act like a property when called on an object.
    Example::
        class Uniform(Distribution):
            def __init__(self, low, high):
                self.low = low
                self.high = high
            @constraint.dependent_property
            def support(self):
                return constraint.Interval(self.low, self.high)
    """
    pass # pylint: disable=unnecessary-pass


class Real(Constraint):
    """
    Constrain to be a real number. (exclude `np.nan`)
    """

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be a real tensor".format(
            value)
        # False when value has NANs
        condition = (value == value) # pylint: disable=comparison-with-itself
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class Boolean(Constraint):
    """
    Constrain to `{0, 1}`.
    """

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be either 0 or 1.".format(
            value)
        condition = (value == 0) | (value == 1)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`
    """

    def __init__(self, lower_bound, upper_bound):
        super(Interval, self).__init__()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be >= {} and <= {}.".format(
            value, self._lower_bound, self._upper_bound)
        condition = (value >= self._lower_bound) & (value <= self._upper_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class OpenInterval(Constraint):
    """
    Constrain to a real interval `(lower_bound, upper_bound)`
    """

    def __init__(self, lower_bound, upper_bound):
        super(OpenInterval, self).__init__()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be > {} and < {}.".format(
            value, self._lower_bound, self._upper_bound)
        condition = (value > self._lower_bound) & (value < self._upper_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class HalfOpenInterval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound)`
    """

    def __init__(self, lower_bound, upper_bound):
        super(HalfOpenInterval, self).__init__()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be >= {} and < {}.".format(
            value, self._lower_bound, self._upper_bound)
        condition = (value >= self._lower_bound) & (value < self._upper_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class IntegerInterval(Constraint):
    """
    Constrain to an integer interval `[lower_bound, upper_bound]`
    """

    def __init__(self, lower_bound, upper_bound):
        super(IntegerInterval, self).__init__()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be integer and be >= {} and <= {}.".format(
            value, self._lower_bound, self._upper_bound)
        condition = value % 1 == 0
        condition = condition & (value >= self._lower_bound) & (
            value <= self._upper_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class IntegerOpenInterval(Constraint):
    """
    Constrain to an integer interval `(lower_bound, upper_bound)`
    """

    def __init__(self, lower_bound, upper_bound):
        super(IntegerOpenInterval, self).__init__()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be integer and be > {} and < {}.".format(
            value, self._lower_bound, self._upper_bound)
        condition = value % 1 == 0
        condition = condition & (value > self._lower_bound) & (
            value < self._upper_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class IntegerHalfOpenInterval(Constraint):
    """
    Constrain to an integer interval `[lower_bound, upper_bound)`
    """

    def __init__(self, lower_bound, upper_bound):
        super(IntegerHalfOpenInterval, self).__init__()
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be integer and be >= {} and < {}.".format(
            value, self._lower_bound, self._upper_bound)
        condition = value % 1 == 0
        condition = condition & (value >= self._lower_bound) & (
            value < self._upper_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class GreaterThan(Constraint):
    """
    Constrain to be greater than `lower_bound`.
    """

    def __init__(self, lower_bound):
        super(GreaterThan, self).__init__()
        self._lower_bound = lower_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be greater than {}".format(
            value, self._lower_bound)
        condition = value > self._lower_bound
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class UnitInterval(Interval):
    """
    Constrain to an unit interval `[0, 1]`
    """

    def __init__(self):
        super(UnitInterval, self).__init__(0, 1)


class GreaterThanEq(Constraint):
    """
    Constrain to be greater than or equal to `lower_bound`.
    """

    def __init__(self, lower_bound):
        super(GreaterThanEq, self).__init__()
        self._lower_bound = lower_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be greater than or equal to {}".format(
            value, self._lower_bound)
        condition = value >= self._lower_bound
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class LessThan(Constraint):
    """
    Constrain to be less than `upper_bound`.
    """

    def __init__(self, upper_bound):
        super(LessThan, self).__init__()
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be less than {}".format(
            value, self._upper_bound)
        condition = value < self._upper_bound
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class LessThanEq(Constraint):
    """
    Constrain to be less than `upper_bound`.
    """

    def __init__(self, upper_bound):
        super(LessThanEq, self).__init__()
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be less than or equal to {}".format(
            value, self._upper_bound)
        condition = value <= self._upper_bound
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class IntegerGreaterThan(Constraint):
    """
    Constrain to be integer and be greater than `lower_bound`.
    """

    def __init__(self, lower_bound):
        super(IntegerGreaterThan, self).__init__()
        self._lower_bound = lower_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be integer and be greater than {}".format(
            value, self._lower_bound)
        condition = value % 1 == 0
        condition = F.np.bitwise_and(condition, value > self._lower_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class IntegerGreaterThanEq(Constraint):
    """
    Constrain to be integer and be greater than or equal to `lower_bound`.
    """

    def __init__(self, lower_bound):
        super(IntegerGreaterThanEq, self).__init__()
        self._lower_bound = lower_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be integer and" \
                  " be greater than or equal to {}".format(
                      value, self._lower_bound)
        condition = value % 1 == 0
        condition = F.np.bitwise_and(condition, value >= self._lower_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class IntegerLessThan(Constraint):
    """
    Constrain to be integer and be less than `upper_bound`.
    """

    def __init__(self, upper_bound):
        super(IntegerLessThan, self).__init__()
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be integer and be less than {}".format(
            value, self._upper_bound)
        condition = value % 1 == 0
        condition = condition & (value < self._upper_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class IntegerLessThanEq(Constraint):
    """
    Constrain to be integer and be less than or equal to `upper_bound`.
    """

    def __init__(self, upper_bound):
        super(IntegerLessThanEq, self).__init__()
        self._upper_bound = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be integer and" \
                  " be less than or equal to {}".format(
                      value, self._upper_bound)
        condition = value % 1 == 0
        condition = condition & (value <= self._upper_bound)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class Positive(GreaterThan):
    """
    Constrain to be greater than zero.
    """

    def __init__(self):
        super(Positive, self).__init__(0)


class NonNegative(GreaterThanEq):
    """
    Constrain to be greater than or equal to zero.
    """

    def __init__(self):
        super(NonNegative, self).__init__(0)


class PositiveInteger(IntegerGreaterThan):
    """
    Constrain to be positive integer.
    """

    def __init__(self):
        super(PositiveInteger, self).__init__(0)


class NonNegativeInteger(IntegerGreaterThanEq):
    """
    Constrain to be non-negative integer.
    """

    def __init__(self):
        super(NonNegativeInteger, self).__init__(0)


class Simplex(Constraint):
    """
    Constraint to the simplex that rightmost dimension lies on a simplex.
    `x >= 0` and `x.sum(-1) == 1`.
    """

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be >= 0 and" \
                  " its rightmost dimension should sum up to 1".format(value)
        condition = F.np.all(value >= 0, axis=-1)
        condition = condition & (F.np.abs(value.sum(-1) - 1) < 1e-6)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class LowerTriangular(Constraint):
    """
    Constraint to square lower triangular matrices.
    """

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be" \
                  " square lower triangular matrices".format(value)
        condition = F.np.tril(value) == value
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class LowerCholesky(Constraint):
    """
    Constraint to square lower triangular matrices with real and positive diagonal entries.
    """

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be" \
                  " square lower triangular matrices" \
                  " with real and positive diagonal entries".format(value)
        condition = F.np.all(F.np.tril(value) == value, axis=-1)
        condition = condition & (F.np.diagonal(value, axis1=-2, axis2=-1) > 0)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class PositiveDefinite(Constraint):
    """
    Constraint to positive-definite matrices.
    """

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be" \
                  " positive definite matrices".format(value)
        eps = 1e-5
        condition = F.np.all(
            F.np.abs(value - F.np.swapaxes(value, -1, -2)) < eps, axis=-1)
        condition = condition & (F.np.linalg.eigvals(value) > 0)
        _value = constraint_check(F)(condition, err_msg) * value
        return _value


class Cat(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `constraint_seq` at the submatrices at `axis`, each of size `lengths[axis]`,
    in compatible with :func:`np.concatenate`.
    """

    def __init__(self, constraint_seq, axis=0, lengths=None):
        assert all(isinstance(c, Constraint) for c in constraint_seq)
        self._constraint_seq = list(constraint_seq)
        if lengths is None:
            lengths = [1] * len(self._constraint_seq)
        self._lengths = list(lengths)
        assert len(self._lengths) == len(self._constraint_seq),\
            "The number of lengths {} should be equal to number" \
            " of constraints {}".format(
                len(self._lengths), len(self._constraint_seq))
        self._axis = axis

    def check(self, value):
        F = getF(value)
        _values = []
        start = 0
        for length in self._lengths:
            v = F.np.take(value, indices=F.np.arange(
                start, start + length), axis=self._axis)
            _values.append(v)
            start = start + length
        _value = F.np.concatenate(_values, self._axis)
        return _value


class Stack(Constraint):
    """
    Constraint functor that applies a sequence of constraints
    `constraint_seq` at the submatrices at `axis`,
    in compatible with :func:`np.stack`.

    Stack is currently only supported in imperative mode.
    """

    def __init__(self, constraint_seq, axis=0):
        assert all(isinstance(c, Constraint) for c in constraint_seq)
        self._constraint_seq = list(constraint_seq)
        self._axis = axis

    def check(self, value):
        F = getF(value)
        assert F is nd, "mxnet.probability.distributions.constraint.Stack" \
                        " is only supported when hybridization is turned off"
        size = value.shape[self._axis]
        value_array = F.np.split(value, size, axis=self._axis)
        value_array = [constraint.check(F.np.squeeze(v)) for v, constraint
                       in zip(value_array, self._constraint_seq)]
        _value = F.np.stack(value_array, self._axis)
        return _value


dependent_property = _DependentProperty
