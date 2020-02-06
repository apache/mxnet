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
__all__ = ["Constraint", "Real", "Boolean", "Interval", "GreaterThan", "Positive"]

from .utils import getF


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


class Real(Constraint):
    """
    Constrain to be a real number. (exclude `np.nan`)
    """
    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be a real tensor".format(value)
        condition = (value == value)
        _value = F.npx.constraint_check(condition, err_msg) * value
        return _value


class Boolean(Constraint):
    """
    Constrain to `{0, 1}`.
    """
    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be either 0 or 1.".format(value)
        # FIXME: replace bitwise_or with logical_or instead
        condition = F.np.bitwise_or(value == 0, value == 1)
        _value = F.npx.constraint_check(condition, err_msg) * value
        return _value


class Interval(Constraint):
    """
    Constrain to a real interval `[lower_bound, upper_bound]`
    """
    def __init__(self, lower_bound, upper_bound):
        super(Interval, self).__init__()
        self._low = lower_bound
        self._up = upper_bound

    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be between {} and {}.".format(
                    value, self._low, self._up)
        # FIXME: replace bitwise_and with logical_and
        condition = F.np.bitwise_and(value > self._low, value < self._up)
        _value = F.npx.constraint_check(condition, err_msg) * value
        return _value


class GreaterThan(Constraint):
    """
    Constrain to be greater than `lower_bound`.
    """
    def __init__(self, lower_bound):
        super(GreaterThan, self).__init__()
        self._low = lower_bound
    
    def check(self, value):
        F = getF(value)
        err_msg = "Constraint violated: {} should be greater than {}".format(
                    value, self._low)
        condition = value > self._low
        _value = F.npx.constraint_check(condition, err_msg) * value
        return _value


class Positive(GreaterThan):
    """
    Constrain to be greater than zero.
    """
    def __init__(self):
        super(Positive, self).__init__(0)
