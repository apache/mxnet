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
"""Classes for registering and storing bijection/transformations from
unconstrained space to a given domain.
"""

from numbers import Number
from .transformation import (
    ExpTransform, AffineTransform, SigmoidTransform, ComposeTransform)
from ..distributions.constraint import (Constraint, Positive, GreaterThan, GreaterThanEq,
                                        LessThan, Interval, HalfOpenInterval)


__all__ = ['domain_map', 'biject_to', 'transform_to']


class domain_map():
    """
    Abstract Class for registering and storing mappings from domain
    to bijections/transformations
    """
    def __init__(self):
        # constraint -> constraint -> transformation
        self._storage = {}
        super(domain_map, self).__init__()

    def register(self, constraint, factory=None):
        """Register a bijection/transformation from unconstrained space to the domain
        specified by `constraint`.

        Parameters
        ----------
        constraint : Type or Object
            A class of constraint or an object of constraint
        factory : callable
            A function that outputs a `transformation` given a `constraint`,
            by default None.
        """
        # Decorator mode
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        if isinstance(constraint, Constraint):
            constraint = type(constraint)

        if not isinstance(constraint, type) or not issubclass(constraint, Constraint):
            raise TypeError('Expected constraint to be either a Constraint subclass or instance, '
                            'but got {}'.format(constraint))

        self._storage[constraint] = factory
        return factory

    def __call__(self, constraint):
        try:
            factory = self._storage[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                'Cannot transform {} constraints'.format(type(constraint).__name__))
        return factory(constraint)


biject_to = domain_map()
transform_to = domain_map()


@biject_to.register(Positive)
@transform_to.register(Positive)
def _transform_to_positive(constraint):
    # Although `constraint` is not used in this factory function,
    # we decide to keep it for the purpose of consistency.
    # pylint: disable=unused-argument
    return ExpTransform()


@biject_to.register(GreaterThan)
@biject_to.register(GreaterThanEq)
@transform_to.register(GreaterThan)
@transform_to.register(GreaterThanEq)
def _transform_to_greater_than(constraint):
    return ComposeTransform([ExpTransform(),
                             AffineTransform(constraint._lower_bound, 1)])


@biject_to.register(LessThan)
@transform_to.register(LessThan)
def _transform_to_less_than(constraint):
    return ComposeTransform([ExpTransform(),
                             AffineTransform(constraint._upper_bound, -1)])


@biject_to.register(Interval)
@biject_to.register(HalfOpenInterval)
@transform_to.register(Interval)
@transform_to.register(HalfOpenInterval)
def _transform_to_interval(constraint):
    # Handle the special case of the unit interval.
    lower_is_0 = isinstance(constraint._lower_bound,
                            Number) and constraint._lower_bound == 0
    upper_is_1 = isinstance(constraint._upper_bound,
                            Number) and constraint._upper_bound == 1
    if lower_is_0 and upper_is_1:
        return SigmoidTransform()

    loc = constraint._lower_bound
    scale = constraint._upper_bound - constraint._lower_bound
    return ComposeTransform([SigmoidTransform(),
                             AffineTransform(loc, scale)])
