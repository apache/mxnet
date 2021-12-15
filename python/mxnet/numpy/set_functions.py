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

"""Standard Array API for creating and operating on sets."""

from collections import namedtuple

from ..ndarray import numpy as _mx_nd_np


__all__ = ['unique_all', 'unique_inverse', 'unique_values']


def unique_all(x):
    """
    Returns the unique elements of an input array `x`

    Notes
    -----
    `unique_all` is a standard API in
    https://data-apis.org/array-api/latest/API_specification/set_functions.html#unique-all-x
    instead of an official NumPy operator.

    Parameters
    ----------
    x : ndarray
        Input array. This will be flattened if it is not already 1-D.

    Returns
    -------
    out : Tuple[ndarray, ndarray, ndarray, ndarray]
        a namedtuple (values, indices, inverse_indices, counts):
        values : ndarray
            The sorted unique values.
        indices : ndarray, optional
            The indices of the first occurrences of the unique values in the
            original array.
        inverse_indices : ndarray
            The indices to reconstruct the original array from the
            unique array.
        counts : ndarray
            The number of times each of the unique values comes up in the
            original array.
    """
    UniqueAll = namedtuple('UniqueAll', ['values', 'indices', 'inverse_indices', 'counts'])
    return UniqueAll(*_mx_nd_np.unique(x, True, True, True))


def unique_inverse(x):
    """
    Returns the unique elements of an input array `x` and the indices
    from the set of unique elements that reconstruct `x`.

    Notes
    -----
    `unique_inverse` is a standard API in
    https://data-apis.org/array-api/latest/API_specification/set_functions.html#unique-inverse-x
    instead of an official NumPy operator.

    Parameters
    ----------
    x : ndarray
        Input array. This will be flattened if it is not already 1-D.

    Returns
    -------
    out : Tuple[ndarray, ndarray]
        a namedtuple (values, inverse_indices):
        values : ndarray
            The sorted unique values.
        inverse_indices : ndarray
            The indices to reconstruct the original array from the
            unique array.
    """
    UniqueInverse = namedtuple('UniqueInverse', ['values', 'inverse_indices'])
    return UniqueInverse(*_mx_nd_np.unique(x, False, True, False))


def unique_values(x):
    """
    Returns the unique elements of an input array `x`.

    Notes
    -----
    `unique_values` is a standard API in
    https://data-apis.org/array-api/latest/API_specification/set_functions.html#unique-values-x
    instead of an official NumPy operator.

    Parameters
    ----------
    x : ndarray
        Input array. This will be flattened if it is not already 1-D.

    Returns
    -------
    out : ndarray
        The sorted unique values.
    """
    return _mx_nd_np.unique(x, False, False, False)
