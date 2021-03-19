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

"""Namespace for the operators not belonging to the official numpy package
used in Gluon dispatched by F=ndarray module."""

from . import _api_internal
from ...util import set_module


__all__ = ['reshape']

@set_module('mxnet.ndarray.numpy_extension')
def reshape(a, newshape, reverse=False, order='C'):
    """
    Gives a new shape to an array without changing its data.
    This function always returns a copy of the input array if
    ``out`` is not provided.

    Parameters
    ----------
    a : ndarray
        Array to be reshaped.

    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.

    reverse : boolean, optional, default=0
        If true then the special values are inferred from right to left

    order : {'C'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. Other order types such as 'F'/'A'
        may be added in the future.

    Returns
    -------
    reshaped_array : ndarray
        It will be always a copy of the original array. This behavior is different
        from the official NumPy ``reshape`` operator where views of the original array may be
        generated.

    See Also
    --------
    ndarray.reshape : Equivalent method.

    Examples
    --------
    >>> a = np.arange(6).reshape((3, 2))
    >>> a
    array([[0., 1.],
           [2., 3.],
           [4., 5.]])

    >>> np.reshape(a, (2, 3)) # C-like index ordering
    array([[0., 1., 2.],
           [3., 4., 5.]])

    >>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
    array([[0., 1., 2.],
           [3., 4., 5.]])

    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> np.reshape(a, 6)
    array([1., 2., 3., 4., 5., 6.])

    >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
    array([[1., 2.],
           [3., 4.],
           [5., 6.]])
    """
    return _api_internal.reshape(a, newshape, reverse, order)
