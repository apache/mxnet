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

# pylint: skip-file

"""Doc placeholder for numpy ops with prefix _np."""


def _np_sometrue(a, axis=None, keepdims=False, out=None):
    """
    Check whether some values are true.

    Refer to `any` for full documentation.

    See Also
    --------
    any : equivalent function; see for details.
    """
    pass


def _npx_nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a ndarray with ndim is 2. Each row contains the indices
    of the non-zero elements. The values in `a` are always tested and returned in
    row-major, C-style order.

    The result of this is always a 2-D array, with a row for
    each non-zero element.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    array : ndarray
        Indices of elements that are non-zero.

    Notes
    -----
    This function differs from the original numpy.nonzero in the following aspects:
        - Does not support python numeric.
        - The return value is same as numpy.transpose(numpy.nonzero(a)).

    Examples
    --------
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> x
    array([[3, 0, 0],
           [0, 4, 0],
           [5, 6, 0]])
    >>> npx.nonzero(x)
    array([[0, 0],
           [1, 1],
           [2, 0],
           [2, 1]], dtype=int64)

    >>> np.transpose(npx.nonzero(x))
    array([[0, 1, 2, 2],
           [0, 1, 0, 1]], dtype=int64)
    """
    pass


def _np_repeat(a, repeats, axis=None):
    """
    Repeat elements of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    repeats : int
        The number of repetitions for each element.
    axis : int, optional
        The axis along which to repeat values.  By default, use the
        flattened input array, and return a flat output array.

    Returns
    -------
    repeated_array : ndarray
        Output array which has the same shape as `a`, except along
        the given axis.

    Notes
    -----
    Unlike the official NumPy ``repeat`` operator, this operator currently
    does not support array of ints for the parameter `repeats`.

    Examples
    --------
    >>> x = np.arange(4).reshape(2, 2)
    >>> x
    array([[0., 1.],
           [2., 3.]])
    >>> np.repeat(x, repeats=3)
    array([0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3.])
    >>> np.repeat(x, repeats=3, axis=0)
    array([[0., 1.],
           [0., 1.],
           [0., 1.],
           [2., 3.],
           [2., 3.],
           [2., 3.]])
    >>> np.repeat(x, repeats=3, axis=1)
    array([[0., 0., 0., 1., 1., 1.],
           [2., 2., 2., 3., 3., 3.]])
    """
    pass


def _np_dot(a, b, out=None):
    """
    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,

    - If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply`
      and using ``np.multiply(a, b)`` or ``a * b`` is preferred.

    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.

    - If `a` is an N-D array and `b` is a 2-D array, it is a
      sum product over the last axis of `a` and the second-to-last axis of `b`::

        dot(a, b)[i,j,k] = sum(a[i,j,:] * b[:,k])

    Parameters
    ----------
    a : ndarray
        First argument.
    b : ndarray
        Second argument.

    out : ndarray, optional
        Output argument. It must have the same shape and type as the expected output.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        If `out` is given, then it is returned

    Examples
    --------
    >>> a = np.array(3)
    >>> b = np.array(4)
    >>> np.dot(a, b)
    array(12.)

    For 2-D arrays it is the matrix product:

    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.dot(a, b)
    array([[4., 1.],
           [2., 2.]])

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(5*6)[::-1].reshape((6,5))
    >>> np.dot(a, b)[2,3,2,2]
    array(29884.)
    >>> np.sum(a[2,3,2,:] * b[:,2])
    array(29884.)
    """
    pass


def _np_copy(a, out=None):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : ndarray
        Input data.
    out : ndarray or None, optional
        Alternative output array in which to place the result. It must have
        the same shape and dtype as the expected output.

    Returns
    -------
    arr : ndarray
        Array interpretation of `a`.

    Notes
    -------
    This function differs from the original `numpy.copy
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.copy.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - Does not support "order" parameter.

    Examples
    --------
    Create an array x, with a reference y and a copy z:

    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)

    Note that, when ``x`` is modified, ``y`` is also modified, but not ``z``:

    >>> x[0] = 10
    >>> x[0] == y[0]
    array([1.])
    >>> x[0] == z[0]
    array([0.])
    """
    pass


def _np_reshape(a, newshape, order='C', out=None):
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


def _np_squeeze(a, axis=None, out=None):
    """
    Remove single-dimensional entries from the shape of an array.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Selects a subset of the single-dimensional entries in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.
    out : ndarray, optional
        Array into which the output is placed. It must have the same size
        and dtype as the input array.

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. It always returns a copy of `a`.

    Raises
    ------
    MXNetError
        If `axis` is not `None`, and an axis being squeezed is not of length 1

    See Also
    --------
    expand_dims : The inverse operation, adding singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones

    Examples
    --------
    >>> x = np.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> np.squeeze(x).shape
    (3,)
    >>> np.squeeze(x, axis=0).shape
    (3, 1)
    >>> np.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    mxnet.base.MXNetError: cannot select an axis to squeeze out which has size=3 not equal to one
    >>> np.squeeze(x, axis=2).shape
    (1, 3)
    """
    pass


def _np_max(a, axis=None, keepdims=False, out=None):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : int, optional
        Axis along which to operate.  By default, flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    max : ndarray
        Maximum of `a`. If `axis` is None, the result is an array of dimension 1.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    min :
        The minimum value of an array along a given axis, ignoring any nan.
    maximum :
        Element-wise maximum of two arrays, ignoring any nan.
    argmax :
        Return the indices of the maximum values.

    Notes
    -----
    NaN in the orginal `numpy` is denoted as nan and will be ignored.

    Don't use `max` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    ``max(a, axis=0)``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0., 1.],
        [2., 3.]])
    >>> np.max(a)            # Maximum of the flattened array
    array(3.)
    >>> np.max(a, axis=0)    # Maxima along the first axis
    array([2., 3.])
    >>> np.max(a, axis=1)    # Maxima along the second axis
    array([1., 3.])

    >>> b = np.arange(5, dtype=np.float32)
    >>> b[2] = np.nan
    >>> np.max(b)
    array(4.)
    """
    pass


def _np_amax(a, axis=None, keepdims=False, out=None):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : int, optional
        Axis along which to operate.  By default, flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    amax : ndarray
        Maximum of `a`. If `axis` is None, the result is an array of dimension 1.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    min :
        The minimum value of an array along a given axis, ignoring any nan.
    maximum :
        Element-wise maximum of two arrays, ignoring any nan.
    argmax :
        Return the indices of the maximum values.

    Notes
    -----
    NaN in the orginal `numpy` is denoted as nan and will be ignored.

    Don't use `amax` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    ``amax(a, axis=0)``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0., 1.],
        [2., 3.]])
    >>> np.amax(a)            # Maximum of the flattened array
    array(3.)
    >>> np.amax(a, axis=0)    # Maxima along the first axis
    array([2., 3.])
    >>> np.amax(a, axis=1)    # Maxima along the second axis
    array([1., 3.])

    >>> b = np.arange(5, dtype=np.float32)
    >>> b[2] = np.nan
    >>> np.amax(b)
    array(4.)
    """
    pass


def _np_min(a, axis=None, keepdims=False, out=None):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : int, optional
        Axis along which to operate.  By default, flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    min : ndarray
        Minimum of `a`. If `axis` is None, the result is an array of dimension 1.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    max :
        The maximum value of an array along a given axis, ignoring any nan.
    minimum :
        Element-wise minimum of two arrays, ignoring any nan.

    Notes
    -----
    NaN in the orginal `numpy` is denoted as nan and will be ignored.

    Don't use `min` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
    ``min(a, axis=0)``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0., 1.],
        [2., 3.]])
    >>> np.min(a)           # Minimum of the flattened array
    array(0.)
    >>> np.min(a, axis=0)   # Minima along the first axis
    array([0., 1.])
    >>> np.min(a, axis=1)   # Minima along the second axis
    array([0., 2.])
    >>> b = np.arange(5, dtype=np.float32)
    >>> b[2] = np.nan
    >>> np.min(b)
    array(0.) # nan will be ignored
    """
    pass


def _np_amin(a, axis=None, keepdims=False, out=None):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : int, optional
        Axis along which to operate.  By default, flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    amin : ndarray
        Minimum of `a`. If `axis` is None, the result is an array of dimension 1.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    max :
        The maximum value of an array along a given axis, ignoring any nan.
    minimum :
        Element-wise minimum of two arrays, ignoring any nan.

    Notes
    -----
    NaN in the orginal `numpy` is denoted as nan and will be ignored.

    Don't use `amin` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
    ``amin(a, axis=0)``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0., 1.],
        [2., 3.]])
    >>> np.amin(a)           # Minimum of the flattened array
    array(0.)
    >>> np.amin(a, axis=0)   # Minima along the first axis
    array([0., 1.])
    >>> np.amin(a, axis=1)   # Minima along the second axis
    array([0., 2.])
    >>> b = np.arange(5, dtype=np.float32)
    >>> b[2] = np.nan
    >>> np.amin(b)
    array(0.) # nan will be ignored
    """
    pass


def _np_prod(a, axis=None, dtype=None, out=None, keepdims=False):
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed.
        The default (`axis` = `None`) is perform a product over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a product is performed on multiple
        axes, instead of a single axis or all the axes as before.
    dtype : data-type, optional
        The data-type of the returned array, as well as of the accumulator
        in which the elements are multiplied.  By default, if `a` is of
        integer type, `dtype` is the default platform integer. (Note: if
        the type of `a` is unsigned, then so is `dtype`.)  Otherwise,
        the dtype is the same as that of `a`.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the
        output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    product_along_axis : ndarray, see `dtype` parameter above.
        An array shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    See Also
    --------
    ndarray.prod : equivalent method

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.  That means that, on a 32-bit platform:

    >>> x = np.array([536870910, 536870910, 536870910, 536870910])
    >>> np.prod(x) #random
    array(8.307675e+34)

    Examples
    --------
    By default, calculate the product of all elements:

    >>> np.prod(np.array([1.,2.]))
    array(2.)

    Even when the input array is two-dimensional:

    >>> np.prod(np.array([1.,2.,3.,4.]).reshape((2,2)))
    array(24.)

    But we can also specify the axis over which to multiply:

    >>> np.prod(np.array([1.,2.,3.,4.]).reshape((2,2)), axis=1)
    array([  2.,  12.])

    If the type of `x` is unsigned, then the output type is
    the unsigned platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.uint8)
    >>> np.prod(x).dtype == np.uint8
    True

    If `x` is of a signed integer type, then the output type
    is the default platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.int8)
    >>> np.prod(x).dtype == np.int8
    True
    """
    pass


def _np_product(a, axis=None, dtype=None, out=None, keepdims=False):
    """
    Return the product of array elements over a given axis.

    See Also
    --------
    prod : equivalent function; see for details.
    """
    pass


def _np_moveaxis(a, source, destination):
    """Move axes of an array to new positions.
    Other axes remain in their original order.

    Parameters
    ----------
    a : ndarray
        The array whose axes should be reordered.
        source : int or sequence of int
        Original positions of the axes to move. These must be unique.
        destination : int or sequence of int
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
        transpose: Permute the dimensions of an array.
        swapaxes: Interchange two axes of an array.

    Examples
    --------
    >>> x = np.zeros((3, 4, 5))
    >>> np.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> np.moveaxis(x, -1, 0).shape
    (5, 3, 4)
    These all achieve the same result:
    >>> np.transpose(x).shape
    (5, 4, 3)
    >>> np.swapaxes(x, 0, -1).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
    (5, 4, 3)
    """
    pass

def _np__random_shuffle(x):
    """
    Modify a sequence in-place by shuffling its contents.

    This function only shuffles the array along the first axis of a
    multi-dimensional array. The order of sub-arrays is changed but
    their contents remain the same.

    Parameters
    ----------
    x: ndarray
        The array or list to be shuffled.

    Returns
    -------
    None

    Examples
    --------
    >>> arr = np.arange(10)
    >>> np.random.shuffle(arr)
    >>> arr
    array([5., 1., 0., 6., 7., 3., 9., 8., 4., 2.])  # random

    Multi-dimensional arrays are only shuffled along the first axis:

    >>> arr = np.arange(9).reshape((3, 3))
    >>> np.random.shuffle(arr)
    >>> arr
    array([[6., 7., 8.], # random
           [3., 4., 5.],
           [0., 1., 2.]])
    """
    pass


def _npx_constraint_check(x, msg):
    """
    This operator will check if all the elements in a boolean tensor is true.
    If not, ValueError exception will be raised in the backend with given error message.
    In order to evaluate this operator, one should multiply the origin tensor by the return value
    of this operator to force this operator become part of the computation graph,
    otherwise the check would not be working under symoblic mode.

    Parameters
    ----------
    x : ndarray
        A boolean tensor.
    msg : string
        The error message in the exception.

    Returns
    -------
    out : ndarray
        If all the elements in the input tensor are true,
        array(True) will be returned, otherwise ValueError exception would
        be raised before anything got returned.

    Examples
    --------
    >>> loc = np.zeros((2,2))
    >>> scale = np.array(#some_value)
    >>> constraint = (scale > 0)
    >>> np.random.normal(loc,
                     scale * npx.constraint_check(constraint, 'Scale should be larger than zero'))

    If elements in the scale tensor are all bigger than zero, npx.constraint_check would return
    `np.array(True)`, which will not change the value of `scale` when multiplied by.
    If some of the elements in the scale tensor violate the constraint,
    i.e. there exists `False` in the boolean tensor `constraint`,
    a `ValueError` exception with given message 'Scale should be larger than zero' would be raised.
    """
    pass


def _npx_reshape(a, newshape, reverse=False, order='C'):
    """
    Gives a new shape to an array without changing its data.
    This function always returns a copy of the input array if
    ``out`` is not provided.

    Parameters
    ----------
    a : ndarray
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape.
        If an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is inferred
        from the length of the array and remaining dimensions.
        -2 to -6 are used for data manipulation.

        - -2 copy this dimension from the input to the output shape.
        - -3 will skip current dimension if and only if the current dim size is one.
        - -4 copy all remain of the input dimensions to the output shape.
        - -5 use the product of two consecutive dimensions of the input
          shape as the output.
        - -6 split one dimension of the input into two dimensions passed
          subsequent to -6 in the new shape.

    reverse : bool, optional
        If set to true, the special values will be inferred from right to left.
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

    Examples
    --------
    >>> x = np.ones((2, 3, 8))
    >>> npx.reshape(x, (-2, -2, 2, -1)).shape
    (2, 3, 2, 4)
    >>> x = np.ones((8, 3, 3, 3, 4, 4))
    >>> npx.reshape(x, (-6, 2, -1, -4)).shape
    (2, 4, 3, 3, 3, 4, 4)
    >>> x = np.ones((8, 3, 3, 3, 4, 4))
    >>> npx.reshape(x, (-5, -4)).shape
    (24, 3, 3, 4, 4)
    >>> x = np.ones((8, 1, 1, 1, 3))
    >>> npx.reshape(x, (-2, -3, -3, -3, -2)).shape
    (8, 3)
    >>> x = np.ones((8, 3, 3, 3, 3, 8))
    >>> npx.reshape(x, (-4, -5), reverse=True).shape
    (8, 3, 3, 3, 24)
    >>> x = np.ones((8, 3, 2, 4, 8))
    >>> npx.reshape(x, (-4, -1, 2, -6), reverse=True).shape
    (8, 3, 2, 4, 4, 2)
    """
    pass


def _np_diag(array, k=0):
    """
    Extracts a diagonal or constructs a diagonal array.
    - 1-D arrays: constructs a 2-D array with the input as its diagonal, all other elements are zero.
    - 2-D arrays: extracts the k-th Diagonal

    Parameters
    ----------
    array : ndarray
        The array to apply diag method.
    k : offset
        extracts or constructs kth diagonal given input array

    Examples
    --------
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.diag(x)
    array([0, 4, 8])
    >>> np.diag(x, k=1)
    array([1, 5])
    >>> np.diag(x, k=-1)
    array([3, 7])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
    """
    pass


def _np_diagonal(a, offset=0, axis1=0, axis2=1):
    """
    If a is 2-D, returns the diagonal of a with the given offset, i.e., the collection of elements of
    the form a[i, i+offset]. If a has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-D sub-array whose diagonal is returned. The shape of the
    resulting array can be determined by removing axis1 and axis2 and appending an index to the
    right equal to the size of the resulting diagonals.

    Parameters
    ----------
    a : Symbol
        Input data from which diagonal are taken.
    offset: int, Optional
        Offset of the diagonal from the main diagonal
    axis1: int, Optional
        Axis to be used as the first axis of the 2-D sub-arrays
    axis2: int, Optional
        Axis to be used as the second axis of the 2-D sub-arrays

    Returns
    -------
    out : Symbol
        Output result

    Raises
    -------
    ValueError:  If the dimension of a is less than 2.

    Examples
    --------
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
        [2, 3]])
    >>> np.diagonal(a)
    array([0, 3])
    >>> np.diagonal(a, 1)
    array([1])

    >>> a = np.arange(8).reshape(2,2,2)
    >>>a
    array([[[0, 1],
            [2, 3]],
            [[4, 5],
            [6, 7]]])
    >>> np.diagonal(a, 0, 0, 1)
    array([[0, 6],
            [1, 7]])
    """
    pass


def _np_diagflat(array, k=0):
    """
    Create a two-dimensional array with the flattened input as a diagonal.
    Parameters
    ----------
    arr : ndarray
        Input data, which is flattened and set as the `k`-th
        diagonal of the output.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal,
        a positive (negative) `k` giving the number of the diagonal above
        (below) the main.
    Returns
    -------
    out : ndarray
        The 2-D output array.
    See Also
    --------
    diag : MATLAB work-alike for 1-D and 2-D arrays.
    diagonal : Return specified diagonals.
    trace : Sum along diagonals.
    Examples
    --------
    >>> np.diagflat([[1,2], [3,4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])
    >>> np.diagflat([1,2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])
    """
    pass
