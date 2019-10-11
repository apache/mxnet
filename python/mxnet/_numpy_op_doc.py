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


def _np_ones_like(a):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.
    """
    pass


def _np_zeros_like(a):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.
    """
    pass


def _np_cumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See `doc.ufuncs`
        (Section "Output arguments") for more details.

    Returns
    -------
    cumsum_along_axis : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned. The
        result has the same size as `a`, and the same shape as `a` if
        `axis` is not None or `a` is a 1-d array.

    Examples
    --------
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.cumsum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
    array([  1.,   3.,   6.,  10.,  15.,  21.])
    >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])
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
    This function differs from the original numpy.prod in the following aspects:
        - Do not support python numeric.
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


def _np_transpose(a, axes=None):
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    axes : list of ints, optional
        By default, reverse the dimensions,
        otherwise permute the axes according to the values given.

    Returns
    -------
    p : ndarray
        a with its axes permuted.

    Notes
    -----
    This function differs from the original `numpy.transpose
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html>`_ in
    the following way(s):

    - only ndarray is accepted as valid input, python iterables are not supported
    - the operator always returns an `ndarray` that does not share the memory with the input

    Examples
    --------
    >>> x = np.arange(4).reshape((2,2))
    >>> x
    array([[0., 1.],
           [2., 3.]])
    >>> np.transpose(x)
    array([[0., 2.],
           [1., 3.]])
    >>> x = np.ones((1, 2, 3))
    >>> np.transpose(x, (1, 0, 2)).shape
    (2, 1, 3)
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


def _np_sum(a, axis=None, dtype=None, keepdims=False, initial=None, out=None):
    r"""
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed. The default type is float32.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-classes `sum` method does not implement `keepdims` any
        exceptions will be raised.
    initial: Currently only supports None as input, optional
        Starting value for the sum.
        Currently not implemented. Please use ``None`` as input or skip this argument.
    out : ndarray or None, optional
        Alternative output array in which to place the result. It must have
        the same shape and dtype as the expected output.

    Returns
    -------
    sum_along_axis : ndarray
        An ndarray with the same shape as `a`, with the specified
        axis removed. If an output array is specified, a reference to
        `out` is returned.

    Notes
    -----
    - Input type does not support Python native iterables.
    - "out" param: cannot perform auto type change. out ndarray's dtype must be the same as the expected output.
    - "initial" param is not supported yet. Please use None as input.
    - Arithmetic is modular when using integer types, and no error is raised on overflow.
    - The sum of an empty array is the neutral element 0:

    >>> a = np.empty(1)
    >>> np.sum(a)
    array(0.)

    This function differs from the original `numpy.sum
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...).
    - "out" param: cannot perform auto type cast. out ndarray's dtype must be the same as the expected output.
    - "initial" param is not supported yet. Please use ``None`` as input or skip it.

    Examples
    --------
    >>> a = np.array([0.5, 1.5])
    >>> np.sum(a)
    array(2.)
    >>> a = np.array([0.5, 0.7, 0.2, 1.5])
    >>> np.sum(a, dtype=np.int32)
    array(2, dtype=int32)
    >>> a = np.array([[0, 1], [0, 5]])
    >>> np.sum(a)
    array(6.)
    >>> np.sum(a, axis=0)
    array([0., 6.])
    >>> np.sum(a, axis=1)
    array([1., 5.])

    With output ndarray:

    >>> a = np.array([[0, 1], [0, 5]])
    >>> b = np.ones((2,), dtype=np.float32)
    >>> np.sum(a, axis = 0, out=b)
    array([0., 6.])
    >>> b
    array([0., 6.])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    array(-128, dtype=int8)
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
    """


def _np__linalg_svd(a):
    r"""
    Singular Value Decomposition.

    When `a` is a 2D array, it is factorized as ``ut @ np.diag(s) @ v``,
    where `ut` and `v` are 2D orthonormal arrays and `s` is a 1D
    array of `a`'s singular values. When `a` is higher-dimensional, SVD is
    applied in stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) ndarray 
        A real or complex array with ``a.ndim >= 2`` and ``M <= N``.

    Returns
    -------
    ut: (..., M, M) ndarray
        Orthonormal array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    s : (..., M) ndarray
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    v : (..., M, N) ndarray
        Orthonormal array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.

    Notes
    -----

    The decomposition is performed using LAPACK routine ``_gesvd``.

    SVD is usually described for the factorization of a 2D matrix :math:`A`.
    The higher-dimensional case will be discussed below. In the 2D case, SVD is
    written as :math:`A = U^T S V`, where :math:`A = a`, :math:`U^T = ut`,
    :math:`S= \mathtt{np.diag}(s)` and :math:`V = v`. The 1D array `s`
    contains the singular values of `a` and `ut` and `v` are orthonormal. The rows
    of `v` are the eigenvectors of :math:`A^T A` and the columns of `ut` are
    the eigenvectors of :math:`A A^T`. In both cases the corresponding
    (possibly non-zero) eigenvalues are given by ``s**2``.

    If `a` has more than two dimensions, then broadcasting rules apply.
    This means that SVD is working in "stacked" mode: it iterates over 
    all indices of the first ``a.ndim - 2`` dimensions and for each
    combination SVD is applied to the last two indices. The matrix `a` 
    can be reconstructed from the decomposition with either 
    ``(ut * s[..., None, :]) @ v`` or
    ``ut @ (s[..., None] * v)``. (The ``@`` operator denotes batch matrix multiplication)

    Examples
    --------
    >>> a = np.arange(54).reshape(6, 9)
    >>> ut, s, v = np.linalg.svd(a)
    >>> ut.shape, s.shape, v.shape
    ((6, 6), (6,), (6, 9))
    >>> s = s.reshape(6, 1)
    >>> ret = np.dot(ut, s * v)
    >>> (ret - a > 1e-3).sum()
    array(0.)
    >>> (ret - a < -1e-3).sum()
    array(0.)
    """
    pass


def _np_roll(a, shift, axis=None):
    """
    Roll array elements along a given axis.
    
    Elements that roll beyond the last position are re-introduced at
    the first.

    Parameters
    ----------
    a : ndarray
        Input array.
    shift : int or tuple of ints
        The number of places by which elements are shifted.  If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number.  If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted.  By default, the
        array is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    Notes
    -----
    Supports rolling over multiple dimensions simultaneously.

    Examples
    --------
    >>> x = np.arange(10)
    >>> np.roll(x, 2)
    array([8., 9., 0., 1., 2., 3., 4., 5., 6., 7.])
    >>> np.roll(x, -2)
    array([2., 3., 4., 5., 6., 7., 8., 9., 0., 1.])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])
    >>> np.roll(x2, 1)
    array([[9., 0., 1., 2., 3.],
           [4., 5., 6., 7., 8.]])
    >>> np.roll(x2, -1)
    array([[1., 2., 3., 4., 5.],
           [6., 7., 8., 9., 0.]])
    >>> np.roll(x2, 1, axis=0)
    array([[5., 6., 7., 8., 9.],
           [0., 1., 2., 3., 4.]])
    >>> np.roll(x2, -1, axis=0)
    array([[5., 6., 7., 8., 9.],
           [0., 1., 2., 3., 4.]])
    >>> np.roll(x2, 1, axis=1)
    array([[4., 0., 1., 2., 3.],
           [9., 5., 6., 7., 8.]])
    >>> np.roll(x2, -1, axis=1)
    array([[1., 2., 3., 4., 0.],
           [6., 7., 8., 9., 5.]])
   """


def _np_trace(a, offset=0, axis1=0, axis2=1, out=None):
    """
    Return the sum along diagonals of the array.
    If `a` is 2-D, the sum along its diagonal with the given offset
    is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.
    If `a` has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-D sub-arrays whose traces are returned.
    The shape of the resulting array is the same as that of `a` with `axis1`
    and `axis2` removed.

    Parameters
    ----------
    a : ndarray
        Input array, from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both positive
        and negative. Defaults to 0.
    axis1, axis2 : int, optional
        Axes to be used as the first and second axis of the 2-D sub-arrays
        from which the diagonals should be taken. Defaults are the first two
        axes of `a`.
    out : ndarray, optional
        Array into which the output is placed. It must be of the right shape
        and right type to hold the output.

    Returns
    -------
    sum_along_diagonals : ndarray
        If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
        larger dimensions, then an array of sums along diagonals is returned.

    Examples
    --------
    >>> a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> np.trace(a)
    array(3.)
    >>> a = np.arange(8).reshape((2, 2, 2))
    >>> np.trace(a)
    array([6., 8.])
    >>> a = np.arange(24).reshape((2, 2, 2, 3))
    >>> np.trace(a).shape
    (2, 3)
    """
    pass
