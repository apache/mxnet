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

"""Namespace for ops used in imperative programming."""

from functools import reduce

from ..ndarray import numpy as _mx_nd_np
from ..util import wrap_data_api_linalg_func
from .fallback_linalg import *  # pylint: disable=wildcard-import,unused-wildcard-import
from . import fallback_linalg

__all__ = ['norm', 'svd', 'cholesky', 'qr', 'inv', 'det', 'slogdet', 'solve', 'tensorinv', 'tensorsolve',
           'pinv', 'eigvals', 'eig', 'eigvalsh', 'eigh', 'lstsq', 'matrix_rank', 'cross', 'diagonal', 'outer',
           'tensordot', 'trace', 'matrix_transpose', 'vecdot', 'svdvals', 'vector_norm', 'matrix_norm']

__all__ += fallback_linalg.__all__


@wrap_data_api_linalg_func
def matrix_rank(M, rtol=None, hermitian=False):
    r"""
    Return matrix rank of array using SVD method

    Rank of the array is the number of singular values of the array that are
    greater than `rtol`.

    Notes
    -----
    `rtol` param is requested in array-api-standard in
    https://data-apis.org/array-api/latest/extensions/generated/signatures.linalg.matrix_rank.html
    instead of a parameter in official NumPy operator.

    Parameters
    ----------
    M : {(M,), (..., M, N)} ndarray
        Input vector or stack of matrices.
    rtol : (...) ndarray, float, optional
        Threshold below which SVD values are considered zero. If `rtol` is
        None, and ``S`` is an array with singular values for `M`, and
        ``eps`` is the epsilon value for datatype of ``S``, then `rtol` is
        set to ``S.max() * max(M.shape) * eps``.
    hermitian : bool, optional
        If True, `M` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Default: False.

    Returns
    -------
    rank : (...) ndarray
        Rank of M.

    Examples
    --------
    >>> from mxnet import np
    >>> np.linalg.matrix_rank(np.eye(4)) # Full rank matrix
    4
    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    >>> np.linalg.matrix_rank(I)
    3
    >>> np.linalg.matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
    1
    >>> np.linalg.matrix_rank(np.zeros((4,)))
    0
    """
    return _mx_nd_np.linalg.matrix_rank(M, rtol, hermitian)


def matrix_transpose(a):
    r"""
    Transposes a matrix (or a stack of matrices) `a`.

    Notes
    -----
    `matrix_transpose` is new in array API spec:
    https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html#linalg-matrix-transpose-x
    instead of an official NumPy operator. Unlike transpose, it only transposes the last two axes.

    Parameters
    ----------
    a : ndarray
        Input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices.

    Returns
    ----------
    out : ndarray
        An array containing the transpose for each matrix and having shape (..., N, M).
        The returned array must have the same data type as `a`.

    Examples
    --------
    >>> x = np.arange(4).reshape((2,2))
    >>> x
    array([[0., 1.],
           [2., 3.]])
    >>> np.linalg.matrix_transpose(x)
    array([[0., 2.],
           [1., 3.]])
    >>> x = np.ones((1, 2, 3))
    >>> np.linalg.matrix_transpose(x)
    array([[[1., 1.],
            [1., 1.],
            [1., 1.]]])
    """
    if a.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return _mx_nd_np.swapaxes(a, -1, -2)


def trace(a, offset=0):
    r"""
    Returns a tensor contraction of `a` and `b` over specific axes.

    Notes
    -----
    `trace` is an alias for `trace`. It is a standard API in
    https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html#linalg-trace-x-offset-0
    instead of an official NumPy operator.

    Parameters
    ----------
    a : ndarray
        Input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices.
        Should have a numeric data type.
    offset : int
        Offset specifying the off-diagonal relative to the main diagonal.

        offset = 0 : the main diagonal.
        offset > 0 : off-diagonal above the main diagonal.
        offset < 0 : off-diagonal below the main diagonal.

        Default: 0.

    Returns
    ----------
    out : ndarray
        An array containing the traces and whose shape is determined by removing the last two dimensions and storing
        the traces in the last array dimension. For example, if `a` has rank `k` and shape `(I, J, K, ..., L, M, N)`,
        then an output array has rank `k-2` and shape `(I, J, K, ..., L)`
        where: `out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])`
        The returned array must have the same data type as `a`.

    Examples
    --------
    >>> x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> np.linalg.trace(x)
    array(3.)
    >>> x = np.arange(8).reshape((2, 2, 2))
    >>> np.linalg.trace(x)
    array([6., 8.])
    >>> x = np.arange(24).reshape((2, 2, 2, 3))
    >>> np.linalg.trace(x).shape
    (2, 3)
    >>> np.linalg.trace(x)
    array([[18., 20., 22.],
        [24., 26., 28.]])
    """
    # axis1, axis2: defaults are the first two axes of `a`.
    return _mx_nd_np.trace(a, offset=offset, axis1=0, axis2=1, out=None)


def tensordot(a, b, axes=2):
    r"""
    Returns a tensor contraction of `a` and `b` over specific axes.

    Notes
    -----
    `tensordot` is an alias for `tensordot`. It is a standard API in
    https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html#linalg-tensordot-x1-x2-axes-2
    instead of an official NumPy operator.

    Parameters
    ----------
    a : ndarray
        First input array. Should have a numeric data type.
    b : ndarray
        Second input array. Must be compatible with `a` (see Broadcasting). Should have a numeric data type.
    axes : int, tuple
        Number of axes to contract or explicit sequences of axes for `a` and `b`, respectively.
        If axes is an int equal to `N` , then contraction must be performed over the last `N` axes of `a`
        and the first `N` axes of `b` in order.
        The size of each corresponding axis (dimension) must match. Must be nonnegative.

        If N equals 0 , the result is the tensor (outer) product.
        If N equals 1 , the result is the tensor dot product.
        If N equals 2 , the result is the tensor double contraction (default).

        Default: 2.

    Returns
    ----------
    out : ndarray
        An array containing the tensor contraction whose shape consists of the non-contracted axes (dimensions) of the
        first array `a`, followed by the non-contracted axes (dimensions) of the second array `b`.

    Examples
    --------
    >>> x = np.arange(60.).reshape(3,4,5)
    >>> y = np.arange(24.).reshape(4,3,2)
    >>> z = np.linalg.tensordot(x, y, axes=([1,0],[0,1]))
    >>> z.shape
    (5, 2)
    >>> z
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    """
    return _mx_nd_np.tensordot(a, b, axes)


def diagonal(a, offset=0):
    r"""
    Returns the specified diagonals of a matrix (or a stack of matrices) `a`.

    Notes
    -----
    `diagonal` is an alias for `diagonal`. It is a standard API in
    https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html#linalg-diagonal-x-offset-0
    instead of an official NumPy operator.

    Parameters
    ----------
    a : ndarray
        The array to apply diag method.
    offset : int
        Extracts or constructs kth diagonal given input array.
        Offset specifying the off-diagonal relative to the main diagonal.

        offset = 0 : the main diagonal.
        offset > 0 : off-diagonal above the main diagonal.
        offset < 0 : off-diagonal below the main diagonal.

        Default: 0.

    Returns
    ----------
    out : ndarray
        An array containing the diagonals and whose shape is determined by removing the last two dimensions and
        appending a dimension equal to the size of the resulting diagonals.
        The returned array must have the same data type as a.

    Examples
    --------
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> np.linalg.diagonal(x)
    array([0., 4., 8.])
    >>> np.linalg.diagonal(x, offset=1)
    array([1., 5.])
    >>> np.linalg.diagonal(x, offset=-1)
    array([3., 7.])
    """
    return _mx_nd_np.diag(a, k=offset)


def cross(a, b, axis=-1):
    r"""
    Returns the cross product of 3-element vectors.

    If `a` and `b` are multi-dimensional arrays (i.e., both have a rank greater than 1),
    then the cross-product of each pair of corresponding 3-element vectors is independently computed.

    Notes
    -----
    `cross` is an alias for `cross`. It is a standard API in
    https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html#linalg-cross-x1-x2-axis-1
    instead of an official NumPy operator.

    Parameters
    ----------
    a : ndarray
        First input array. Should have a numeric data type.
    b : ndarray
        Second input array. Must have the same shape as a. Should have a numeric data type.
    axis : int
        If defined, the axis of `a` and `b` that defines the vector(s) and cross product(s).

        Default: -1.

    Returns
    -------
    out : (...) ndarray
        An array containing the cross products.

    Examples
    --------
    Vector cross-product.

    >>> x = np.array([1., 2., 3.])
    >>> y = np.array([4., 5., 6.])
    >>> np.linalg.cross(x, y)
    array([-3.,  6., -3.])

    One vector with dimension 2.

    >>> x = np.array([1., 2.])
    >>> y = np.array([4., 5., 6.])
    >>> np.linalg.cross(x, y)
    array([12., -6., -3.])

    Equivalently:

    >>> x = np.array([1., 2., 0.])
    >>> y = np.array([4., 5., 6.])
    >>>np.linalg.cross(x, y)
    array([12., -6., -3.])

    Both vectors with dimension 2.

    >>> x = np.array([1., 2.])
    >>> y = np.array([4., 5.])
    >>> np.linalg.cross(x, y)
    array(-3.)

    Multiple vector cross-products. Note that the direction of the cross
    product vector is defined by the `right-hand rule`.

    >>> x = np.array([[1., 2., 3.], [4., 5., 6.]])
    >>> y = np.array([[4., 5., 6.], [1., 2., 3.]])
    >>> np.linalg.cross(x, y)
    array([[-3.,  6., -3.],
           [ 3., -6.,  3.]])
    """
    # For a given API standard, the axis of axisa, axisb, axisc are equal to the axis
    return _mx_nd_np.cross(a, b, axisa=axis, axisb=axis, axisc=axis, axis=axis)


def outer(a, b):
    r"""
    Computes the outer product of two vectors `a` and `b`.

    Notes
    -----
    `outer` is an alias for `outer`. It is a standard API in
    https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html#linalg-outer-x1-x2
    instead of an official NumPy operator.

    Parameters
    ----------
    a : ndarray
        One-dimensional input array of size `N` . Should have a numeric data type.
    b : ndarray
        One-dimensional input array of size `M` . Should have a numeric data type.

    Returns
    -------
    out : ndarray
        A two-dimensional array containing the outer product and whose shape is `(N, M)`.
        The returned array must have a data type determined by Type Promotion Rules.

    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:

    >>> x = np.linalg.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> x
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    """
    return _mx_nd_np.tensordot(a.flatten(), b.flatten(), 0)


def vecdot(a, b, axis=None):
    r"""
    Return the dot product of two vectors.
    Note that `vecdot` handles multidimensional arrays differently than `dot`:
    it does *not* perform a matrix product, but flattens input arguments
    to 1-D vectors first. Consequently, it should only be used for vectors.

    Notes
    ----------
    `vecdot` is a alias for `vdot`. It is a standard API in
    https://data-apis.org/array-api/latest/API_specification/linear_algebra_functions.html#vecdot-x1-x2-axis-1
    instead of an official NumPy operator.

    Parameters
    ----------
    a : ndarray
        First argument to the dot product.
    b : ndarray
        Second argument to the dot product.
    axis : axis over which to compute the dot product. Must be an integer on
        the interval [-N, N) , where N is the rank (number of dimensions) of
        the shape determined according to Broadcasting . If specified as a
        negative integer, the function must determine the axis along which
        to compute the dot product by counting backward from the last dimension
        (where -1 refers to the last dimension). If None , the function must
        compute the dot product over the last axis. Default: None .

    Returns
    -------
    output : ndarray
        Dot product of `a` and `b`.

    See Also
    --------
    dot : Return the dot product without using the complex conjugate of the
        first argument.

    Examples
    --------
    Note that higher-dimensional arrays are flattened!

    >>> a = np.array([[1, 4], [5, 6]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.linalg.vecdot(a, b)
    array(30.)
    >>> np.linalg.vecdot(b, a)
    array(30.)
    >>> 1*4 + 4*1 + 5*2 + 6*2
    30
    """
    return _mx_nd_np.tensordot(a.flatten(), b.flatten(), axis)


def lstsq(a, b, rcond='warn'):
    r"""
    Return the least-squares solution to a linear matrix equation.

    Solves the equation :math:`a x = b` by computing a vector `x` that
    minimizes the squared Euclidean 2-norm :math:`\| b - a x \|^2_2`.
    The equation may be under-, well-, or over-determined (i.e., the
    number of linearly independent rows of `a` can be less than, equal
    to, or greater than its number of linearly independent columns).
    If `a` is square and of full rank, then `x` (but for round-off error)
    is the "exact" solution of the equation.

    Parameters
    ----------
    a : (M, N) ndarray
        "Coefficient" matrix.
    b : {(M,), (M, K)} ndarray
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    rcond : float, optional
        Cut-off ratio for small singular values of `a`.
        For the purposes of rank determination, singular values are treated
        as zero if they are smaller than `rcond` times the largest singular
        value of `a`
        The default of ``warn`` or ``-1`` will use the machine precision as
        `rcond` parameter. The default of ``None`` will use the machine
        precision times `max(M, N)`.

    Returns
    -------
    x : {(N,), (N, K)} ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.
    residuals : {(1,), (K,), (0,)} ndarray
        Sums of residuals.
        Squared Euclidean 2-norm for each column in ``b - a*x``.
        If the rank of `a` is < N or M <= N, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : int
        Rank of matrix `a`.
    s : (min(M, N),) ndarray
        Singular values of `a`.

    Raises
    ------
    MXNetError
        If computation does not converge.

    Notes
    -----
    If `b` is a matrix, then all array results are returned as matrices.

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])
    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])
    >>> m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    >>> m, c
    (1.0 -0.95) # may vary
    """
    return _mx_nd_np.linalg.lstsq(a, b, rcond)


@wrap_data_api_linalg_func
def pinv(a, rtol=None, hermitian=False):
    r"""
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    *large* singular values.

    Notes
    -----
    `rtol` param is requested in array-api-standard in
    https://data-apis.org/array-api/latest/extensions/generated/signatures.linalg.pinv.html
    instead of a parameter in official NumPy operator.

    Parameters
    ----------
    a : (..., M, N) ndarray
        Matrix or stack of matrices to be pseudo-inverted.
    rtol : (...) {float or ndarray of float}, optional
        Cutoff for small singular values.
        Singular values less than or equal to
        ``rtol * largest_singular_value`` are set to zero.
        Broadcasts against the stack of matrices.
    hermitian : bool, optional
        If True, `a` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.

    Returns
    -------
    B : (..., N, M) ndarray
        The pseudo-inverse of `a`. If `a` is a `matrix` instance, then so
        is `B`.

    Raises
    ------
    MXNetError
        If the SVD computation does not converge.

    Notes
    -----
    The pseudo-inverse of a matrix A, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    It can be shown that if :math:`Q_1 \\Sigma Q_2^T = A` is the singular
    value decomposition of A, then
    :math:`A^+ = Q_2 \\Sigma^+ Q_1^T`, where :math:`Q_{1,2}` are
    orthogonal matrices, :math:`\\Sigma` is a diagonal matrix consisting
    of A's so-called singular values, (followed, typically, by
    zeros), and then :math:`\\Sigma^+` is simply the diagonal matrix
    consisting of the reciprocals of A's singular values
    (again, followed by zeros). [1]_

    References
    ----------
    .. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
           FL, Academic Press, Inc., 1980, pp. 139-142.

    Examples
    --------
    The following example checks that ``a * a+ * a == a`` and
    ``a+ * a * a+ == a+``:
    >>> a = np.random.randn(2, 3)
    >>> pinv_a = np.linalg.pinv(a)
    >>> (a - np.dot(a, np.dot(pinv_a, a))).sum()
    array(0.)
    >>> (pinv_a - np.dot(pinv_a, np.dot(a, pinv_a))).sum()
    array(0.)
    """
    return _mx_nd_np.linalg.pinv(a, rtol, hermitian)


def norm(x, ord=None, axis=None, keepdims=False):
    r"""
    Matrix or vector norm.

    This function can only support Frobenius norm for now.
    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    Parameters
    ----------
    x : ndarray
        Input array.
    ord : {'fro'}, optional
        Order of the norm.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None, the norm of the whole ndarray is
        returned.

    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    Notes
    -----
    This operator differs from NumPy in the aspect that it always returns a
    zero-dim tensor for the cases where Python float values are expected
    in NumPy.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) - 4
    >>> a
    array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4., -3., -2.],
           [-1.,  0.,  1.],
           [ 2.,  3.,  4.]])
    >>> LA.norm(a)
    array(7.745967)
    >>>
    >>> LA.norm(b)
    array(7.745967)
    >>> LA.norm(b, 'fro')
    array(7.745967)
    """
    return _mx_nd_np.linalg.norm(x, ord, axis, keepdims)


def vector_norm(x, ord=None, axis=None, keepdims=False):
    r"""
    Computes the vector norm of a vector (or batch of vectors) `x`.

    Parameters
    ----------
    x : ndarray
        Input array. Should have a floating-point data type.
    ord : {non-zero int, inf, -inf}, optional
        Order of the norm.
    axis : {int, n-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a n-tuple, it specifies the
        axes along which to compute batched vector norms. If `axis` is None,
        the norm of the whole ndarray is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the vector(s).

    Notes
    -----
    `vector_norm` is a standard API in
    https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html#linalg-vector-norm-x-axis-none-keepdims-false-ord-2
    instead of an official NumPy operator.

    """
    if axis is None:
        x = x.flatten()
        axis = 0
    elif isinstance(axis, tuple):
        rest = tuple(i for i in range(x.ndim) if i not in axis)
        newshape = axis + rest
        x = _mx_nd_np.transpose(x, newshape).\
            reshape((reduce(lambda a, b: a * b, [x.shape[a] for a in axis]),\
                     *[x.shape[i] for i in rest]))
        axis = 0
    return _mx_nd_np.linalg.norm(x, axis=axis, keepdims=keepdims, ord=ord)


def matrix_norm(x, ord='fro', axis=(-2, -1), keepdims=False):
    r"""
    Computes the matrix norm of a matrix (or a stack of matrices) `x`.

    Parameters
    ----------
    x : ndarray
        Input array. Should have a floating-point data type.
    ord : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
        Order of the norm.
    axis : {2-tuple of ints}
        a 2-tuple which specifies the axes (dimensions) defining two-dimensional
        matrices for which to compute matrix norms.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix.

    Notes
    -----
    `matrix_norm` is a standard API in
    https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html#linalg-matrix-norm-x-axis-2-1-keepdims-false-ord-fro
    instead of an official NumPy operator.

    """
    if isinstance(axis, tuple) and len(axis) == 2:
        return _mx_nd_np.linalg.norm(x, axis=axis, keepdims=keepdims, ord=ord)
    raise ValueError("The axis of matrix_norm must be a 2-tuple of ints")


def svd(a):
    r"""
    Singular Value Decomposition.

    When `a` is a 2D array, it is factorized as ``ut @ np.diag(s) @ v``,
    where `ut` and `v` are 2D orthonormal arrays and `s` is a 1D
    array of `a`'s singular values. When `a` is higher-dimensional, SVD is
    applied in stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) ndarray
        A real array with ``a.ndim >= 2`` and ``M <= N``.

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

    .. note::
       The decomposition is performed using LAPACK routine ``_gesvd``.

       SVD is usually described for the factorization of a 2D matrix :math:`A`.
       The higher-dimensional case will be discussed below. In the 2D case, SVD is
       written as :math:`A = U^T S V`, where :math:`A = a`, :math:`U^T = ut`,
       :math:`S= \mathtt{np.diag}(s)` and :math:`V = v`. The 1D array `s`
       contains the singular values of `a` and `ut` and `v` are orthonormal. The rows
       of `v` are the eigenvectors of :math:`A^T A` and the columns of `ut` are
       the eigenvectors of :math:`A A^T`. In both cases the corresponding
       (possibly non-zero) eigenvalues are given by ``s**2``.

       The sign of rows of `u` and `v` are determined as described in
       `Auto-Differentiating Linear Algebra <https://arxiv.org/pdf/1710.08717.pdf>`_.

       If `a` has more than two dimensions, then broadcasting rules apply.
       This means that SVD is working in "stacked" mode: it iterates over
       all indices of the first ``a.ndim - 2`` dimensions and for each
       combination SVD is applied to the last two indices. The matrix `a`
       can be reconstructed from the decomposition with either
       ``(ut * s[..., None, :]) @ v`` or
       ``ut @ (s[..., None] * v)``. (The ``@`` operator denotes batch matrix multiplication)

       This function differs from the original `numpy.linalg.svd
       <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html>`_ in
       the following way(s):
       * The sign of rows of `u` and `v` may differ.
       * Does not support complex input.

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
    return _mx_nd_np.linalg.svd(a)


def svdvals(a):
    r"""
    Computes the singular values of a matrix (or a stack of matrices) `x`.

    Parameters
    ----------
    a : (..., M, N) ndarray
        A real array with ``a.ndim >= 2`` and ``M <= N``.

    Returns
    -------
    out : (..., M) ndarray
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.

    .. note::
       `svdvals` is a standard api in
       https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html#linalg-svdvals-x
       instead of an official NumPy operator.
    """
    _, s, _ = _mx_nd_np.linalg.svd(a)
    return s


def cholesky(a, upper=False):
    r"""
    Cholesky decomposition.

    Notes
    -----
    `upper` param is requested by API standardization in
    https://data-apis.org/array-api/latest/extensions/generated/signatures.linalg.cholesky.html
    instead of parameter in official NumPy operator.

    Return the Cholesky decomposition, `L * L.T`, of the square matrix `a`,
    where `L` is lower-triangular and .T is the transpose operator. `a` must be
    symmetric and positive-definite. Only `L` is actually returned. Complex-valued
    input is currently not supported.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Symmetric, positive-definite input matrix.
    upper : bool
        If `True`, the result must be the upper-triangular Cholesky factor.
        If `False`, the result must be the lower-triangular Cholesky factor.
        Default: `False`.

    Returns
    -------
    L : (..., M, M) ndarray
        Lower-triangular Cholesky factor of `a`.

    Raises
    ------
    MXNetError
        If the decomposition fails, for example, if `a` is not positive-definite.

    Notes
    -----
    Broadcasting rules apply.

    The Cholesky decomposition is often used as a fast way of solving

    .. math:: A \mathbf{x} = \mathbf{b}

    (when `A` is both symmetric and positive-definite).

    First, we solve for :math:`\mathbf{y}` in

    .. math:: L \mathbf{y} = \mathbf{b},

    and then for :math:`\mathbf{x}` in

    .. math:: L.T \mathbf{x} = \mathbf{y}.

    Examples
    --------
    >>> A = np.array([[16, 4], [4, 10]])
    >>> A
    array([[16.,  4.],
           [ 4., 10.]])
    >>> L = np.linalg.cholesky(A)
    >>> L
    array([[4., 0.],
           [1., 3.]])
    >>> np.dot(L, L.T)
    array([[16.,  4.],
           [ 4., 10.]])
    """
    return _mx_nd_np.linalg.cholesky(a, upper)


def qr(a, mode='reduced'):
    r"""
    Compute the qr factorization of a matrix a.
    Factor the matrix a as qr, where q is orthonormal and r is upper-triangular.

    Parameters
    ----------
    a : (..., M, N) ndarray
        Matrix or stack of matrices to be qr factored.
    mode: {‘reduced’, ‘complete’, ‘r’, ‘raw’, ‘full’, ‘economic’}, optional
        Only default mode, 'reduced', is implemented. If K = min(M, N), then
        * 'reduced’ : returns q, r with dimensions (M, K), (K, N) (default)

    Returns
    -------
    q : (..., M, K) ndarray
        A matrix or stack of matrices with K orthonormal columns, with K = min(M, N).
    r : (..., K, N) ndarray
        A matrix or stack of upper triangular matrices.

    Raises
    ------
    MXNetError
        If factoring fails.

    Notes
    -----
    Currently, the gradient for the QR factorization is well-defined
    only when the first K columns of the input matrix are linearly independent.

    Examples
    --------
    >>> from mxnet import np
    >>> a = np.random.uniform(-10, 10, (2, 2))
    >>> q, r = np.linalg.qr(a)
    >>> q
    array([[-0.22121978, -0.97522414],
           [-0.97522414,  0.22121954]])
    >>> r
    array([[-4.4131265 , -7.1255064 ],
           [ 0.        , -0.28771925]])
    >>> a = np.random.uniform(-10, 10, (2, 3))
    >>> q, r = np.linalg.qr(a)
    >>> q
    array([[-0.28376842, -0.9588929 ],
           [-0.9588929 ,  0.28376836]])
    >>> r
    array([[-7.242763  , -0.5673361 , -2.624416  ],
           [ 0.        , -7.297918  , -0.15949416]])
    >>> a = np.random.uniform(-10, 10, (3, 2))
    >>> q, r = np.linalg.qr(a)
    >>> q
    array([[-0.34515655,  0.10919492],
           [ 0.14765628, -0.97452265],
           [-0.92685735, -0.19591334]])
    >>> r
    array([[-8.453794,  8.4175  ],
           [ 0.      ,  5.430561]])
    """
    return _mx_nd_np.linalg.qr(a, mode)


def inv(a):
    r"""
    Compute the (multiplicative) inverse of a matrix.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Matrix to be inverted.

    Returns
    -------
    ainv : (..., M, M) ndarray
        (Multiplicative) inverse of the matrix `a`.

    Raises
    ------
    MXNetError
        If `a` is not square or inversion fails.

    Examples
    --------
    >>> from mxnet import np
    >>> a = np.array([[1., 2.], [3., 4.]])
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])

    Inverses of several matrices can be computed at once:

    >>> a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
    >>> np.linalg.inv(a)
    array([[[-2.        ,  1.        ],
            [ 1.5       , -0.5       ]],

           [[-1.2500001 ,  0.75000006],
            [ 0.75000006, -0.25000003]]])
    """
    return _mx_nd_np.linalg.inv(a)


def det(a):
    r"""
    Compute the determinant of an array.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Input array to compute determinants for.

    Returns
    -------
    det : (...) ndarray
        Determinant of `a`.

    See Also
    --------
    slogdet : Another way to represent the determinant, more suitable
    for large matrices where underflow/overflow may occur.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.
    The determinant is computed via LU factorization using the LAPACK
    routine z/dgetrf.

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.linalg.det(a)
    -2.0

    Computing determinants for a stack of matrices:
    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)

    >>> np.linalg.det(a)
    array([-2., -3., -8.])
    """
    return _mx_nd_np.linalg.det(a)


def slogdet(a):
    r"""
    Compute the sign and (natural) logarithm of the determinant of an array.
    If an array has a very small or very large determinant, then a call to
    `det` may overflow or underflow. This routine is more robust against such
    issues, because it computes the logarithm of the determinant rather than
    the determinant itself.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Input array, has to be a square 2-D array.

    Returns
    -------
    sign : (...) ndarray
        A number representing the sign of the determinant. For a real matrix,
        this is 1, 0, or -1.
    logdet : (...) array_like
        The natural log of the absolute value of the determinant.
    If the determinant is zero, then `sign` will be 0 and `logdet` will be
    -Inf. In all cases, the determinant is equal to ``sign * np.exp(logdet)``.

    See Also
    --------
    det

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.
    The determinant is computed via LU factorization using the LAPACK
    routine z/dgetrf.

    Examples
    --------
    The determinant of a 2-D array ``[[a, b], [c, d]]`` is ``ad - bc``:
    >>> a = np.array([[1, 2], [3, 4]])
    >>> (sign, logdet) = np.linalg.slogdet(a)
    >>> (sign, logdet)
    (-1., 0.69314718055994529)

    >>> sign * np.exp(logdet)
    -2.0

    Computing log-determinants for a stack of matrices:
    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)

    >>> sign, logdet = np.linalg.slogdet(a)
    >>> (sign, logdet)
    (array([-1., -1., -1.]), array([ 0.69314718,  1.09861229,  2.07944154]))

    >>> sign * np.exp(logdet)
    array([-2., -3., -8.])

    This routine succeeds where ordinary `det` does not:
    >>> np.linalg.det(np.eye(500) * 0.1)
    0.0
    >>> np.linalg.slogdet(np.eye(500) * 0.1)
    (1., -1151.2925464970228)
    """
    return _mx_nd_np.linalg.slogdet(a)


def solve(a, b):
    r"""
    Solve a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, `x`, of the well-determined, i.e., full
    rank, linear matrix equation `ax = b`.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Coefficient matrix.
    b : {(..., M,), (..., M, K)}, ndarray
        Ordinate or "dependent variable" values.

    Returns
    -------
    x : {(..., M,), (..., M, K)} ndarray
        Solution to the system a x = b.  Returned shape is identical to `b`.

    Raises
    ------
    MXNetError
        If `a` is singular or not square.

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The solutions are computed using LAPACK routine ``_gesv``.

    `a` must be square and of full-rank, i.e., all rows (or, equivalently,
    columns) must be linearly independent; if either is not true, use
    `lstsq` for the least-squares best "solution" of the
    system/equation.

    Examples
    --------
    Solve the system of equations ``3 * x0 + x1 = 9`` and ``x0 + 2 * x1 = 8``:

    >>> a = np.array([[3,1], [1,2]])
    >>> b = np.array([9,8])
    >>> x = np.linalg.solve(a, b)
    >>> x
    array([2.,  3.])

    Check that the solution is correct:

    >>> np.allclose(np.dot(a, x), b)
    True
    """
    return _mx_nd_np.linalg.solve(a, b)


def tensorinv(a, ind=2):
    r"""
    Compute the 'inverse' of an N-dimensional array.

    The result is an inverse for `a` relative to the tensordot operation
    ``tensordot(a, b, ind)``, i. e., up to floating-point accuracy,
    ``tensordot(tensorinv(a), a, ind)`` is the "identity" tensor for the
    tensordot operation.

    Parameters
    ----------
    a : array_like
        Tensor to 'invert'. Its shape must be 'square', i. e.,
        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
    ind : int, optional
        Number of first indices that are involved in the inverse sum.
        Must be a positive integer, default is 2.

    Returns
    -------
    b : ndarray
        `a`'s tensordot inverse, shape ``a.shape[ind:] + a.shape[:ind]``.

    Raises
    ------
    MXNetError
        If `a` is singular or not 'square' (in the above sense).

    See Also
    --------
    tensordot, tensorsolve

    Examples
    --------
    >>> a = np.eye(4*6)
    >>> a.shape = (4, 6, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=2)
    >>> ainv.shape
    (8, 3, 4, 6)
    >>> b = np.random.randn(4, 6)
    >>> np.allclose(np.tensordot(ainv, b), np.linalg.tensorsolve(a, b))
    True

    >>> a = np.eye(4*6)
    >>> a.shape = (24, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=1)
    >>> ainv.shape
    (8, 3, 24)
    >>> b = np.random.randn(24)
    >>> np.allclose(np.tensordot(ainv, b, 1), np.linalg.tensorsolve(a, b))
    True
    """
    return _mx_nd_np.linalg.tensorinv(a, ind)


def tensorsolve(a, b, axes=None):
    r"""
    Solve the tensor equation ``a x = b`` for x.
    It is assumed that all indices of `x` are summed over in the product,
    together with the rightmost indices of `a`, as is done in, for example,
    ``tensordot(a, x, axes=b.ndim)``.

    Parameters
    ----------
    a : ndarray
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
        'square').
    b : ndarray
        Right-hand tensor, which can be of any shape.
    axes : tuple of ints, optional
        Axes in `a` to reorder to the right, before inversion.
        If None (default), no reordering is done.

    Returns
    -------
    x : ndarray, shape Q

    Raises
    ------
    MXNetError
        If `a` is singular or not 'square' (in the above sense).

    See Also
    --------
    numpy.tensordot, tensorinv, numpy.einsum

    Examples
    --------
    >>> a = np.eye(2*3*4)
    >>> a.shape = (2*3, 4, 2, 3, 4)
    >>> b = np.random.randn(2*3, 4)
    >>> x = np.linalg.tensorsolve(a, b)
    >>> x.shape
    (2, 3, 4)
    >>> np.allclose(np.tensordot(a, x, axes=3), b)
    True
    """
    return _mx_nd_np.linalg.tensorsolve(a, b, axes)


def eigvals(a):
    r"""
    Compute the eigenvalues of a general matrix.

    Main difference between `eigvals` and `eig`: the eigenvectors aren't
    returned.

    Parameters
    ----------
    a : (..., M, M) ndarray
        A real-valued matrix whose eigenvalues will be computed.

    Returns
    -------
    w : (..., M,) ndarray
        The eigenvalues, each repeated according to its multiplicity.
        They are not necessarily ordered.

    Raises
    ------
    MXNetError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays
    eigh : eigenvalues and eigenvectors of a real symmetric array.
    eigvalsh : eigenvalues of a real symmetric.

    .. note::
       Broadcasting rules apply, see the `numpy.linalg` documentation for
       details.

       This is implemented using the ``_geev`` LAPACK routines which compute
       the eigenvalues and eigenvectors of general square arrays.

       This function differs from the original `numpy.linalg.eigvals
       <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigvals.html>`_ in
       the following way(s):
       * Does not support complex input and output.

    Examples
    --------
    Illustration, using the fact that the eigenvalues of a diagonal matrix
    are its diagonal elements, that multiplying a matrix on the left
    by an orthogonal matrix, `Q`, and on the right by `Q.T` (the transpose
    of `Q`), preserves the eigenvalues of the "middle" matrix.  In other words,
    if `Q` is orthogonal, then ``Q * A * Q.T`` has the same eigenvalues as
    ``A``:

    >>> from numpy import linalg as LA
    >>> x = np.random.random()
    >>> Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    >>> LA.norm(Q[0, :]), LA.norm(Q[1, :]), np.dot(Q[0, :],Q[1, :])
    (1.0, 1.0, 0.0)

    Now multiply a diagonal matrix by ``Q`` on one side and by ``Q.T`` on the other:

    >>> D = np.diag((-1,1))
    >>> LA.eigvals(D)
    array([-1.,  1.])
    >>> A = np.dot(Q, D)
    >>> A = np.dot(A, Q.T)
    >>> LA.eigvals(A)
    array([ 1., -1.]) # random
    """
    return _mx_nd_np.linalg.eigvals(a)


@wrap_data_api_linalg_func
def eigvalsh(a, upper=False):
    r"""
    Compute the eigenvalues real symmetric matrix.

    Main difference from eigh: the eigenvectors are not computed.

    Parameters
    ----------
    a : (..., M, M) ndarray
        A real-valued matrix whose eigenvalues are to be computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will
        be considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    w : (..., M,) ndarray
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.

    Raises
    ------
    MXNetError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays
    eigvals : eigenvalues of a non-symmetric array.
    eigh : eigenvalues and eigenvectors of a real symmetric array.

    .. note::
       Broadcasting rules apply, see the `numpy.linalg` documentation for
       details.

       The eigenvalues are computed using LAPACK routines ``_syevd``.

       This function differs from the original `numpy.linalg.eigvalsh
       <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigvalsh.html>`_ in
       the following way(s):
       * Does not support complex input and output.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[ 5.4119368 ,  8.996273  , -5.086096  ],
    ...               [ 0.8866155 ,  1.7490431 , -4.6107802 ],
    ...               [-0.08034172,  4.4172044 ,  1.4528792 ]])
    >>> LA.eigvalsh(a, UPLO='L')
    array([-2.87381886,  5.10144682,  6.38623114]) # in ascending order
    """
    if not upper:
        UPLO = 'L'
    else:
        UPLO = 'U'
    return _mx_nd_np.linalg.eigvalsh(a, UPLO)


def eig(a):
    r"""
    Compute the eigenvalues and right eigenvectors of a square array.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Matrices for which the eigenvalues and right eigenvectors will
        be computed

    Returns
    -------
    w : (..., M) ndarray
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered.
    v : (..., M, M) ndarray
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.

    Raises
    ------
    MXNetError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eigvals : eigenvalues of a non-symmetric array.
    eigh : eigenvalues and eigenvectors of a real symmetric array.
    eigvalsh : eigenvalues of a real symmetric.

    .. note::
       This is implemented using the ``_geev`` LAPACK routines which compute
       the eigenvalues and eigenvectors of general square arrays.

       The number `w` is an eigenvalue of `a` if there exists a vector
       `v` such that ``dot(a,v) = w * v``. Thus, the arrays `a`, `w`, and
       `v` satisfy the equations ``dot(a[:,:], v[:,i]) = w[i] * v[:,i]``
       for :math:`i \\in \\{0,...,M-1\\}`.

       The array `v` of eigenvectors may not be of maximum rank, that is, some
       of the columns may be linearly dependent, although round-off error may
       obscure that fact. If the eigenvalues are all different, then theoretically
       the eigenvectors are linearly independent.

       This function differs from the original `numpy.linalg.eig
       <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html>`_ in
       the following way(s):
       * Does not support complex input and output.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[-1.9147992 ,  6.054115  , 18.046988  ],
    ...               [ 0.77563655, -4.860152  ,  2.1012988 ],
    ...               [ 2.6083658 ,  2.3705218 ,  0.3192524 ]])
    >>> w, v = LA.eig(a)
    >>> w
    array([ 6.9683027, -7.768063 , -5.655937 ])
    >>> v
    array([[ 0.90617794,  0.9543622 ,  0.2492316 ],
           [ 0.13086087, -0.04077047, -0.9325615 ],
           [ 0.4021404 , -0.29585576,  0.26117516]])
    """
    return _mx_nd_np.linalg.eig(a)


@wrap_data_api_linalg_func
def eigh(a, upper=False):
    r"""
    Return the eigenvalues and eigenvectors real symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    Parameters
    ----------
    a : (..., M, M) ndarray
        real symmetric matrices whose eigenvalues and eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will
        be considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    w : (..., M) ndarray
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.
    v : {(..., M, M) ndarray, (..., M, M) matrix}
        The column ``v[:, i]`` is the normalized eigenvector corresponding
        to the eigenvalue ``w[i]``.  Will return a matrix object if `a` is
        a matrix object.

    Raises
    ------
    MXNetError
        If the eigenvalue computation does not converge.

    See Also
    --------
    eig : eigenvalues and right eigenvectors of general arrays
    eigvals : eigenvalues of a non-symmetric array.
    eigvalsh : eigenvalues of a real symmetric.

    .. note::

       The eigenvalues/eigenvectors are computed using LAPACK routines ``_syevd``.

       This function differs from the original `numpy.linalg.eigh
       <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html>`_ in
       the following way(s):
       * Does not support complex input and output.

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.array([[ 6.8189726 , -3.926585  ,  4.3990498 ],
    ...               [-0.59656644, -1.9166266 ,  9.54532   ],
    ...               [ 2.1093285 ,  0.19688708, -1.1634291 ]])
    >>> w, v = LA.eigh(a, upper=False)
    >>> w
    array([-2.175445 , -1.4581827,  7.3725457])
    >>> v
    array([[ 0.1805163 , -0.16569263,  0.9695154 ],
           [ 0.8242942 ,  0.56326365, -0.05721384],
           [-0.53661287,  0.80949366,  0.23825769]])
    """
    if not upper:
        UPLO = 'L'
    else:
        UPLO = 'U'
    return _mx_nd_np.linalg.eigh(a, UPLO)
