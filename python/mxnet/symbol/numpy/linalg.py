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

"""Namespace for operators used in Gluon dispatched by F=symbol."""

from . import _symbol
from . import _op as _mx_sym_np
from . import _internal as _npi

__all__ = ['norm', 'svd', 'cholesky', 'inv', 'det', 'slogdet', 'solve', 'tensorinv', 'tensorsolve', 'pinv',
           'eigvals', 'eig', 'eigvalsh', 'eigh']


def pinv(a, rcond=1e-15, hermitian=False):
    r"""
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    *large* singular values.

    Parameters
    ----------
    a : (..., M, N) ndarray
        Matrix or stack of matrices to be pseudo-inverted.
    rcond : (...) {float or ndarray of float}, optional
        Cutoff for small singular values.
        Singular values less than or equal to
        ``rcond * largest_singular_value`` are set to zero.
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
    if hermitian is True:
        raise NotImplementedError("hermitian is not supported yet...")
    if _symbol._np.isscalar(rcond):
        return _npi.pinv_scalar_rcond(a, rcond, hermitian)
    return _npi.pinv(a, rcond, hermitian)


# pylint: disable=too-many-return-statements
def norm(x, ord=None, axis=None, keepdims=False):
    r"""Matrix or vector norm.
    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.
    Parameters
    ----------
    x : _Symbol
        Input array.  If `axis` is None, `x` must be 1-D or 2-D.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.
    Returns
    -------
    n : _Symbol
        Norm of the matrix or vector(s).
    Notes
    -----
    For values of ``ord <= 0``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.
    The following norms can be calculated:
    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  --                            --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      --                            as below
    -2     --                            as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================
    The Frobenius norm is given by [1]_:
        :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`
    The nuclear norm is the sum of the singular values.
    When you want to operate norm for matrices,if you ord is (-1, 1, inf, -inf),
    you must give you axis, it is not support default axis.
    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
    Examples
    --------
    >>> from mxnet import np
    >>> a = np.arange(9) - 4
    >>> a
    array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4., -3., -2.],
           [-1.,  0.,  1.],
           [ 2.,  3.,  4.]])
    >>> np.linalg.norm(a)
    array(7.745967)
    >>> np.linalg.norm(b)
    array(7.745967)
    >>> np.linalg.norm(b, 'fro')
    array(7.745967)
    >>> np.linalg.norm(a, 'inf')
    array(4.)
    >>> np.linalg.norm(b, 'inf', axis=(0, 1))
    array(9.)
    >>> np.linalg.norm(a, '-inf')
    array(0.)
    >>> np.linalg.norm(b, '-inf', axis=(0, 1))
    array(2.)
    >>> np.linalg.norm(a, 1)
    array(20.)
    >>> np.linalg.norm(b, 1, axis=(0, 1))
    array(7.)
    >>> np.linalg.norm(a, -1)
    array(0.)
    >>> np.linalg.norm(b, -1, axis=(0, 1))
    array(6.)
    >>> np.linalg.norm(a, 2)
    array(7.745967)
    >>> np.linalg.norm(a, -2)
    array(0.)
    >>> np.linalg.norm(a, 3)
    array(5.8480353)
    >>> np.linalg.norm(a, -3)
    array(0.)
    Using the `axis` argument to compute vector norms:
    >>> c = np.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> np.linalg.norm(c, axis=0)
    array([1.4142135, 2.236068 , 5.       ])
    >>> np.linalg.norm(c, axis=1)
    array([3.7416573, 4.2426405])
    >>> np.linalg.norm(c, ord=1, axis=1)
    array([6., 6.])
    Using the `axis` argument to compute matrix norms:
    >>> m = np.arange(8).reshape(2,2,2)
    >>> np.linalg.norm(m, axis=(1,2))
    array([ 3.7416573, 11.224973 ])
    >>> np.linalg.norm(m[0, :, :]), np.linalg.norm(m[1, :, :])
    (array(3.7416573), array(11.224973))
    """
    if axis is None and ord is None:
        return _npi.norm(x, ord=2, axis=None, keepdims=keepdims, flag=-2)
    if axis is None or isinstance(axis, (int, tuple)):  # pylint: disable=too-many-nested-blocks
        if axis is not None:
            if isinstance(axis, int):
                axis = (axis, )
            if len(axis) == 2:
                if ord in ['inf', '-inf']:
                    row_axis, col_axis = axis
                    if not keepdims:
                        if row_axis > col_axis:
                            row_axis -= 1
                    if ord == 'inf':
                        return _mx_sym_np.sum(_symbol.abs(x), axis=col_axis, keepdims=keepdims).max(axis=row_axis, keepdims=keepdims)  # pylint: disable=line-too-long
                    else:
                        return _mx_sym_np.sum(_symbol.abs(x), axis=col_axis, keepdims=keepdims).min(axis=row_axis, keepdims=keepdims)  # pylint: disable=line-too-long
                if ord in [1, -1]:
                    row_axis, col_axis = axis
                    if not keepdims:
                        if row_axis < col_axis:
                            col_axis -= 1
                    if ord == 1:
                        return _mx_sym_np.sum(_symbol.abs(x), axis=row_axis, keepdims=keepdims).max(axis=col_axis, keepdims=keepdims)  # pylint: disable=line-too-long
                    elif ord == -1:
                        return _mx_sym_np.sum(_symbol.abs(x), axis=row_axis, keepdims=keepdims).min(axis=col_axis, keepdims=keepdims)  # pylint: disable=line-too-long
                if ord in [2, -2]:
                    return _npi.norm(x, ord=ord, axis=axis, keepdims=keepdims, flag=0)
                if ord is None:
                    return _npi.norm(x, ord=2, axis=axis, keepdims=keepdims, flag=1)
        if ord == 'inf':
            return _mx_sym_np.max(_symbol.abs(x), axis=axis, keepdims=keepdims)
            #return _npi.norm(x, ord=float('inf'), axis=axis, keepdims=keepdims, flag=3)
        elif ord == '-inf':
            return _mx_sym_np.min(_symbol.abs(x), axis=axis, keepdims=keepdims)
            #return _npi.norm(x, ord=-float('inf'), axis=axis, keepdims=keepdims, flag=4)
        elif ord is None:
            return _npi.norm(x, ord=2, axis=axis, keepdims=keepdims, flag=1)
        elif ord == 2:
            return _npi.norm(x, ord=2, axis=axis, keepdims=keepdims, flag=-1)
        elif ord == 'nuc':
            return _npi.norm(x, ord=2, axis=axis, keepdims=keepdims, flag=2)
        elif ord in ['fro', 'f']:
            return _npi.norm(x, ord=2, axis=axis, keepdims=keepdims, flag=1)
        else:
            return _npi.norm(x, ord=ord, axis=axis, keepdims=keepdims, flag=-1)
    else:
        raise TypeError("'axis' must be None, an integer or a tuple of integers.")
# pylint: enable=too-many-return-statements


def svd(a):
    r"""
    Singular Value Decomposition.

    When `a` is a 2D array, it is factorized as ``ut @ np.diag(s) @ v``,
    where `ut` and `v` are 2D orthonormal arrays and `s` is a 1D
    array of `a`'s singular values. When `a` is higher-dimensional, SVD is
    applied in stacked mode as explained below.

    Parameters
    ----------
    a : (..., M, N) _Symbol
        A real array with ``a.ndim >= 2`` and ``M <= N``.

    Returns
    -------
    ut: (..., M, M) _Symbol
        Orthonormal array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    s : (..., M) _Symbol
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    v : (..., M, N) _Symbol
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
     - The sign of rows of `u` and `v` may differ.
     - Does not support complex input.
    """
    return _npi.svd(a)


def cholesky(a):
    r"""
    Cholesky decomposition.

    Return the Cholesky decomposition, `L * L.T`, of the square matrix `a`,
    where `L` is lower-triangular and .T is the transpose operator. `a` must be
    symmetric and positive-definite. Only `L` is actually returned. Complex-valued
    input is currently not supported.

    Parameters
    ----------
    a : (..., M, M) ndarray
        Symmetric, positive-definite input matrix.

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
    return _npi.cholesky(a)


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
    return _npi.inv(a)


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
    return _npi.det(a)


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
    return _npi.slogdet(a)


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
    return _npi.solve(a, b)


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
    return _npi.tensorinv(a, ind)


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
    return _npi.tensorsolve(a, b, axes)


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

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    This is implemented using the ``_geev`` LAPACK routines which compute
    the eigenvalues and eigenvectors of general square arrays.

    This function differs from the original `numpy.linalg.eigvals
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigvals.html>`_ in
    the following way(s):
     - Does not support complex input and output.
    """
    return _npi.eigvals(a)


def eigvalsh(a, UPLO='L'):
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

    Notes
    -----
    Broadcasting rules apply, see the `numpy.linalg` documentation for
    details.

    The eigenvalues are computed using LAPACK routines ``_syevd``.

    This function differs from the original `numpy.linalg.eigvalsh
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigvalsh.html>`_ in
    the following way(s):
     - Does not support complex input and output.
    """
    return _npi.eigvalsh(a, UPLO)


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

    Notes
    -----
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
     - Does not support complex input and output.
    """
    return _npi.eig(a)


def eigh(a, UPLO='L'):
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

    Notes
    -----
    The eigenvalues/eigenvectors are computed using LAPACK routines ``_syevd``.

    This function differs from the original `numpy.linalg.eigh
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html>`_ in
    the following way(s):
     - Does not support complex input and output.
    """
    return _npi.eigh(a, UPLO)
