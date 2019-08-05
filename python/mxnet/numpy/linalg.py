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

from __future__ import absolute_import
from ..ndarray import numpy as _mx_nd_np


__all__ = ['norm']


def norm(x, ord=None, axis=None, keepdims=False):
    r"""Matrix or vector norm..
    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : ndarray
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
    n : ndarray
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
    return _mx_nd_np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
