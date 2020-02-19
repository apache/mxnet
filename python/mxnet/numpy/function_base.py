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

"""Numpy basic functions."""

from .stride_tricks import broadcast_arrays

__all__ = ['meshgrid']


def meshgrid(*xi, **kwargs):
    """
    Return coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    Parameters
    ----------
    x1, x2,..., xn : ndarrays
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.

    sparse : bool, optional
        If True a sparse grid is returned in order to conserve memory.
        Default is False. Please note that `sparse=True` is currently
        not supported.

    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True. Please note that `copy=False`
        is currently not supported.

    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape
    (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
    with inputs of length M, N and P, outputs are of shape (N, M, P) for
    'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
    illustrated by the following code snippet::

        xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.
    """
    ndim = len(xi)

    copy_ = kwargs.pop('copy', True)
    if not copy_:
        raise NotImplementedError('copy=False is not implemented')
    sparse = kwargs.pop('sparse', False)
    if sparse:
        raise NotImplementedError('sparse=False is not implemented')
    indexing = kwargs.pop('indexing', 'xy')

    if kwargs:
        raise TypeError("meshgrid() got an unexpected keyword argument '%s'"
                        % (list(kwargs)[0],))

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1:])
              for i, x in enumerate(xi)]

    if indexing == 'xy' and ndim > 1:
        # switch first and second axis
        output[0] = output[0].reshape(1, -1, *s0[2:])
        output[1] = output[1].reshape(-1, 1, *s0[2:])

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = broadcast_arrays(*output)

    return output
