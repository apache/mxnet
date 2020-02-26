# pylint: disable=C0302
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

# pylint: disable=unused-argument
"""Namespace for numpy operators used in Gluon dispatched by F=ndarray."""

import numpy as _np
from ...base import numeric_types, integer_types
from ...util import _sanity_check_params, set_module
from ...util import wrap_np_unary_func, wrap_np_binary_func
from ...context import current_context
from . import _internal as _npi
from . import _api_internal
from ..ndarray import NDArray


__all__ = ['shape', 'zeros', 'zeros_like', 'ones', 'ones_like', 'full', 'full_like', 'empty_like', 'invert', 'delete',
           'add', 'broadcast_to', 'subtract', 'multiply', 'divide', 'mod', 'remainder', 'power', 'bitwise_not',
           'arctan2', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'log10', 'sqrt', 'cbrt', 'abs', 'insert', 'fabs',
           'absolute', 'exp', 'expm1', 'arcsin', 'arccos', 'arctan', 'sign', 'log', 'degrees', 'log2', 'matmul',
           'log1p', 'rint', 'radians', 'reciprocal', 'square', 'negative', 'fix', 'ceil', 'floor', 'histogram',
           'trunc', 'logical_not', 'arcsinh', 'arccosh', 'arctanh', 'argsort', 'sort',
           'tensordot', 'eye', 'linspace',
           'logspace', 'expand_dims', 'tile', 'arange', 'array_split', 'split', 'hsplit', 'vsplit', 'dsplit',
           'concatenate', 'append', 'stack', 'vstack', 'row_stack', 'column_stack', 'hstack', 'dstack',
           'average', 'mean', 'maximum', 'minimum', 'around', 'round', 'round_',
           'swapaxes', 'clip', 'argmax', 'argmin', 'std', 'var', 'indices', 'copysign', 'ravel', 'unravel_index',
           'diag_indices_from', 'hanning', 'hamming', 'blackman', 'flip', 'flipud', 'fliplr',
           'hypot', 'bitwise_and', 'bitwise_xor', 'bitwise_or', 'rad2deg', 'deg2rad', 'unique', 'lcm',
           'tril', 'identity', 'take', 'ldexp', 'vdot', 'inner', 'outer',
           'equal', 'not_equal', 'greater', 'less', 'greater_equal', 'less_equal', 'rot90', 'einsum',
           'true_divide', 'nonzero', 'quantile', 'percentile', 'shares_memory', 'may_share_memory',
           'diff', 'ediff1d', 'resize', 'polyval', 'nan_to_num', 'isnan', 'isinf', 'isposinf', 'isneginf', 'isfinite',
           'where', 'bincount', 'pad']


@set_module('mxnet.ndarray.numpy')
def shape(a):
    """
    Return the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    See Also
    --------
    ndarray.shape : Equivalent array method.

    Examples
    --------
    >>> np.shape(np.eye(3))
    (3, 3)
    >>> np.shape([[1, 2]])
    (1, 2)
    >>> np.shape([0])
    (1,)
    >>> np.shape(0)
    ()
    """
    return a.shape


@set_module('mxnet.ndarray.numpy')
def zeros(shape, dtype=None, order='C', ctx=None):  # pylint: disable=redefined-outer-name
    """Return a new array of given shape and type, filled with zeros.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `numpy.float32`. Note that this
        behavior is different from NumPy's `zeros` function where `float64`
        is the default value, because `float32` is considered as the default
        data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    # If the following code (4 lines) regarding ctx is removed
    # np.zeros((3, 4)) can be as fast as 4.96 us
    if ctx is None:
        ctx = str(current_context())
    else:
        ctx = str(ctx)
    if dtype is not None and not isinstance(dtype, str):
        dtype = _np.dtype(dtype).name
    return _api_internal.zeros(shape, dtype, ctx)


@set_module('mxnet.ndarray.numpy')
def ones(shape, dtype=_np.float32, order='C', ctx=None):  # pylint: disable=redefined-outer-name
    """Return a new array of given shape and type, filled with ones.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `numpy.float32`. Note that this
        behavior is different from NumPy's `ones` function where `float64`
        is the default value, because `float32` is considered as the default
        data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        Array of ones with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.ones(shape=shape, ctx=ctx, dtype=dtype)


# pylint: disable=too-many-arguments, redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def zeros_like(a, dtype=None, order='C', ctx=None, out=None):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
        Temporarily do not support boolean type.
    order : {'C'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. Currently only supports C order.
    ctx: to specify the device, e.g. the i-th GPU.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray
          Array of zeros with the same shape and type as a.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0., 1., 2.],
           [3., 4., 5.]])
    >>> np.zeros_like(x)
    array([[0., 0., 0.],
           [0., 0., 0.]])
    >>> np.zeros_like(x, int)
    array([[0, 0, 0],
           [0, 0, 0]], dtype=int64)
    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.], dtype=float64)
    >>> np.zeros_like(y)
    array([0., 0., 0.], dtype=float64)
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    return _npi.full_like(a, fill_value=0, dtype=dtype, ctx=ctx, out=out)


@set_module('mxnet.ndarray.numpy')
def ones_like(a, dtype=None, order='C', ctx=None, out=None):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
        Temporarily do not support boolean type.
    order : {'C'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. Currently only supports C order.
    ctx: to specify the device, e.g. the i-th GPU.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as a.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    ones : Return a new array setting values to one.

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0., 1., 2.],
           [3., 4., 5.]])
    >>> np.ones_like(x)
    array([[1., 1., 1.],
           [1., 1., 1.]])
    >>> np.ones_like(x, int)
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int64)
    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.], dtype=float64)
    >>> np.ones_like(y)
    array([1., 1., 1.], dtype=float64)
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    return _npi.full_like(a, fill_value=1, dtype=dtype, ctx=ctx, out=out)


@set_module('mxnet.ndarray.numpy')
def broadcast_to(array, shape):
    """
    Broadcast an array to a new shape.

    Parameters
    ----------
    array : ndarray or scalar
        The array to broadcast.
    shape : tuple
        The shape of the desired array.

    Returns
    -------
    broadcast : array
        A readonly view on the original array with the given shape. It is
        typically not contiguous. Furthermore, more than one element of a
        broadcasted array may refer to a single memory location.

    Raises
    ------
    MXNetError
        If the array is not compatible with the new shape according to NumPy's
        broadcasting rules.
    """
    if _np.isscalar(array):
        return full(shape, array)
    return _npi.broadcast_to(array, shape)


@set_module('mxnet.ndarray.numpy')
def full(shape, fill_value, dtype=None, order='C', ctx=None, out=None):  # pylint: disable=too-many-arguments
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar or ndarray
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array. The default, `None`, means
        `np.array(fill_value).dtype`.
    order : {'C'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. Currently only supports C order.
    ctx: to specify the device, e.g. the i-th GPU.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape, dtype, and order.
        If `fill_value` is an ndarray, out will have the same context as `fill_value`
        regardless of the provided `ctx`.

    Notes
    -----
    This function differs from the original `numpy.full
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html`_ in
    the following way(s):
    - Have an additional `ctx` argument to specify the device
    - Have an additional `out` argument
    - Currently does not support `order` selection

    See Also
    --------
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Examples
    --------
    >>> np.full((2, 2), 10)
    array([[10., 10.],
           [10., 10.]])
    >>> np.full((2, 2), 2, dtype=np.int32, ctx=mx.cpu(0))
    array([[2, 2],
           [2, 2]], dtype=int32)

    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    if isinstance(fill_value, NDArray):
        if dtype is None:
            ret = broadcast_to(fill_value, shape)
        else:
            ret = broadcast_to(fill_value, shape).astype(dtype)
        return ret
    if isinstance(fill_value, bool):
        fill_value = int(fill_value)
        dtype = _np.bool if dtype is None else dtype
    dtype = _np.float32 if dtype is None else dtype
    return _npi.full(shape=shape, value=fill_value, ctx=ctx, dtype=dtype, out=out)
# pylint: enable=too-many-arguments, redefined-outer-name


@set_module('mxnet.ndarray.numpy')
def full_like(a, fill_value, dtype=None, order='C', ctx=None, out=None): # pylint: disable=too-many-arguments
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : ndarray
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
        Temporarily do not support boolean type.
    order : {'C'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. Currently only supports C order.
    ctx: to specify the device, e.g. the i-th GPU.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> x = np.arange(6, dtype=int)
    >>> np.full_like(x, 1)
    array([1, 1, 1, 1, 1, 1], dtype=int64)
    >>> np.full_like(x, 0.1)
    array([0, 0, 0, 0, 0, 0], dtype=int64)
    >>> np.full_like(x, 0.1, dtype=np.float64)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float64)
    >>> np.full_like(x, np.nan, dtype=np.double)
    array([nan, nan, nan, nan, nan, nan], dtype=float64)
    >>> y = np.arange(6, dtype=np.float32)
    >>> np.full_like(y, 0.1)
    array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    if isinstance(fill_value, bool):
        fill_value = int(fill_value)
    return _npi.full_like(a, fill_value=fill_value, dtype=dtype, ctx=ctx, out=out)


@set_module('mxnet.ndarray.numpy')
def empty_like(prototype, dtype=None, order='C', subok=False, shape=None): # pylint: disable=W0621
    """
    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : ndarray
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. Currently only supports C order.
    subok : {False}, optional
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to False.
        (Only support False at this moment)
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
        (Not supported at this moment)

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `prototype`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.

    Notes
    -----
    This function does *not* initialize the returned array; to do that use
    `zeros_like` or `ones_like` instead.  It may be marginally faster than
    the functions that do set the array values.

    Examples
    --------
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> np.empty_like(a)
    array([[-5764607523034234880, -2305834244544065442,           4563075075], # uninitialized
           [          4567052944, -5764607523034234880,      844424930131968]])
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    >>> np.empty_like(a)
    array([[4.9e-324, 9.9e-324, 1.5e-323], # uninitialized
           [2.0e-323, 2.5e-323, 3.0e-323]])
    """
    dtype_list = {None:'None', _np.int8:'int8', _np.uint8:'uint8', _np.int32:'int32',
                  _np.int64:'int64', _np.float16:'float16', _np.float32:'float32',
                  _np.float64:'float64', _np.bool_:'bool_', bool:'bool', int:'int64', float:'float64'}
    if order != 'C':
        raise NotImplementedError("Only support C-order at this moment")
    if subok:
        raise NotImplementedError("Creating array by using sub-class is not supported at this moment")
    if shape is not None:
        raise NotImplementedError("Assigning new shape is not supported at this moment")
    try:
        dtype = dtype if isinstance(dtype, str) else dtype_list[dtype]
    except:
        raise NotImplementedError("Do not support this dtype at this moment")
    return _npi.empty_like_fallback(prototype, dtype=dtype, order=order, subok=subok, shape=shape)


@set_module('mxnet.ndarray.numpy')
def arange(start, stop=None, step=1, dtype=None, ctx=None):
    """Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    Parameters
    ----------
    start : number, optional
        Start of interval. The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval. The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values. For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array. The default is `float32`.

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.
    """
    if dtype is None:
        dtype = 'float32'
    if ctx is None:
        ctx = current_context()
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    if start is None and stop is None:
        raise ValueError('start and stop cannot be both None')
    if step == 0:
        raise ZeroDivisionError('step cannot be 0')
    return _npi.arange(start=start, stop=stop, step=step, dtype=dtype, ctx=ctx)


@set_module('mxnet.ndarray.numpy')
def identity(n, dtype=None, ctx=None):
    """
    Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``numpy.float32``.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> np.identity(3)
    >>> np.identity(3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    if not isinstance(n, int):
        raise TypeError("Input 'n' should be an integer")
    if n < 0:
        raise ValueError("Input 'n' cannot be negative")
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.identity(shape=(n, n), ctx=ctx, dtype=dtype)


# pylint: disable=redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def take(a, indices, axis=None, mode='raise', out=None):
    r"""
    Take elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``np.take(arr, indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    Explained without fancy indexing, this is equivalent to the following use
    of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
    indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        Nj = indices.shape
        for ii in ndindex(Ni):
            for jj in ndindex(Nj):
                for kk in ndindex(Nk):
                    out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

    Parameters
    ----------
    a : ndarray
        The source array.
    indices : ndarray
        The indices of the values to extract. Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.
    out : ndarray, optional
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype.
    mode : {'clip', 'wrap'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'clip' -- clip to the range (default)
        * 'wrap' -- wrap around

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    Returns
    -------
    out : ndarray
        The returned array has the same type as `a`.

    Notes
    -----

    This function differs from the original `numpy.take
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html>`_ in
    the following way(s):

    - Only ndarray or scalar ndarray is accepted as valid input.

    Examples
    --------
    >>> a = np.array([4, 3, 5, 7, 6, 8])
    >>> indices = np.array([0, 1, 4])
    >>> np.take(a, indices)
    array([4., 3., 6.])

    In this example for `a` is an ndarray, "fancy" indexing can be used.

    >>> a[indices]
    array([4., 3., 6.])

    If `indices` is not one dimensional, the output also has these dimensions.

    >>> np.take(a, np.array([[0, 1], [2, 3]]))
    array([[4., 3.],
           [5., 7.]])
    """
    if mode not in ('wrap', 'clip', 'raise'):
        raise NotImplementedError(
            "function take does not support mode '{}'".format(mode))
    if axis is None:
        return _npi.take(_npi.reshape(a, -1), indices, 0, mode, out)
    else:
        return _npi.take(a, indices, axis, mode, out)
# pylint: enable=redefined-outer-name


@set_module('mxnet.ndarray.numpy')
def insert(arr, obj, values, axis=None):
    """
    Insert values along the given axis before the given indices.

    Parameters
    ----------
    arr : ndarray
        Input array.
    obj : int, slice or ndarray of int64
        Object that defines the index or indices before which `values` is
        inserted.
        Support for multiple insertions when `obj` is a single scalar or a
        sequence with one element (only support int32 and int64 element).
    values : ndarray
        Values to insert into `arr`.
        If the type of values is different from that of arr, values is converted
        to the type of arr.
    axis : int, optional
        Axis along which to insert `values`.  If `axis` is None then `arr`
        is flattened first.

    Returns
    -------
    out : ndarray
        A copy of `arr` with `values` inserted.  Note that `insert`
        does not occur in-place: a new array is returned. If
        `axis` is None, `out` is a flattened array.

    Notes
    -----
    - Note that for higher dimensional inserts `obj=0` behaves very different
    from `obj=[0]` just like `arr[:,0,:] = values` is different from
    `arr[:,[0],:] = values`.
    - If obj is a ndarray, it's dtype only supports int64

    Examples
    --------
    >>> a = np.array([[1, 1], [2, 2], [3, 3]])
    >>> a
    array([[1., 1.],
           [2., 2.],
           [3., 3.]])
    >>> np.insert(a, 1, np.array(5))
    array([1., 5., 1., 2., 2., 3., 3.])
    >>> np.insert(a, 1, np.array(5), axis=1)
    array([[1., 5., 1.],
           [2., 5., 2.],
           [3., 5., 3.]])

    Difference between sequence and scalars:

    >>> np.insert(a, np.array([1], dtype=np.int64), np.array([[1],[2],[3]]), axis=1)
    array([[1., 1., 1.],
           [2., 2., 2.],
           [3., 3., 3.]])
    >>> np.insert(a, 1, np.array([1, 2, 3]), axis=1)
    array([[1., 1., 1.],
           [2., 2., 2.],
           [3., 3., 3.]])

    >>> b = a.flatten()
    >>> b
    array([1., 1., 2., 2., 3., 3.])
    >>> np.insert(b, np.array([2, 2], dtype=np.int64), np.array([5, 6]))
    array([1., 1., 5., 6., 2., 2., 3., 3.])

    >>> np.insert(b, slice(2, 4), np.array([5, 6]))
    array([1., 1., 5., 2., 6., 2., 3., 3.])

    # type casting
    >>> np.insert(b.astype(np.int32), np.array([2, 2],dtype='int64'), np.array([7.13, False]))
    array([1, 1, 7, 0, 2, 2, 3, 3], dtype=int32)

    >>> x = np.arange(8).reshape(2, 4)
    >>> idx = np.array([1, 3], dtype=np.int64)
    >>> np.insert(x, idx, np.array([999]), axis=1)
    array([[  0., 999.,   1.,   2., 999.,   3.],
           [  4., 999.,   5.,   6., 999.,   7.]])
    """
    if isinstance(values, numeric_types):
        if isinstance(obj, slice):
            start = obj.start
            stop = obj.stop
            step = 1 if obj.step is None else obj.step
            return _npi.insert_slice(arr, val=values, start=start, stop=stop, step=step, axis=axis)
        elif isinstance(obj, integer_types):
            return _npi.insert_scalar(arr, val=values, int_ind=obj, axis=axis)
        elif isinstance(obj, NDArray):
            return _npi.insert_tensor(arr, obj, val=values, axis=axis)

    if not isinstance(arr, NDArray):
        raise TypeError("'arr' can not support type {}".format(str(type(arr))))
    if not isinstance(values, NDArray):
        raise TypeError("'values' can not support type {}".format(str(type(values))))
    if isinstance(obj, slice):
        start = obj.start
        stop = obj.stop
        step = 1 if obj.step is None else obj.step
        return _npi.insert_slice(arr, values, start=start, stop=stop, step=step, axis=axis)
    elif isinstance(obj, integer_types):
        return _npi.insert_scalar(arr, values, int_ind=obj, axis=axis)
    elif isinstance(obj, NDArray):
        return _npi.insert_tensor(arr, values, obj, axis=axis)
    else:
        raise TypeError("'obj' can not support type {}".format(str(type(obj))))


#pylint: disable= too-many-arguments, no-member, protected-access
def _ufunc_helper(lhs, rhs, fn_array, fn_scalar, lfn_scalar, rfn_scalar=None, out=None):
    """ Helper function for element-wise operation.
    The function will perform numpy-like broadcasting if needed and call different functions.

    Parameters
    --------
    lhs : ndarray or numeric value
        Left-hand side operand.

    rhs : ndarray or numeric value
        Right-hand operand,

    fn_array : function
        Function to be called if both lhs and rhs are of ``ndarray`` type.

    fn_scalar : function
        Function to be called if both lhs and rhs are numeric values.

    lfn_scalar : function
        Function to be called if lhs is ``ndarray`` while rhs is numeric value

    rfn_scalar : function
        Function to be called if lhs is numeric value while rhs is ``ndarray``;
        if none is provided, then the function is commutative, so rfn_scalar is equal to lfn_scalar

    Returns
    --------
    mxnet.numpy.ndarray or scalar
        result array or scalar
    """
    from ...numpy import ndarray
    from ..ndarray import from_numpy  # pylint: disable=unused-import
    if isinstance(lhs, numeric_types):
        if isinstance(rhs, numeric_types):
            return fn_scalar(lhs, rhs, out=out)
        else:
            if rfn_scalar is None:
                # commutative function
                return lfn_scalar(rhs, float(lhs), out=out)
            else:
                return rfn_scalar(rhs, float(lhs), out=out)
    elif isinstance(rhs, numeric_types):
        return lfn_scalar(lhs, float(rhs), out=out)
    elif isinstance(lhs, ndarray) and isinstance(rhs, ndarray):
        return fn_array(lhs, rhs, out=out)
    else:
        raise TypeError('type {} not supported'.format(str(type(rhs))))
#pylint: enable= too-many-arguments, no-member, protected-access


@set_module('mxnet.ndarray.numpy')
def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
    """
    Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Parameters
    ----------
    ar : ndarray
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details. The default is None.

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    Notes
    -----
    When an axis is specified the subarrays indexed by the axis are sorted.
    This is done by making the specified axis the first dimension of the array
    and then flattening the subarrays in C order. The flattened subarrays are
    then viewed as a structured type with each element given a label, with the
    effect that we end up with a 1-D array of structured types that can be
    treated in the same way as any other 1-D array. The result is that the
    flattened subarrays are sorted in lexicographic order starting with the
    first element.

    This function differs from the original `numpy.unique
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html>`_ in
    the following aspects:

    - Only support ndarray as input.
    - Object arrays or structured arrays are not supported.

    Examples
    --------
    >>> np.unique(np.array([1, 1, 2, 2, 3, 3]))
    array([1., 2., 3.])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1., 2., 3.])

    Return the unique rows of a 2D array

    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1., 0., 0.],
           [2., 3., 4.]])

    Return the indices of the original array that give the unique values:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array([1., 2., 3., 4., 6.])
    >>> indices
    array([0, 1, 5, 3, 2], dtype=int64)
    >>> a[indices]
    array([1., 2., 3., 4., 6.])

    Reconstruct the input array from the unique values:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1., 2., 3., 4., 6.])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1], dtype=int64)
    >>> u[indices]
    array([1., 2., 6., 4., 2., 3., 2.])
    """
    ret = _npi.unique(ar, return_index, return_inverse, return_counts, axis)
    if isinstance(ret, list):
        return tuple(ret)
    else:
        return ret


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def add(x1, x2, out=None, **kwargs):
    """
    Add arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays to be added. If x1.shape != x2.shape, they must be broadcastable to
        a common shape (which may be the shape of one or the other).

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    add : ndarray or scalar
        The sum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.

    Notes
    -----
    This operator now supports automatic type promotion. The resulting type will be determined
    according to the following rules:
        * If both inputs are of floating number types, the output is the more precise type.
        * If only one of the inputs is floating number type, the result is that type.
        * If both inputs are of integer types (including boolean), not supported yet.
    """
    return _ufunc_helper(x1, x2, _npi.add, _np.add, _npi.add_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def subtract(x1, x2, out=None, **kwargs):
    """
    Subtract arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays to be subtracted from each other. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which may be the shape
        of one or the other).

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    subtract : ndarray or scalar
        The difference of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.

    Notes
    -----
    This operator now supports automatic type promotion. The resulting type will be determined
    according to the following rules:
        * If both inputs are of floating number types, the output is the more precise type.
        * If only one of the inputs is floating number type, the result is that type.
        * If both inputs are of integer types (including boolean), not supported yet.
    """
    return _ufunc_helper(x1, x2, _npi.subtract, _np.subtract, _npi.subtract_scalar,
                         _npi.rsubtract_scalar, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def multiply(x1, x2, out=None, **kwargs):
    """
    Multiply arguments element-wise.

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays to be multiplied. If x1.shape != x2.shape, they must be broadcastable to
        a common shape (which may be the shape of one or the other).

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        The multiplication of x1 and x2, element-wise. This is a scalar if both x1 and x2
        are scalars.

    Notes
    -----
    This operator now supports automatic type promotion. The resulting type will be determined
    according to the following rules:
        * If both inputs are of floating number types, the output is the more precise type.
        * If only one of the inputs is floating number type, the result is that type.
        * If both inputs are of integer types (including boolean), not supported yet.
    """
    return _ufunc_helper(x1, x2, _npi.multiply, _np.multiply, _npi.multiply_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def divide(x1, x2, out=None, **kwargs):
    """
    Returns a true division of the inputs, element-wise.

    Parameters
    ----------
    x1 : ndarray or scalar
        Dividend array.

    x2 : ndarray or scalar
        Divisor array.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both x1 and x2 are scalars.

    Notes
    -----
    This operator now supports automatic type promotion. The resulting type will be determined
    according to the following rules:
        * If both inputs are of floating number types, the output is the more precise type.
        * If only one of the inputs is floating number type, the result is that type.
        * If both inputs are of integer types (including boolean), the output is of float32 type.
    """
    return _ufunc_helper(x1, x2, _npi.true_divide, _np.divide, _npi.true_divide_scalar,
                         _npi.rtrue_divide_scalar, out)


@set_module('mxnet.ndarray.numpy')
def true_divide(x1, x2, out=None):
    """Returns a true division of the inputs, element-wise.

    Instead of the Python traditional 'floor division', this returns a true
    division.  True division adjusts the output type to present the best
    answer, regardless of input types.

    Parameters
    ----------
    x1 : ndarray or scalar
        Dividend array.

    x2 : ndarray or scalar
        Divisor array.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both x1 and x2 are scalars.

    Notes
    -----
    This operator now supports automatic type promotion. The resulting type will be determined
    according to the following rules:
        * If both inputs are of floating number types, the output is the more precise type.
        * If only one of the inputs is floating number type, the result is that type.
        * If both inputs are of integer types (including boolean), the output is of float32 type.
    """
    return _ufunc_helper(x1, x2, _npi.true_divide, _np.divide, _npi.true_divide_scalar,
                         _npi.rtrue_divide_scalar, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def mod(x1, x2, out=None, **kwargs):
    """
    Return element-wise remainder of division.

    Parameters
    ----------
    x1 : ndarray or scalar
        Dividend array.

    x2 : ndarray or scalar
        Divisor array.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both x1 and x2 are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.mod, _np.mod, _npi.mod_scalar, _npi.rmod_scalar, out)


@set_module('mxnet.ndarray.numpy')
def delete(arr, obj, axis=None):
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by
    `arr[obj]`.

    Parameters
    ----------
    arr : ndarray
      Input array.
    obj : slice, int or ndarray of ints
      Indicate indices of sub-arrays to remove along the specified axis.
    axis : int, optional
      The axis along which to delete the subarray defined by `obj`.
      If `axis` is None, `obj` is applied to the flattened array.

    Returns
    -------
    out : ndarray
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is None, `out` is
        a flattened array.

    Examples
    --------
    >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    >>> arr
    array([[ 1.,  2.,  3.,  4.],
           [ 5.,  6.,  7.,  8.],
           [ 9., 10., 11., 12.]])

    >>> np.delete(arr, 1, 0)
    array([[ 1.,  2.,  3.,  4.],
           [ 9., 10., 11., 12.]])

    >>> np.delete(arr, slice(None, None, 2), 1)
    array([[ 2.,  4.],
           [ 6.,  8.],
           [10., 12.]])

    >>> np.delete(arr, np.array([1,3,5]), None)
    array([ 1.,  3.,  5.,  7.,  8.,  9., 10., 11., 12.])
    >>> np.delete(arr, np.array([1,1,5]), None)
    array([ 1.,  3.,  4.,  5.,  7.,  8.,  9., 10., 11., 12.])
    """
    if not isinstance(arr, NDArray):
        raise TypeError("'arr' can not support type {}".format(str(type(arr))))
    if isinstance(obj, slice):
        start = obj.start
        stop = obj.stop
        step = 1 if obj.step is None else obj.step
        return _npi.delete(arr, start=start, stop=stop, step=step, axis=axis)
    elif isinstance(obj, integer_types):
        return _npi.delete(arr, int_ind=obj, axis=axis)
    elif isinstance(obj, NDArray):
        return _npi.delete(arr, obj, axis=axis)
    else:
        raise TypeError("'obj' can not support type {}".format(str(type(obj))))


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def matmul(a, b, out=None):
    """
    Matrix product of two arrays.

    Parameters
    ----------
    a, b : ndarray
        Input arrays, scalars not allowed.
    out : ndarray, optional
        A location into which the result is stored.
        If provided, it must have a shape that matches the signature (n,k),(k,m)->(n,m).
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray
        The matrix product of the inputs.
        This is a scalar only when both x1, x2 are 1-d vectors.

    Raises
    ------
    MXNetError
        If the last dimension of a is not the same size as the second-to-last dimension of b.
        If a scalar value is passed in.

    See Also
    --------
    tensordot :
        Sum products over arbitrary axes.
    dot :
        alternative matrix product with different broadcasting rules.
    einsum :
        Einstein summation convention.

    Notes
    -----
    The behavior depends on the arguments in the following way.

    - If both arguments are 2-D they are multiplied like conventional matrices.
    - If either argument is N-D, N > 2, it is treated as a stack of matrices
      residing in the last two indexes and broadcast accordingly.
    - If the first argument is 1-D, it is promoted to a matrix by prepending
      a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
    - If the second argument is 1-D, it is promoted to a matrix by appending a 1
      to its dimensions. After matrix multiplication the appended 1 is removed.

    matmul differs from dot in two important ways:

    - Multiplication by scalars is not allowed, use multiply instead.
    - Stacks of matrices are broadcast together as if the matrices were elements,
    respecting the signature (n,k),(k,m)->(n,m):
    >>> a = np.ones([9, 5, 7, 4])
    >>> c = np.ones([9, 5, 4, 3])
    >>> np.dot(a, c).shape
    (9, 5, 7, 9, 5, 3)
    >>> np.matmul(a, c).shape
    (9, 5, 7, 3)
    >>> # n is 7, k is 4, m is 3

    Examples
    --------
    For 2-D arrays it is the matrix product:
    >>> a = np.array([[1, 0],
    ...               [0, 1]])
    >>> b = np.array([[4, 1],
    ...               [2, 2]])
    >>> np.matmul(a, b)
    array([[4., 1.],
           [2., 2.]])

    For 2-D mixed with 1-D, the result is the usual.
    >>> a = np.array([[1, 0],
    ...               [0, 1]])
    >>> b = np.array([1, 2])
    >>> np.matmul(a, b)
    array([1., 2.])
    >>> np.matmul(b, a)
    array([1., 2.])

    Broadcasting is conventional for stacks of arrays
    >>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
    >>> b = np.arange(2 * 2 * 4).reshape((2, 4, 2))
    >>> np.matmul(a, b).shape
    (2, 2, 2)
    >>> np.matmul(a, b)[0, 1, 1]
    array(98.)
    >>> sum(a[0, 1, :] * b[0, :, 1])
    array(98.)

    Scalar multiplication raises an error.
    >>> np.matmul([1, 2], 3)
    Traceback (most recent call last):
    ...
    mxnet.base.MXNetError: ... : Multiplication by scalars is not allowed.
    """
    return _npi.matmul(a, b, out=out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def remainder(x1, x2, out=None):
    """
    Return element-wise remainder of division.

    Parameters
    ----------
    x1 : ndarray or scalar
        Dividend array.

    x2 : ndarray or scalar
        Divisor array.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both x1 and x2 are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.mod, _np.mod, _npi.mod_scalar, _npi.rmod_scalar, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def power(x1, x2, out=None, **kwargs):
    """
    First array elements raised to powers from second array, element-wise.

    Parameters
    ----------
    x1 : ndarray or scalar
        The bases.

    x2 : ndarray or scalar
        The exponent.

    out : ndarray
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    out : ndarray or scalar
        The bases in x1 raised to the exponents in x2.
        This is a scalar if both x1 and x2 are scalars.
    """
    return _ufunc_helper(x1, x2, _npi.power, _np.power, _npi.power_scalar, _npi.rpower_scalar, out)


@set_module('mxnet.ndarray.numpy')
def argsort(a, axis=-1, kind=None, order=None):
    """
    Returns the indices that would sort an array.
    Perform an indirect sort along the given axis using the algorithm specified
    by the `kind` keyword. It returns an array of indices of the same shape as
    `a` that index data along the given axis in sorted order.

    Parameters
    ----------
    a : ndarray
        Array to sort.
    axis : int or None, optional
        Axis along which to sort.  The default is -1 (the last axis). If None,
        the flattened array is used.
    kind : string, optional
        This argument can take any string, but it does not have any effect on the
        final result.
    order : str or list of str, optional
        Not supported yet, will raise NotImplementedError if not None.

    Returns
    -------
    index_array : ndarray, int
        Array of indices that sort `a` along the specified `axis`.
        If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.
        More generally, ``np.take_along_axis(a, index_array, axis=axis)``
        always yields the sorted `a`, irrespective of dimensionality.

    Notes
    -----
    This operator does not support different sorting algorithms.

    Examples
    --------
    One dimensional array:

    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])

    Two-dimensional array:

    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])
    >>> ind = np.argsort(x, axis=0)  # sorts along first axis (down)
    >>> ind
    array([[0, 1],
           [1, 0]])
    >>> np.take_along_axis(x, ind, axis=0)  # same as np.sort(x, axis=0)
    array([[0, 2],
           [2, 3]])
    >>> ind = np.argsort(x, axis=1)  # sorts along last axis (across)
    >>> ind
    array([[0, 1],
           [0, 1]])
    >>> np.take_along_axis(x, ind, axis=1)  # same as np.sort(x, axis=1)
    array([[0, 3],
           [2, 2]])

    Indices of the sorted elements of a N-dimensional array:

    >>> ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
    >>> ind
    (array([0, 1, 1, 0]), array([0, 0, 1, 1]))
    >>> x[ind]  # same as np.sort(x, axis=None)
    array([0, 2, 2, 3])
    """
    if order is not None:
        raise NotImplementedError("order not supported here")

    return _npi.argsort(data=a, axis=axis, is_ascend=True, dtype='int64')


@set_module('mxnet.ndarray.numpy')
def sort(a, axis=-1, kind=None, order=None):
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    a : ndarray
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort.  The default is -1 (the last axis). If None,
        the flattened array is used.
    kind : string, optional
        This argument can take any string, but it does not have any effect on the
        final result.
    order : str or list of str, optional
        Not supported yet, will raise NotImplementedError if not None.

    Returns
    -------
    sorted_array : ndarray
        Array of the same type and shape as `a`.

    Notes
    -----
    This operator does not support different sorting algorithms.

    Examples
    --------
    >>> a = np.array([[1,4],[3,1]])
    >>> np.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> np.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> np.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])
    """
    if order is not None:
        raise NotImplementedError("order not supported here")
    return _npi.sort(data=a, axis=axis, is_ascend=True)


@set_module('mxnet.ndarray.numpy')
def tensordot(a, b, axes=2):
    r"""
    tensordot(a, b, axes=2)
    Compute tensor dot product along specified axes for arrays >= 1-D.
    Given two tensors (arrays of dimension greater than or equal to one),
    `a` and `b`, and an ndarray object containing two ndarray
    objects, ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s
    elements (components) over the axes specified by ``a_axes`` and
    ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N``
    dimensions of `a` and the first ``N`` dimensions of `b` are summed
    over.
    Parameters
    ----------
    a, b : ndarray, len(shape) >= 1
        Tensors to "dot".
    axes : int or (2,) ndarray
        * integer_like
        If an int N, sum over the last N axes of `a` and the first N axes
        of `b` in order. The sizes of the corresponding axes must match.
        * (2,) ndarray
        Or, a list of axes to be summed over, first sequence applying to `a`,
        second to `b`. Both elements ndarray must be of the same length.
    See Also
    --------
    dot, einsum
    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a\otimes b`
        * ``axes = 1`` : tensor dot product :math:`a\cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`
    When `axes` is integer_like, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.
    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.
    Examples
    --------
    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> c = np.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    """
    return _api_internal.tensordot(a, b, axes)


@set_module('mxnet.ndarray.numpy')
def histogram(a, bins=10, range=None, normed=None, weights=None, density=None):  # pylint: disable=too-many-arguments
    """
    Compute the histogram of a set of data.

    Parameters
    ----------
    a : ndarray
        Input data. The histogram is computed over the flattened array.
    bins : int or NDArray
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.
        .. versionadded:: 1.11.0
        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.
    range : (float, float)
        The lower and upper range of the bins. Required when `bins` is an integer.
        Values outside the range are ignored. The first element of the range must
        be less than or equal to the second.
    normed : bool, optional
        Not supported yet, coming soon.
    weights : array_like, optional
        Not supported yet, coming soon.
    density : bool, optional
        Not supported yet, coming soon.
    """
    if normed is True:
        raise NotImplementedError("normed is not supported yet...")
    if weights is not None:
        raise NotImplementedError("weights is not supported yet...")
    if density is True:
        raise NotImplementedError("density is not supported yet...")
    if isinstance(bins, numeric_types):
        if range is None:
            raise NotImplementedError("automatic range is not supported yet...")
        return _npi.histogram(a, bin_cnt=bins, range=range)
    if isinstance(bins, (list, tuple)):
        raise NotImplementedError("array_like bins is not supported yet...")
    if isinstance(bins, str):
        raise NotImplementedError("string bins is not supported yet...")
    if isinstance(bins, NDArray):
        return _npi.histogram(a, bins=bins)
    raise ValueError("np.histogram fails with", locals())


@set_module('mxnet.ndarray.numpy')
def eye(N, M=None, k=0, dtype=_np.float32, **kwargs):
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output. If None, defaults to N.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal,
        and a negative value to a lower diagonal.
    dtype : data-type, optional
        Data-type of the returned array.

    Returns
    -------
    I : ndarray of shape (N,M)
        An array where all elements are equal to zero,
        except for the k-th diagonal, whose values are equal to one.
    """
    _sanity_check_params('eye', ['order'], kwargs)
    ctx = kwargs.pop('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    return _npi.eye(N, M, k, ctx, dtype)


@set_module('mxnet.ndarray.numpy')
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, ctx=None):  # pylint: disable=too-many-arguments
    r"""
    Return evenly spaced numbers over a specified interval.
    Returns num evenly spaced samples, calculated over the interval [start, stop].
    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : real number
        The starting value of the sequence.
    stop : real number
        The end value of the sequence, unless endpoint is set to False. In
        that case, the sequence consists of all but the last of num + 1
        evenly spaced samples, so that stop is excluded. Note that the step
        size changes when endpoint is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, stop is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (samples, step), where step is the spacing between samples.
    dtype : dtype, optional
        The type of the output array. If dtype is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start or
        stop are array-like. By default (0), the samples will be along a new
        axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        There are num equally spaced samples in the closed interval
        `[start, stop]` or the half-open interval `[start, stop)`
        (depending on whether endpoint is True or False).
    step : float, optional
        Only returned if retstep is True
        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).

    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    array([2. , 2.2, 2.4, 2.6, 2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    (array([2.  , 2.25, 2.5 , 2.75, 3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1.asnumpy(), y.asnumpy(), 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2.asnumpy(), (y + 0.5).asnumpy(), 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    Notes
    -----

    This function differs from the original `numpy.linspace
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html>`_ in
    the following aspects:

    - `start` and `stop` do not support list, numpy ndarray and mxnet ndarray
    - axis could only be 0
    - There could be an additional `ctx` argument to specify the device, e.g. the i-th
      GPU.
    """
    if isinstance(start, (list, _np.ndarray, NDArray)) or \
       isinstance(stop, (list, _np.ndarray, NDArray)):
        raise NotImplementedError('start and stop only support int')
    if axis != 0:
        raise NotImplementedError("the function only support axis 0")
    if ctx is None:
        ctx = current_context()
    if retstep:
        step = (stop - start) / (num - 1)
        return _npi.linspace(start=start, stop=stop, num=num, endpoint=endpoint, ctx=ctx, dtype=dtype), step
    else:
        return _npi.linspace(start=start, stop=stop, num=num, endpoint=endpoint, ctx=ctx, dtype=dtype)


@set_module('mxnet.ndarray.numpy')
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0, ctx=None):  # pylint: disable=too-many-arguments
    r"""Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

        Non-scalar `start` and `stop` are now supported.

    Parameters
    ----------
    start : int or float
        ``base ** start`` is the starting value of the sequence.
    stop : int or float
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : float, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Now, axis only support axis = 0.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    arange : Similar to linspace, with the step size specified instead of the
             number of samples. Note that, when used with a float endpoint, the
             endpoint may or may not be included.
    linspace : Similar to logspace, but with the samples uniformly distributed
               in linear space, instead of log space.

    Notes
    -----
    Logspace is equivalent to the code. Now wo only support axis = 0.

    >>> y = np.linspace(start, stop, num=num, endpoint=endpoint)
    ...
    >>> power(base, y).astype(dtype)
    ...

    Examples
    --------
    >>> np.logspace(2.0, 3.0, num=4)
    array([ 100.     ,  215.44347,  464.15887, 1000.     ])
    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)
    array([100.     , 177.82794, 316.22775, 562.3413 ])
    >>> np.logspace(2.0, 3.0, num=4, base=2.0)
    array([4.       , 5.0396843, 6.349604 , 8.       ])
    >>> np.logspace(2.0, 3.0, num=4, base=2.0, dtype=np.int32)
    array([4, 5, 6, 8], dtype=int32)
    >>> np.logspace(2.0, 3.0, num=4, ctx=npx.gpu(0))
    array([ 100.     ,  215.44347,  464.15887, 1000.     ], ctx=gpu(0))
    """
    if isinstance(start, (list, tuple, _np.ndarray, NDArray)) or \
       isinstance(stop, (list, tuple, _np.ndarray, NDArray)):
        raise NotImplementedError('start and stop only support int and float')
    if axis != 0:
        raise NotImplementedError("the function only support axis 0")
    if ctx is None:
        ctx = current_context()
    return _npi.logspace(start=start, stop=stop, num=num, endpoint=endpoint, base=base, ctx=ctx, dtype=dtype)


@set_module('mxnet.ndarray.numpy')
def expand_dims(a, axis):
    """Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : ndarray
        Output array. The number of dimensions is one greater than that of
        the input array.
    """
    return _npi.expand_dims(a, axis)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def lcm(x1, x2, out=None, **kwargs):
    """
    Returns the lowest common multiple of ``|x1|`` and ``|x2|``

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays for computing lowest common multiple. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which may be the shape of
        one or the other).

    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    y : ndarray or scalar
        The lowest common multiple of the absolute value of the inputs
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    gcd : The greatest common divisor

    Examples
    --------
    >>> np.lcm(12, 20)
    60
    >>> np.lcm(np.arange(6, dtype=int), 20)
    array([ 0, 20, 20, 60, 20, 20], dtype=int64)
    """
    return _ufunc_helper(x1, x2, _npi.lcm, _np.lcm, _npi.lcm_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def tril(m, k=0):
    r"""
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : ndarray, shape (M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril : ndarray, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : same thing, only for the upper triangle

    Examples
    --------
    >>> a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    >>> np.tril(a, -1)
    array([[ 0.,  0.,  0.],
           [ 4.,  0.,  0.],
           [ 7.,  8.,  0.],
           [10., 11., 12.]])
    """
    return _npi.tril(m, k)


def _unary_func_helper(x, fn_array, fn_scalar, out=None, **kwargs):
    """Helper function for unary operators.

    Parameters
    ----------
    x : ndarray or scalar
        Input of the unary operator.
    fn_array : function
        Function to be called if x is of ``ndarray`` type.
    fn_scalar : function
        Function to be called if x is a Python scalar.
    out : ndarray
        The buffer ndarray for storing the result of the unary function.

    Returns
    -------
    out : mxnet.numpy.ndarray or scalar
        Result array or scalar.
    """
    if isinstance(x, numeric_types):
        return fn_scalar(x, **kwargs)
    elif isinstance(x, NDArray):
        return fn_array(x, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def sin(x, out=None, **kwargs):
    r"""
    Trigonometric sine, element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.

    Returns
    -------
    y : ndarray or scalar
        The sine of each element of x. This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.

    Examples
    --------
    >>> np.sin(np.pi/2.)
    1.0
    >>> np.sin(np.array((0., 30., 45., 60., 90.)) * np.pi / 180.)
    array([0.        , 0.5       , 0.70710677, 0.86602545, 1.        ])
    """
    return _unary_func_helper(x, _npi.sin, _np.sin, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def cos(x, out=None, **kwargs):
    r"""
    Cosine, element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.

    Returns
    -------
    y : ndarray or scalar
        The corresponding cosine values. This is a scalar if x is a scalar.

    Notes
    ----
    This function only supports input type of float.

    Examples
    --------
    >>> np.cos(np.array([0, np.pi/2, np.pi]))
    array([ 1.000000e+00, -4.371139e-08, -1.000000e+00])
    >>> # Example of providing the optional output parameter
    >>> out1 = np.array([0], dtype='f')
    >>> out2 = np.cos(np.array([0.1]), out1)
    >>> out2 is out1
    True
    """
    return _unary_func_helper(x, _npi.cos, _np.cos, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def sinh(x, out=None, **kwargs):
    """
    Hyperbolic sine, element-wise.
    Equivalent to ``1/2 * (np.exp(x) - np.exp(-x))`` or ``-1j * np.sin(1j*x)``.

    Parameters
    ----------
    x : ndarray or scalar
        Input array or scalar.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.

    Returns
    -------
    y : ndarray or scalar
        The corresponding hyperbolic sine values. This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.

    Examples
    --------
    >>> np.sinh(0)
    0.0
    >>> # Example of providing the optional output parameter
    >>> out1 = np.array([0], dtype='f')
    >>> out2 = np.sinh(np.array([0.1]), out1)
    >>> out2 is out1
    True
    """
    return _unary_func_helper(x, _npi.sinh, _np.sinh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def cosh(x, out=None, **kwargs):
    """
    Hyperbolic cosine, element-wise.
    Equivalent to ``1/2 * (np.exp(x) + np.exp(-x))`` and ``np.cos(1j*x)``.

    Parameters
    ----------
    x : ndarray or scalar
        Input array or scalar.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.

    Returns
    -------
    y : ndarray or scalar
        The corresponding hyperbolic cosine values. This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.

    Examples
    --------
    >>> np.cosh(0)
    1.0
    """
    return _unary_func_helper(x, _npi.cosh, _np.cosh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def tanh(x, out=None, **kwargs):
    """
    Compute hyperbolic tangent element-wise.
    Equivalent to ``np.sinh(x)/np.cosh(x)``.

    Parameters
    ----------
    x : ndarray or scalar.
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.

    Returns
    -------
    y : ndarray or scalar
       The corresponding hyperbolic tangent values.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)
    - input x does not support complex computation (like imaginary number)
    >>> np.tanh(np.pi*1j)
    TypeError: type <type 'complex'> not supported

    Examples
    --------
    >>> np.tanh(np.array[0, np.pi]))
    array([0.       , 0.9962721])
    >>> np.tanh(np.pi)
    0.99627207622075
    >>> # Example of providing the optional output parameter illustrating
    >>> # that what is returned is a reference to said parameter
    >>> out1 = np.array(1)
    >>> out2 = np.tanh(np.array(0.1), out1)
    >>> out2 is out1
    True
    """
    return _unary_func_helper(x, _npi.tanh, _np.tanh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def log10(x, out=None, **kwargs):
    """
    Return the base 10 logarithm of the input array, element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        Input array or scalar.
    out : ndarray or None
        A location into which t'absolute', he result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output is the same as that of the input if the input is an ndarray.

    Returns
    -------
    y : ndarray or scalar
        The logarithm to the base 10 of `x`, element-wise. NaNs are
        returned where x is negative. This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.

    Examples
    --------
    >>> np.log10(np.array([1e-15, -3.]))
    array([-15.,  nan])
    """
    return _unary_func_helper(x, _npi.log10, _np.log10, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def sqrt(x, out=None, **kwargs):
    """
    Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        The values whose square-roots are required.
    out : ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`. This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.

    Examples
    --------
    >>> np.sqrt(np.array([1,4,9]))
    array([1., 2., 3.])
    >>> np.sqrt(np.array([4, -1, _np.inf]))
    array([ 2., nan, inf])
    """
    return _unary_func_helper(x, _npi.sqrt, _np.sqrt, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def cbrt(x, out=None, **kwargs):
    r"""
    Return the cube-root of an array, element-wise.

    Parameters
    ----------
    x : ndarray
        The values whose cube-roots are required.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that the
        inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        A tuple (possible only as a keyword argument) must have length equal to the number of outputs.

    Returns
    ----------
    y : ndarray
        An array of the same shape as x, containing the cube cube-root of each element in x.
        If out was provided, y is a reference to it. This is a scalar if x is a scalar.

    Examples
    ----------
    >>> np.cbrt([1,8,27])
    array([ 1.,  2.,  3.])
    """
    return _unary_func_helper(x, _npi.cbrt, _np.cbrt, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def abs(x, out=None, **kwargs):
    r"""
    Calculate the absolute value element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    absolute : ndarray
        An ndarray containing the absolute value of
        each element in `x`. This is a scalar if `x` is a scalar.

    Examples
    --------
    >>> x = np.array([-1.2, 1.2])
    >>> np.abs(x)
    array([1.2, 1.2])
    """
    return _unary_func_helper(x, _npi.abs, _np.abs, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def fabs(x, out=None, **kwargs):
    r"""
    Calculate the absolute value element-wise.

    This function returns the absolute values (positive magnitude) of the
    data in `x`. Complex values are not handled, use `absolute` to find the
    absolute values of complex data.

    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    absolute : ndarray
        An ndarray containing the absolute value of
        each element in `x`. This is a scalar if `x` is a scalar.

    Examples
    --------
    >>> np.fabs(-1)
    1.0
    >>> np.fabs(np.array([-1.2, 1.2]))s
    array([ 1.2,  1.2])
    """
    return _unary_func_helper(x, _npi.abs, _np.abs, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def absolute(x, out=None, **kwargs):
    r"""
    Calculate the absolute value element-wise.
    np.abs is a shorthand for this function.

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        A tuple (possible only as a keyword argument) must have length equal to the number of outputs.

    Returns
    ----------
    absolute : ndarray
        An ndarray containing the absolute value of each element in x.

    Examples
    ----------
    >>> x = np.array([-1.2, 1.2])
    >>> np.absolute(x)
    array([ 1.2,  1.2])
    """
    return _unary_func_helper(x, _npi.absolute, _np.absolute, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def sign(x, out=None, **kwargs):
    r"""
    Returns an element-wise indication of the sign of a number.
    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``. Only supports real number.

    Parameters
    ----------
    x : ndarray or a scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray
        The sign of `x`.
        This is a scalar if `x` is a scalar.

    Note
    -------
    - Only supports real number as input elements.
    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    Examples
    --------
    >>> a = np.array([-5., 4.5])
    >>> np.sign(a)
    array([-1.,  1.])
    >>> # Use scalars as inputs:
    >>> np.sign(4.0)
    1.0
    >>> np.sign(0)
    0
    >>> # Use ``out`` parameter:
    >>> b = np.zeros((2, ))
    >>> np.sign(a, out=b)
    array([-1.,  1.])
    >>> b
    array([-1.,  1.])
    """
    return _unary_func_helper(x, _npi.sign, _np.sign, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def exp(x, out=None, **kwargs):
    r"""
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise exponential of `x`.
        This is a scalar if `x` is a scalar.

    Examples
    --------
    >>> np.exp(1)
    2.718281828459045
    >>> x = np.array([-1, 1, -2, 2])
    >>> np.exp(x)
    array([0.36787945, 2.7182817 , 0.13533528, 7.389056  ])
    """
    return _unary_func_helper(x, _npi.exp, _np.exp, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def expm1(x, out=None, **kwargs):
    r"""
    Calculate `exp(x) - 1` of all elements in the input array.

    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise exponential minus one: `out = exp(x) - 1`.
        This is a scalar if `x` is a scalar.

    Examples
    --------
    >>> np.expm1(1)
    1.718281828459045
    >>> x = np.array([-1, 1, -2, 2])
    >>> np.expm1(x)
    array([-0.63212056,  1.71828183, -0.86466472,  6.3890561])
    """
    return _unary_func_helper(x, _npi.expm1, _np.expm1, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def arcsin(x, out=None, **kwargs):
    r"""
    Inverse sine, element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        `y`-coordinate on the unit circle.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape as the input.
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    angle : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.
        The inverse sine of each element in `x`, in radians and in the
        closed interval ``[-pi/2, pi/2]``.

    Examples
    --------
    >>> np.arcsin(1)     # pi/2
    1.5707963267948966
    >>> np.arcsin(-1)    # -pi/2
    -1.5707963267948966
    >>> np.arcsin(0)
    0.0

    Notes
    -----
    `arcsin` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that :math:`sin(z) = x`.  The convention is to
    return the angle `z` whose real part lies in [-pi/2, pi/2].
    For real-valued input data types, *arcsin* always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    The inverse sine is also known as `asin` or sin^{-1}.
    The output `ndarray` has the same `ctx` as the input `ndarray`.
    This function differs from the original `numpy.arcsin
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsin.html>`_ in
    the following aspects:
    - Only support ndarray or scalar now.
    - `where` argument is not supported.
    - Complex input is not supported.

    References
    ----------
    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79ff.
    http://www.math.sfu.ca/~cbm/aands/
    """
    return _unary_func_helper(x, _npi.arcsin, _np.arcsin, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def arccos(x, out=None, **kwargs):
    r"""
    Trigonometric inverse cosine, element-wise.
    The inverse of cos so that, if y = cos(x), then x = arccos(y).

    Parameters
    ----------
    x : ndarray
        x-coordinate on the unit circle. For real arguments, the domain is [-1, 1].
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that
        the inputs broadcast to. If not provided or None, a freshly-allocated array is returned.
        A tuple (possible only as a keyword argument) must have length equal to the number of outputs.

    Returns
    ----------
    angle : ndarray
        The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi].
        This is a scalar if x is a scalar.

    See also
    ----------
    cos, arctan, arcsin

    Notes
    ----------
    arccos is a multivalued function: for each x there are infinitely many numbers z such that
    cos(z) = x. The convention is to return the angle z whose real part lies in [0, pi].
    For real-valued input data types, arccos always returns real output.
    For each value that cannot be expressed as a real number or infinity, it yields nan and sets
    the invalid floating point error flag.
    The inverse cos is also known as acos or cos^-1.

    Examples
    ----------
    >>> np.arccos([1, -1])
    array([ 0.        ,  3.14159265])
    """
    return _unary_func_helper(x, _npi.arccos, _np.arccos, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def arctan(x, out=None, **kwargs):
    r"""
    Trigonometric inverse tangent, element-wise.
    The inverse of tan, so that if ``y = tan(x)`` then ``x = arctan(y)``.

    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray or scalar
        Out has the same shape as `x`. It lies is in
        ``[-pi/2, pi/2]`` (``arctan(+/-inf)`` returns ``+/-pi/2``).
        This is a scalar if `x` is a scalar.

    Notes
    -----
    `arctan` is a multi-valued function: for each `x` there are infinitely
    many numbers `z` such that tan(`z`) = `x`.  The convention is to return
    the angle `z` whose real part lies in [-pi/2, pi/2].
    For real-valued input data types, `arctan` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    For complex-valued input, we do not have support for them yet.
    The inverse tangent is also known as `atan` or tan^{-1}.

    Examples
    --------
    >>> x = np.array([0, 1])
    >>> np.arctan(x)
    array([0.       , 0.7853982])
    >>> np.pi/4
    0.7853981633974483
    """
    return _unary_func_helper(x, _npi.arctan, _np.arctan, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def log(x, out=None, **kwargs):
    """
    Natural logarithm, element-wise.
    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base
    `e`.

    Parameters
    ----------
    x : ndarray
        Input value. Elements must be of real value.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray
        The natural logarithm of `x`, element-wise.
        This is a scalar if `x` is a scalar.

    Notes
    -----
     Currently only supports data of real values and ``inf`` as input. Returns data of real value, ``inf``, ``-inf`` and
    ``nan`` according to the input.
    This function differs from the original `numpy.log
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html>`_ in
    the following aspects:
    - Does not support complex number for now
    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    Examples
    --------
    >>> a = np.array([1, np.exp(1), np.exp(2), 0], dtype=np.float64)
    >>> np.log(a)
    array([  0.,   1.,   2., -inf], dtype=float64)
    >>> # Using default float32 dtype may lead to slightly different behavior:
    >>> a = np.array([1, np.exp(1), np.exp(2), 0], dtype=np.float32)
    >>> np.log(a)
    array([  0.,  0.99999994,   2., -inf])
    >>> np.log(1)
    0.0
    """
    return _unary_func_helper(x, _npi.log, _np.log, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def degrees(x, out=None, **kwargs):
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : ndarray
        Input value. Elements must be of real value.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray
        The corresponding degree values; if `out` was supplied this is a
        reference to it.
        This is a scalar if `x` is a scalar.

    Notes
    -------
    This function differs from the original `numpy.degrees
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.degrees.html>`_ in
    the following aspects:
    - Input type does not support Python native iterables(list, tuple, ...). Only ndarray is supported.
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    Examples
    --------
    >>> rad = np.arange(12.) * np.pi / 6
    >>> np.degrees(rad)
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    >>> # Use specified ``out`` ndarray:
    >>> out = np.zeros((rad.shape))
    >>> np.degrees(rad, out)
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    >>> out
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    """
    return _unary_func_helper(x, _npi.degrees, _np.degrees, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def rad2deg(x, out=None, **kwargs):
    r"""
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : ndarray or scalar
        Angles in degrees.
    out : ndarray or None, optional
        A location into which the result is stored. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        The corresponding angle in radians.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    "rad2deg(x)" is "x *180 / pi".

    This function differs from the original numpy.arange in the following aspects:
        - Only support float32 and float64.
        - `out` must be in the same size of input.

    Examples
    --------
    >>> np.rad2deg(np.pi/2)
    90.0
    """
    return _unary_func_helper(x, _npi.rad2deg, _np.rad2deg, out=out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def rint(x, out=None, **kwargs):
    """
    Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.

    Notes
    -----
    This function differs from the original `numpy.rint
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.rint.html>`_ in
    the following way(s):
    - only ndarray or scalar is accpted as valid input, tuple of ndarray is not supported
    - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.rint(a)
    array([-2., -2., -0.,  0.,  1.,  2.,  2.])
    """
    return _unary_func_helper(x, _npi.rint, _np.rint, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def log2(x, out=None, **kwargs):
    """
    Base-2 logarithm of x.

    Parameters
    ----------
    x : ndarray or scalar
        Input values.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray
        The logarithm base two of `x`, element-wise.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    This function differs from the original `numpy.log2
    <https://www.google.com/search?q=numpy+log2>`_ in
    the following way(s):
    - only ndarray or scalar is accpted as valid input, tuple of ndarray is not supported
    - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param

    Examples
    --------
    >>> x = np.array([0, 1, 2, 2**4])
    >>> np.log2(x)
    array([-inf,   0.,   1.,   4.])
    """
    return _unary_func_helper(x, _npi.log2, _np.log2, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def log1p(x, out=None, **kwargs):
    """
    Return the natural logarithm of one plus the input array, element-wise.
    Calculates ``log(1 + x)``.

    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.

    Returns
    -------
    y : ndarray or scalar
        Natural logarithm of 1 + x, element-wise. This is a scalar
        if x is a scalar.

    Notes
    -----
    For real-valued input, `log1p` is accurate also for `x` so small
    that `1 + x == 1` in floating-point accuracy.
    Logarithm is a multivalued function: for each `x` there is an infinite
    number of `z` such that `exp(z) = 1 + x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi, pi]`.
    For real-valued input data types, `log1p` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.
    cannot support complex-valued input.

    Examples
    --------
    >>> np.log1p(1e-99)
    1e-99
    >>> a = np.array([3, 4, 5])
    >>> np.log1p(a)
    array([1.3862944, 1.609438 , 1.7917595])
    """
    return _unary_func_helper(x, _npi.log1p, _np.log1p, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def radians(x, out=None, **kwargs):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : ndarray or scalar
        Input array in degrees.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray
        The corresponding radian values. This is a scalar if x is a scalar.

    Notes
    -----
    This function differs from the original `numpy.radians
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.radians.html>`_ in
    the following way(s):
    - only ndarray or scalar is accpted as valid input, tuple of ndarray is not supported
    - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param

    Examples
    --------
    >>> deg = np.arange(12.) * 30.
    >>> np.radians(deg)
    array([0.       , 0.5235988, 1.0471976, 1.5707964, 2.0943952, 2.6179938,
           3.1415927, 3.6651914, 4.1887903, 4.712389 , 5.2359877, 5.7595863],
           dtype=float32)
    """
    return _unary_func_helper(x, _npi.radians, _np.radians, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def deg2rad(x, out=None, **kwargs):
    r"""
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : ndarray or scalar
        Angles in degrees.
    out : ndarray or None, optional
        A location into which the result is stored. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        The corresponding angle in radians.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    "deg2rad(x)" is "x * pi / 180".

    This function differs from the original numpy.arange in the following aspects:
        - Only support float32 and float64.
        - `out` must be in the same size of input.

    Examples
    --------
    >>> np.deg2rad(180)
    3.1415927
    """
    return _unary_func_helper(x, _npi.deg2rad, _np.deg2rad, out=out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def reciprocal(x, out=None, **kwargs):
    r"""
    Return the reciprocal of the argument, element-wise.
    Calculates ``1/x``.

    Parameters
    ----------
    x : ndarray or scalar
        The values whose reciprocals are required.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape as the input.
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.

    Examples
    --------
    >>> np.reciprocal(2.)
    0.5
    >>> x = np.array([1, 2., 3.33])
    >>> np.reciprocal(x)
    array([1.       , 0.5      , 0.3003003])

    Notes
    -----
    .. note::
        This function is not designed to work with integers.
    For integer arguments with absolute value larger than 1 the result is
    always zero because of the way Python handles integer division.  For
    integer zero the result is an overflow.
    The output `ndarray` has the same `ctx` as the input `ndarray`.
    This function differs from the original `numpy.reciprocal
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.reciprocal.html>`_ in
    the following aspects:
    - Only support ndarray and scalar now.
    - `where` argument is not supported.
    """
    return _unary_func_helper(x, _npi.reciprocal, _np.reciprocal, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def square(x, out=None, **kwargs):
    r"""
    Return the element-wise square of the input.

    Parameters
    ----------
    x : ndarray or scalar
        The values whose squares are required.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape as the input.
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.

    Examples
    --------
    >>> np.square(2.)
    4.0
    >>> x = np.array([1, 2., -1])
    >>> np.square(x)
    array([1., 4., 1.])

    Notes
    -----
    The output `ndarray` has the same `ctx` as the input `ndarray`.
    This function differs from the original `numpy.square
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html>`_ in
    the following aspects:
    - Only support ndarray and scalar now.
    - `where` argument is not supported.
    - Complex input is not supported.
    """
    return _unary_func_helper(x, _npi.square, _np.square, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def negative(x, out=None, **kwargs):
    r"""
    Numerical negative, element-wise.

    Parameters:
    ------------
    x : ndarray or scalar
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
          A location into which the result is stored.

    Returns:
    ---------
    y : ndarray or scalar
        Returned array or scalar: y = -x. This is a scalar if x is a scalar.

    Examples:
    ---------
    >>> np.negative(1)
    -1
    """
    return _unary_func_helper(x, _npi.negative, _np.negative, out=out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def fix(x, out=None, **kwargs):
    r"""
    Round an array of floats element-wise to nearest integer towards zero.
    The rounded values are returned as floats.

    Parameters:
    ----------
    x : ndarray
        An array of floats to be rounded
    out : ndarray, optional
        Output array

    Returns:
    -------
    y : ndarray of floats

    Examples
    ---------
    >>> np.fix(3.14)
    3
    """
    return _unary_func_helper(x, _npi.fix, _np.fix, out=out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def tan(x, out=None, **kwargs):
    r"""
    Compute tangent element-wise.
    Equivalent to np.sin(x)/np.cos(x) element-wise.

    Parameters:
    ----------
    x : ndarray
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
          A location into which the result is stored. If provided,
          it must have a shape that the inputs broadcast to. If not provided or None,
          a freshly-allocated array is returned. A tuple (possible only as a keyword argument)
          must have length equal to the number of outputs.
    where : ndarray, optional
            Values of True indicate to calculate the ufunc at that position,
            values of False indicate to leave the value in the output alone.

    Returns:
    -------
    y : ndarray
    The corresponding tangent values. This is a scalar if x is a scalar.

    Examples:
    ---------
    >>> np.tan(0.5)
    0.5463024898437905
    """

    return _unary_func_helper(x, _npi.tan, _np.tan, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def ceil(x, out=None, **kwargs):
    r"""
    Return the ceiling of the input, element-wise.
    The ceil of the ndarray `x` is the smallest integer `i`, such that
    `i >= x`.  It is often denoted as :math:`\lceil x \rceil`.

    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a same shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.

    Returns
    -------
    y : ndarray or scalar
        The ceiling of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.ceil(a)
    array([-1., -1., -0.,  1.,  2.,  2.,  2.])
    >>> #if you use parameter out, x and out must be ndarray.
    >>> a = np.array(1)
    >>> np.ceil(np.array(3.5), a)
    array(4.)
    >>> a
    array(4.)
    """
    return _unary_func_helper(x, _npi.ceil, _np.ceil, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def floor(x, out=None, **kwargs):
    r"""
    Return the floor of the input, element-wise.
    The floor of the ndarray `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.

    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None
        A location into which the result is stored. If provided, it
        must have a same shape that the inputs fill into. If not provided
        or None, a freshly-allocated array is returned. The dtype of the
        output and input must be the same.

    Returns
    -------
    y : ndarray or scalar
        The floor of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.floor(a)
    array([-2., -2., -1.,  0.,  1.,  1.,  2.])
    >>> #if you use parameter out, x and out must be ndarray.
    >>> a = np.array(1)
    >>> np.floor(np.array(3.5), a)
    array(3.)
    >>> a
    array(3.)
    """
    return _unary_func_helper(x, _npi.floor, _np.floor, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def bitwise_not(x, out=None, **kwargs):
    r"""
    Compute bit-wise inversion, or bit-wise NOT, element-wise.
    Computes the bit-wise NOT of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``~``.

    Parameters
    ----------
    x : array_like
        Only integer and boolean types are handled.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.

    Returns
    -------
    out : ndarray or scalar
        Result.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    bitwise_and, bitwise_or, bitwise_xor
    logical_not
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    We've seen that 13 is represented by ``00001101``.
    The invert or bit-wise NOT of 13 is then:

    >>> x = np.invert(np.array(13, dtype=np.uint8))
    >>> x
    242
    >>> np.binary_repr(x, width=8)
    '11110010'

    Notes
    -----
    `bitwise_not` is an alias for `invert`:

    >>> np.bitwise_not is np.invert
    True
    """
    return _unary_func_helper(x, _npi.bitwise_not, _np.bitwise_not, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def invert(x, out=None, **kwargs):
    r"""
    Compute bit-wise inversion, or bit-wise NOT, element-wise.
    Computes the bit-wise NOT of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``~``.

    Parameters
    ----------
    x : array_like
        Only integer and boolean types are handled.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.

    Returns
    -------
    out : ndarray or scalar
        Result.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    bitwise_and, bitwise_or, bitwise_xor
    logical_not
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    We've seen that 13 is represented by ``00001101``.
    The invert or bit-wise NOT of 13 is then:

    >>> x = np.invert(np.array(13, dtype=np.uint8))
    >>> x
    242
    >>> np.binary_repr(x, width=8)
    '11110010'

    Notes
    -----
    `bitwise_not` is an alias for `invert`:

    >>> np.bitwise_not is np.invert
    True
    """
    return _unary_func_helper(x, _npi.bitwise_not, _np.bitwise_not, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def trunc(x, out=None, **kwargs):
    r"""
    Return the truncated value of the input, element-wise.
    The truncated value of the scalar `x` is the nearest integer `i` which
    is closer to zero than `x` is. In short, the fractional part of the
    signed number `x` is discarded.

    Parameters
    ----------
    x : ndarray or scalar
        Input data.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    y : ndarray or scalar
        The truncated value of each element in `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    This function differs from the original numpy.trunc in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.trunc(a)
    array([-1., -1., -0.,  0.,  1.,  1.,  2.])
    """
    return _unary_func_helper(x, _npi.trunc, _np.trunc, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def logical_not(x, out=None, **kwargs):
    r"""
    Compute the truth value of NOT x element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        Logical NOT is applied to the elements of `x`.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    y : bool or ndarray of bool
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    This function differs from the original numpy.logical_not in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> x= np.array([True, False, 0, 1])
    >>> np.logical_not(x)
    array([False,  True,  True, False])

    >>> x = np.arange(5)
    >>> np.logical_not(x<3)
    array([False, False, False,  True,  True])
    """
    return _unary_func_helper(x, _npi.logical_not, _np.logical_not, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def arcsinh(x, out=None, **kwargs):
    r"""
    Inverse hyperbolic sine, element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    arcsinh : ndarray
        Array of the same shape as `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    `arcsinh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `sinh(z) = x`.

    For real-valued input data types, `arcsinh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    This function differs from the original numpy.arcsinh in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Do not support complex-valued input.
        - Cannot cast type automatically. DType of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> a = np.array([3.2, 5.0])
    >>> np.arcsinh(a)
    array([1.8309381, 2.2924316])
    >>> np.arcsinh(1)
    0.0
    """
    return _unary_func_helper(x, _npi.arcsinh, _np.arcsinh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def arccosh(x, out=None, **kwargs):
    r"""
    Inverse hyperbolic cosine, element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    arccosh : ndarray
        Array of the same shape as `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    `arccosh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cosh(z) = x`.

    For real-valued input data types, `arccosh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    This function differs from the original numpy.arccosh in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Do not support complex-valued input.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> a = np.array([3.2, 5.0])
    >>> np.arccosh(a)
    array([1.8309381, 2.2924316])
    >>> np.arccosh(1)
    0.0
    """
    return _unary_func_helper(x, _npi.arccosh, _np.arccosh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def arctanh(x, out=None, **kwargs):
    r"""
    Inverse hyperbolic tangent, element-wise.

    Parameters
    ----------
    x : ndarray or scalar
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.

    Returns
    -------
    arctanh : ndarray
        Array of the same shape as `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    `arctanh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `tanh(z) = x`.

    For real-valued input data types, `arctanh` always returns real output.
    For each value that cannot be expressed as a real number or infinity, it
    yields ``nan`` and sets the `invalid` floating point error flag.

    This function differs from the original numpy.arctanh in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Do not support complex-valued input.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.

    Examples
    --------
    >>> a = np.array([0.0, -0.5])
    >>> np.arctanh(a)
    array([0., -0.54930615])
    >>> np.arctanh(0.0)
    0.0
    """
    return _unary_func_helper(x, _npi.arctanh, _np.arctanh, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def tile(A, reps):
    r"""
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Parameters
    ----------
    A : ndarray or scalar
        An input array or a scalar to repeat.
    reps : a single integer or tuple of integers
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.

    Examples
    --------
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0., 1., 2., 0., 1., 2.])
    >>> np.tile(a, (2, 2))
    array([[0., 1., 2., 0., 1., 2.],
           [0., 1., 2., 0., 1., 2.]])
    >>> np.tile(a, (2, 1, 2))
    array([[[0., 1., 2., 0., 1., 2.]],
           [[0., 1., 2., 0., 1., 2.]]])

    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1., 2., 1., 2.],
           [3., 4., 3., 4.]])
    >>> np.(b, (2, 1))
    array([[1., 2.],
           [3., 4.],
           [1., 2.],
           [3., 4.]])

    >>> c = np.array([1,2,3,4])
    >>> np.tile(c,(4,1))
    array([[1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.]])

    Scalar as input:

    >>> np.tile(2, 3)
    array([2, 2, 2]) # repeating integer `2`

    """
    return _unary_func_helper(A, _npi.tile, _np.tile, reps=reps)

# pylint: disable=redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def split(ary, indices_or_sections, axis=0):
    """
    Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D python tuple, list or set.
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.
        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in
          - ary[:2]
          - ary[2:3]
          - ary[3:]
        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division.
    """
    axis_size = ary.shape[axis]
    if isinstance(indices_or_sections, integer_types):
        sections = indices_or_sections
        if axis_size % sections:
            raise ValueError('array split does not result in an equal division')
        section_size = int(axis_size / sections)
        indices = [i * section_size for i in range(sections)]
    elif isinstance(indices_or_sections, (list, set, tuple)):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must be either int, or tuple / list / set of ints')
    ret = _npi.split(ary, indices, axis, False)
    assert isinstance(ret, list), 'Output of split should be list,' \
                                  ' got a return type {}'.format(type(ret))
    return ret
# pylint: enable=redefined-outer-name


# pylint: disable=redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def array_split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.

    If `indices_or_sections` is an integer, N, the array will be divided
    into N equal arrays along `axis`.  If such a split is not possible,
    an array of length l that should be split into n sections, it returns
    l % n sub-arrays of size l//n + 1 and the rest of size l//n.

    If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in
          - ary[:2]
          - ary[2:3]
          - ary[3:]
    If an index exceeds the dimension of the array along `axis`,
    an empty sub-array is returned correspondingly.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D Python tuple, list or set.
        Param used to determine the number and size of the subarray.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays.

    Examples
    --------
    >>> x = np.arange(9.0)
    >>> np.array_split(x, 3)
    [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7., 8.])]

    >>> np.array_split(x, [3, 5, 6, 8])
    [array([0., 1., 2.]), array([3., 4.]), array([5.]), array([6., 7.]), array([])]

    >>> x = np.arange(8.0)
    >>> np.array_split(x, 3)
    [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.])]

    >>> x = np.arange(7.0)
    >>> np.array_split(x, 3)
    [array([0.,  1.,  2.]), array([3.,  4.]), array([5.,  6.])]
    """
    indices = []
    sections = 0
    if isinstance(indices_or_sections, integer_types):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, (list, set, tuple)):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must be either int, or tuple / list / set of ints')
    ret = _npi.split(ary, indices, axis, False, sections)
    if not isinstance(ret, list):
        return [ret]
    return ret
# pylint: enable=redefined-outer-name


# pylint: disable=redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def hsplit(ary, indices_or_sections):
    """Split an array into multiple sub-arrays horizontally (column-wise).

    This is equivalent to ``split`` with ``axis=0`` if ``ary`` has one
    dimension, and otherwise that with ``axis=1``.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int, list of ints or tuple of ints.
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.

        If `indices_or_sections` is a list of sorted integers, the entries
        indicate where along `axis` the array is split.

        If an index exceeds the dimension of the array along `axis`,
        it will raises errors. so index must less than or euqal to
        the dimension of the array along axis.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays.

    Notes
    ------
    - If `indices_or_sections` is given as an integer, but a split
      does not result in equal division.It will raises ValueErrors.

    - If indices_or_sections is an integer, and the number is 1, it will
      raises an error. Because single output from split is not supported yet...

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.],
           [12., 13., 14., 15.]])
    >>> np.hsplit(x, 2)
    [array([[ 0.,  1.],
           [ 4.,  5.],
           [ 8.,  9.],
           [12., 13.]]),
    array([[ 2.,  3.],
           [ 6.,  7.],
           [10., 11.],
           [14., 15.]])]
    >>> np.hsplit(x, [3, 6])
    [array([[ 0.,  1.,  2.],
           [ 4.,  5.,  6.],
           [ 8.,  9., 10.],
           [12., 13., 14.]]),
    array([[ 3.],
           [ 7.],
           [11.],
           [15.]]),
    array([], shape=(4, 0), dtype=float32)]

    With a higher dimensional array the split is still along the second axis.

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[ 0.,  1.],
            [ 2.,  3.]],
           [[ 4.,  5.],
            [ 6.,  7.]]])
    >>> np.hsplit(x, 2)
    [array([[[ 0.,  1.]],
            [[ 4.,  5.]]]),
     array([[[ 2.,  3.]],
            [[ 6.,  7.]]])]

    If ``ary`` has one dimension, 'axis' = 0.
    >>> x = np.arange(4)
    array([0., 1., 2., 3.])
    >>> np.hsplit(x, 2)
    [array([0., 1.]), array([2., 3.])]

    If you want to produce an empty sub-array, you can see an example.
    >>> np.hsplit(x, [2, 2])
    [array([0., 1.]), array([], dtype=float32), array([2., 3.])]
    """
    if len(ary.shape) < 1:
        raise ValueError('hsplit only works on arrays of 1 or more dimensions')
    indices = []
    sections = 0
    if isinstance(indices_or_sections, integer_types):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, (list, set, tuple)):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must be either int, or tuple / list / set of ints')
    ret = _npi.hsplit(ary, indices, 1, False, sections)
    if not isinstance(ret, list):
        return [ret]
    return ret
# pylint: enable=redefined-outer-name


@set_module('mxnet.ndarray.numpy')
def vsplit(ary, indices_or_sections):
    r"""
    vsplit(ary, indices_or_sections)

    Split an array into multiple sub-arrays vertically (row-wise).

    ``vsplit`` is equivalent to ``split`` with `axis=0` (default): the array is always split
    along the first axis regardless of the array dimension.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1 - D Python tuple, list or set.
        If `indices_or_sections` is an integer, N, the array will be divided into N equal arrays
        along axis 0.  If such a split is not possible, an error is raised.

        If `indices_or_sections` is a 1-D array of sorted integers, the entries indicate where
        along axis 0 the array is split.  For example, ``[2, 3]`` would result in

          - ary[:2]
          - ary[2:3]
          - ary[3:]

        If an index exceeds the dimension of the array along axis 0, an error will be thrown.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays.

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Notes
    -------
    This function differs from the original `numpy.degrees
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.degrees.html>`_ in
    the following aspects:

    - Currently parameter ``indices_or_sections`` does not support ndarray, but supports scalar,
    tuple and list.
    - In ``indices_or_sections``, if an index exceeds the dimension of the array along axis 0,
    an error will be thrown.

    Examples
    --------
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.],
           [ 12.,  13.,  14.,  15.]])
    >>> np.vsplit(x, 2)
    [array([[0., 1., 2., 3.],
            [4., 5., 6., 7.]]), array([[ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])]

    With a higher dimensional array the split is still along the first axis.

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[ 0.,  1.],
            [ 2.,  3.]],
           [[ 4.,  5.],
            [ 6.,  7.]]])
    >>> np.vsplit(x, 2)
    [array([[[0., 1.],
            [2., 3.]]]), array([[[4., 5.],
            [6., 7.]]])]

    """
    if len(ary.shape) < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split(ary, indices_or_sections, 0)


# pylint: disable=redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def dsplit(ary, indices_or_sections):
    """
    Split array into multiple sub-arrays along the 3rd axis (depth).

    Please refer to the `split` documentation.  `dsplit` is equivalent
    to `split` with ``axis=2``, the array is always split along the third
    axis provided the array dimension is greater than or equal to 3.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1 - D Python tuple, list or set.
        If `indices_or_sections` is an integer, N, the array will be divided into N equal arrays
        along axis 2.  If such a split is not possible, an error is raised.

        If `indices_or_sections` is a 1-D array of sorted integers, the entries indicate where
        along axis 2 the array is split.  For example, ``[2, 3]`` would result in

          - ary[:, :, :2]
          - ary[:, :, 2:3]
          - ary[:, :, 3:]

        If an index exceeds the dimension of the array along axis 2, an error will be thrown.

    Examples
    --------
    >>> x = np.arange(16.0).reshape(2, 2, 4)
    >>> x
    array([[[ 0.,   1.,   2.,   3.],
            [ 4.,   5.,   6.,   7.]],
           [[ 8.,   9.,  10.,  11.],
            [12.,  13.,  14.,  15.]]])
    >>> np.dsplit(x, 2)
    [array([[[ 0.,  1.],
            [ 4.,  5.]],
           [[ 8.,  9.],
            [12., 13.]]]), array([[[ 2.,  3.],
            [ 6.,  7.]],
           [[10., 11.],
            [14., 15.]]])]
    >>> np.dsplit(x, np.array([3, 6]))
    [array([[[ 0.,   1.,   2.],
            [ 4.,   5.,   6.]],
           [[ 8.,   9.,  10.],
            [12.,  13.,  14.]]]),
     array([[[ 3.],
            [ 7.]],
           [[11.],
            [15.]]]),
    array([], shape=(2, 2, 0), dtype=float64)]
    """
    if len(ary.shape) < 3:
        raise ValueError('dsplit only works on arrays of 3 or more dimensions')
    return split(ary, indices_or_sections, 2)
# pylint: enable=redefined-outer-name


@set_module('mxnet.ndarray.numpy')
def concatenate(seq, axis=0, out=None):
    """
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : sequence of ndarray
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.

    Returns
    -------
    res : ndarray
        The concatenated array.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6]])
    >>> np.concatenate((a, b), axis=0)
    array([[1., 2.],
           [3., 4.],
           [5., 6.]])

    >>> np.concatenate((a, b), axis=None)
    array([1., 2., 3., 4., 5., 6.])

    >>> np.concatenate((a, b.T), axis=1)
    array([[1., 2., 5.],
           [3., 4., 6.]])
    """
    return _npi.concatenate(*seq, axis=axis, out=out)


@set_module('mxnet.ndarray.numpy')
def append(arr, values, axis=None):  # pylint: disable=redefined-outer-name
    """
    Append values to the end of an array.

    Parameters
    ----------
    arr : ndarray
        Values are appended to a copy of this array.
    values : ndarray
        These values are appended to a copy of `arr`.  It must be of the
        correct shape (the same shape as `arr`, excluding `axis`).  If
        `axis` is not specified, `values` can be any shape and will be
        flattened before use.
    axis : int, optional
        The axis along which `values` are appended.  If `axis` is not
        given, both `arr` and `values` are flattened before use.

    Returns
    -------
    append : ndarray
        A copy of `arr` with `values` appended to `axis`.  Note that
        `append` does not occur in-place: a new array is allocated and
        filled.  If `axis` is None, `out` is a flattened array.

    Examples
    --------
    >>> np.append(np.array([1, 2, 3]), np.array([[4, 5, 6],[7, 8, 9]]))
    array([1., 2., 3., 4., 5., 6., 7., 8., 9.])

    When `axis` is specified, `values` must have the correct shape.

    >>> np.append(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9]]), axis=0)
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])
    """
    return _npi.concatenate(arr, values, axis=axis, out=None)


@set_module('mxnet.ndarray.numpy')
def stack(arrays, axis=0, out=None):
    """Join a sequence of arrays along a new axis.
        The axis parameter specifies the index of the new axis in the dimensions of the result.
        For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last dimension.

    Parameters
    ----------
    arrays : sequence of ndarray
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what stack would have returned if no out argument were specified.

    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays."""
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]

    arrays = get_list(arrays)
    return _npi.stack(*arrays, axis=axis, out=out)


@set_module('mxnet.ndarray.numpy')
def vstack(arrays, out=None):
    r"""Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate` and `stack`
    provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> np.vstack((a, b))
    array([[1., 2., 3.],
            [2., 3., 4.]])

    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[2], [3], [4]])
    >>> np.vstack((a, b))
    array([[1.],
            [2.],
            [3.],
            [2.],
            [3.],
            [4.]])
    """
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]

    arrays = get_list(arrays)
    return _npi.vstack(*arrays)


@set_module('mxnet.ndarray.numpy')
def row_stack(arrays):
    r"""Stack arrays in sequence vertically (row wise).
    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.
    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate` and `stack`
    provide more general stacking and concatenation operations.
    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.
    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.
    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> np.vstack((a, b))
    array([[1., 2., 3.],
            [2., 3., 4.]])
    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[2], [3], [4]])
    >>> np.vstack((a, b))
    array([[1.],
            [2.],
            [3.],
            [2.],
            [3.],
            [4.]])
    """
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]

    arrays = get_list(arrays)
    return _npi.vstack(*arrays)


@set_module('mxnet.ndarray.numpy')
def column_stack(tup):
    """
    Stack 1-D arrays as columns into a 2-D array.
    Take a sequence of 1-D arrays and stack them as columns
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `hstack`.  1-D arrays are turned into 2-D columns
    first.

    Returns
    --------
    stacked : 2-D array
        The array formed by stacking the given arrays.

    See Also
    --------
    stack, hstack, vstack, concatenate

    Examples
    --------
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.column_stack((a,b))
    array([[1., 2.],
           [2., 3.],
           [3., 4.]])
    """
    return _npi.column_stack(*tup)


@set_module('mxnet.ndarray.numpy')
def hstack(arrays):
    """
    Stack arrays in sequence horizontally (column wise).
    This is equivalent to concatenation along the second axis,
    except for 1-D arrays where it concatenates along the first axis.
    Rebuilds arrays divided by hsplit.
    This function makes most sense for arrays with up to 3 dimensions.
    For instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions concatenate,
    stack and block provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of ndarrays
        The arrays must have the same shape along all but the second axis, except 1-D arrays which can be any length.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.

    Examples
    --------
    >>> from mxnet import np,npx
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.hstack((a,b))
    array([1., 2., 3., 2., 3., 4.])
    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[2],[3],[4]])
    >>> np.hstack((a,b))
    array([[1., 2.],
           [2., 3.],
           [3., 4.]])
    """
    return _npi.hstack(*arrays)


@set_module('mxnet.ndarray.numpy')
def dstack(arrays):
    """
    Stack arrays in sequence depth wise (along third axis).
    This is equivalent to concatenation along the third axis after 2-D arrays
    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by
    `dsplit`.
    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of arrays
        The arrays must have the same shape along all but the third axis.
        1-D or 2-D arrays must have the same shape.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 3-D.

    Examples
    --------
    >>> a = np.array((1,2,3))
    >>> b = np.array((2,3,4))
    >>> np.dstack((a,b))
    array([[[1, 2],
            [2, 3],
            [3, 4]]])
    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[2],[3],[4]])
    >>> np.dstack((a,b))
    array([[[1, 2]],
           [[2, 3]],
           [[3, 4]]])
    """
    return _npi.dstack(*arrays)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def maximum(x1, x2, out=None, **kwargs):
    """
    Returns element-wise maximum of the input arrays with broadcasting.

    Parameters
    ----------
    x1, x2 : scalar or mxnet.numpy.ndarray
        The arrays holding the elements to be compared. They must have the same shape,
        or shapes that can be broadcast to a single shape.

    Returns
    -------
    out : mxnet.numpy.ndarray or scalar
        The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars."""
    return _ufunc_helper(x1, x2, _npi.maximum, _np.maximum, _npi.maximum_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def minimum(x1, x2, out=None, **kwargs):
    """
    Returns element-wise minimum of the input arrays with broadcasting.

    Parameters
    ----------
    x1, x2 : scalar or mxnet.numpy.ndarray
        The arrays holding the elements to be compared. They must have the same shape,
        or shapes that can be broadcast to a single shape.

    Returns
    -------
    out : mxnet.numpy.ndarray or scalar
        The minimum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars."""
    return _ufunc_helper(x1, x2, _npi.minimum, _np.minimum, _npi.minimum_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : ndarray
        Swapped array. This is always a copy of the input array.
    """
    return _npi.swapaxes(a, dim1=axis1, dim2=axis2)


@set_module('mxnet.ndarray.numpy')
def clip(a, a_min, a_max, out=None):
    """clip(a, a_min, a_max, out=None)

    Clip (limit) the values in an array.
    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : ndarray
        Array containing elements to clip.
    a_min : scalar or `None`
        Minimum value. If `None`, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.
    a_max : scalar or `None`
        Maximum value. If `None`, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    Notes
    -----
    ndarray `a_min` and `a_max` are not supported.

    Examples
    --------
    >>> a = np.arange(10)
    >>> np.clip(a, 1, 8)
    array([1., 1., 2., 3., 4., 5., 6., 7., 8., 8.], dtype=float32)
    >>> a
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float32)
    >>> np.clip(a, 3, 6, out=a)
    array([3., 3., 3., 3., 4., 5., 6., 6., 6., 6.], dtype=float32)
    """
    if a_min is None and a_max is None:
        raise ValueError('array_clip: must set either max or min')
    if a_min is None:
        a_min = float('-inf')
    if a_max is None:
        a_max = float('inf')
    return _npi.clip(a, a_min, a_max, out=out)


@set_module('mxnet.ndarray.numpy')
def argmax(a, axis=None, out=None):
    r"""
    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : ndarray
        Input array. Only support ndarrays of dtype `float16`, `float32`, and `float64`.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    index_array : ndarray of indices whose dtype is same as the input ndarray.
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    This function differs from the original `numpy.argmax
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    Examples
    --------
    >>> a = np.arange(6).reshape(2,3) + 10
    >>> a
    array([[10., 11., 12.],
           [13., 14., 15.]])
    >>> np.argmax(a)
    array(5.)
    >>> np.argmax(a, axis=0)
    array([1., 1., 1.])
    >>> np.argmax(a, axis=1)
    array([2., 2.])

    >>> b = np.arange(6)
    >>> b[1] = 5
    >>> b
    array([0., 5., 2., 3., 4., 5.])
    >>> np.argmax(b)  # Only the first occurrence is returned.
    array(1.)

    Specify ``out`` ndarray:

    >>> a = np.arange(6).reshape(2,3) + 10
    >>> b = np.zeros((2,))
    >>> np.argmax(a, axis=1, out=b)
    array([2., 2.])
    >>> b
    array([2., 2.])
    """
    return _npi.argmax(a, axis=axis, keepdims=False, out=out)


@set_module('mxnet.ndarray.numpy')
def argmin(a, axis=None, out=None):
    r"""
    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : ndarray
        Input array. Only support ndarrays of dtype `float16`, `float32`, and `float64`.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : ndarray or None, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of indices whose dtype is same as the input ndarray.
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    This function differs from the original `numpy.argmax
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    Examples
    --------
    >>> a = np.arange(6).reshape(2,3) + 10
    >>> a
    array([[10., 11., 12.],
           [13., 14., 15.]])
    >>> np.argmin(a)
    array(0.)
    >>> np.argmin(a, axis=0)
    array([0., 0., 0.])
    >>> np.argmin(a, axis=1)
    array([0., 0.])

    >>> b = np.arange(6)
    >>> b[2] = 0
    >>> b
    array([0., 1., 0., 3., 4., 5.])
    >>> np.argmax(b)  # Only the first occurrence is returned.
    array(0.)

    Specify ``out`` ndarray:

    >>> a = np.arange(6).reshape(2,3) + 10
    >>> b = np.zeros((2,))
    >>> np.argmin(a, axis=1, out=b)
    array([0., 0.])
    >>> b
    array([0., 0.])
    """
    return _npi.argmin(a, axis=axis, keepdims=False, out=out)


@set_module('mxnet.ndarray.numpy')
def average(a, axis=None, weights=None, returned=False, out=None):
    """
    Compute the weighted average along the specified axis.

    Parameters
    --------
    a : ndarray
        Array containing data to be averaged.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average a.
        The default, axis=None, will average over
        all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.
        New in version 1.7.0.
        If axis is a tuple of ints, averaging is
        performed on all of the axes specified in the tuple
        instead of a single axis or all the axes as before.
    weights : ndarray, optional
        An array of weights associated with the values in a, must be the same dtype with a.
        Each value in a contributes to the average according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of a along the given axis) or of the same shape as a.
        If weights=None, then all data in a are assumed to have a weight equal to one.
        The 1-D calculation is: avg = sum(a * weights) / sum(weights)
        The only constraint on weights is that sum(weights) must not be 0.
    returned : bool, optional
        Default is False.
        If True, the tuple (average, sum_of_weights) is returned,
        otherwise only the average is returned.
        If weights=None, sum_of_weights is equivalent to
        the number of elements over which the average is taken.
    out : ndarray, optional
        If provided, the calculation is done into this array.

    Returns
    --------
    retval, [sum_of_weights] : ndarray
        Return the average along the specified axis.
        When returned is True, return a tuple with the average as the first element
        and the sum of the weights as the second element. sum_of_weights is of the same type as retval.
        If a is integral, the result dtype will be float32, otherwise it will be the same as dtype of a.

    Raises
    --------
        MXNetError
        - When all weights along axis sum to zero.
        - When the length of 1D weights is not the same as the shape of a along axis.
        - When given 1D weights, the axis is not specified or is not int.
        - When the shape of weights and a differ, but weights are not 1D.

    See also
    --------
        mean

    Notes
    --------
    This function differs from the original `numpy.average`
    <https://numpy.org/devdocs/reference/generated/numpy.average.html>`_ in
    the following way(s):

    - Does not guarantee the same behavior with numpy when given float16 dtype and overflow happens
    - Does not support complex dtype
    - The dtypes of a and weights must be the same
    - Integral a results in float32 returned dtype, not float64

    Examples
    --------
    >>> data = np.arange(1, 5)
    >>> data
    array([1., 2., 3., 4.])
    >>> np.average(data)
    array(2.5)
    >>> np.average(np.arange(1, 11), weights=np.arange(10, 0, -1))
    array(4.)
    >>> data = np.arange(6).reshape((3,2))
    >>> data
    array([[0., 1.],
           [2., 3.],
           [4., 5.]])
    >>> weights = np.array([0.25, 0.75])
    array([0.25, 0.75])
    >>> np.average(data, axis=1, weights=weights)
    array([0.75, 2.75, 4.75])
    """
    if weights is None:
        return _npi.average(a, axis=axis, weights=None, returned=returned, weighted=False, out=out)
    else:
        return _npi.average(a, axis=axis, weights=weights, returned=returned, out=out)


@set_module('mxnet.ndarray.numpy')
def mean(a, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
    """
    mean(a, axis=None, dtype=None, out=None, keepdims=None)
    Compute the arithmetic mean along the specified axis.
    Returns the average of the array elements.
    The average is taken over the flattened array by default, otherwise over the specified axis.
    Parameters
    ----------
    a : ndarray
        ndarray containing numbers whose mean is desired.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean. For integer inputs, the default is float32;
        for floating point inputs, it is the same as the input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result. The default is None; if provided,
        it must have the same shape and type as the expected output
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
        If the default value is passed, then keepdims will not be passed through to the mean
        method of sub-classes of ndarray, however any non-default value will be. If the sub-class
        method does not implement keepdims any exceptions will be raised.
    Returns
    -------
    m : ndarray, see dtype parameter above
        If out=None, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.
    Notes
    -----
    This function differs from the original `numpy.mean
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html>`_ in
    the following way(s):
    - only ndarray is accepted as valid input, python iterables or scalar is not supported
    - default data type for integer input is float32
    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.mean(a)
    array(2.5)
    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0,:] = 1.0
    >>> a[1,:] = 0.1
    >>> np.mean(a)
    array(0.55)
    >>> np.mean(a, dtype=np.float64)
    array(0.55)
    """
    return _npi.mean(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@set_module('mxnet.ndarray.numpy')
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):  # pylint: disable=too-many-arguments
    """
    Compute the standard deviation along the specified axis.
    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : ndarray
        Calculate the standard deviation of these values.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.std(a)
    1.1180339887498949 # may vary
    >>> np.std(a, axis=0)
    array([1.,  1.])
    >>> np.std(a, axis=1)
    array([0.5,  0.5])
    In single precision, std() can be inaccurate:
    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.std(a)
    array(0.45)
    >>> np.std(a, dtype=np.float64)
    array(0.45, dtype=float64)
    """
    return _npi.std(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, out=out)


@set_module('mxnet.ndarray.numpy')
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):  # pylint: disable=too-many-arguments
    """
    Compute the variance along the specified axis.
    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : ndarray
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float32`; for arrays of float types it is the same as
        the array type.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : int, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance;
        otherwise, a reference to the output array is returned.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.var(a)
    array(1.25)
    >>> np.var(a, axis=0)
    array([1.,  1.])
    >>> np.var(a, axis=1)
    array([0.25,  0.25])

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.var(a)
    array(0.2025)
    >>> np.var(a, dtype=np.float64)
    array(0.2025, dtype=float64)
    >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
    0.2025
    """
    return _npi.var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, out=out)


# pylint: disable=redefined-outer-name
@set_module('mxnet.ndarray.numpy')
def indices(dimensions, dtype=_np.int32, ctx=None):
    """Return an array representing the indices of a grid.

    Compute an array where the subarrays contain index values 0,1,...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : data-type, optional
        The desired data-type for the array. Default is `float32`.
    ctx : device context, optional
        Device context on which the memory is allocated. Default is
        `mxnet.context.current_context()`.

    Returns
    -------
    grid : ndarray
        The array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.

    Notes
    -----
    The output shape is obtained by prepending the number of dimensions
    in front of the tuple of dimensions, i.e. if `dimensions` is a tuple
    ``(r0, ..., rN-1)`` of length ``N``, the output shape is
    ``(N,r0,...,rN-1)``.

    The subarrays ``grid[k]`` contains the N-D array of indices along the
    ``k-th`` axis. Explicitly::

        grid[k,i0,i1,...,iN-1] = ik

    Examples
    --------
    >>> grid = np.indices((2, 3))
    >>> grid.shape
    (2, 2, 3)
    >>> grid[0]        # row indices
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1]        # column indices
    array([[0, 0, 0],
           [1, 1, 1]], dtype=int32)

    The indices can be used as an index into an array.

    >>> x = np.arange(20).reshape(5, 4)
    >>> row, col = np.indices((2, 3))
    >>> x[row, col]
    array([[0., 1., 2.],
           [4., 5., 6.]])

    Note that it would be more straightforward in the above example to
    extract the required elements directly with ``x[:2, :3]``.
    """
    if isinstance(dimensions, (tuple, list)):
        if ctx is None:
            ctx = current_context()
        return _npi.indices(dimensions=dimensions, dtype=dtype, ctx=ctx)
    else:
        raise ValueError("The dimensions must be sequence of ints")
# pylint: enable=redefined-outer-name


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def copysign(x1, x2, out=None, **kwargs):
    r"""
    Change the sign of x1 to that of x2, element-wise.

    If `x2` is a scalar, its sign will be copied to all elements of `x1`.

    Parameters
    ----------
    x1 : ndarray or scalar
        Values to change the sign of.
    x2 : ndarray or scalar
        The sign of `x2` is copied to `x1`.
    out : ndarray or None, optional
        A location into which the result is stored. It must be of the
        right shape and right type to hold the output. If not provided
        or `None`,a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray or scalar
        The values of `x1` with the sign of `x2`.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -------
    This function differs from the original `numpy.copysign
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.copysign.html>`_ in
    the following aspects:

    - ``where`` param is not supported.

    Examples
    --------
    >>> np.copysign(1.3, -1)
    -1.3
    >>> 1/np.copysign(0, 1)
    inf
    >>> 1/np.copysign(0, -1)
    -inf

    >>> a = np.array([-1, 0, 1])
    >>> np.copysign(a, -1.1)
    array([-1., -0., -1.])
    >>> np.copysign(a, np.arange(3)-1)
    array([-1.,  0.,  1.])
    """
    return _ufunc_helper(x1, x2, _npi.copysign, _np.copysign, _npi.copysign_scalar, _npi.rcopysign_scalar, out)


@set_module('mxnet.ndarray.numpy')
def ravel(x, order='C'):
    r"""
    ravel(x)

    Return a contiguous flattened array.
    A 1-D array, containing the elements of the input, is returned.  A copy is
    made only if needed.

    Parameters
    ----------
    x : ndarray
        Input array.  The elements in `x` are read in row-major, C-style order and
        packed as a 1-D array.
    order : `C`, optional
        Only support row-major, C-style order.

    Returns
    -------
    y : ndarray
        y is an array of the same subtype as `x`, with shape ``(x.size,)``.
        Note that matrices are special cased for backward compatibility, if `x`
        is a matrix, then y is a 1-D ndarray.

    Notes
    -----
    This function differs from the original numpy.arange in the following aspects:
        - Only support row-major, C-style order.

    Examples
    --------
    It is equivalent to ``reshape(x, -1)``.

    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> print(np.ravel(x))
    [1. 2. 3. 4. 5. 6.]

    >>> print(x.reshape(-1))
    [1. 2. 3. 4. 5. 6.]

    >>> print(np.ravel(x.T))
    [1. 4. 2. 5. 3. 6.]
    """
    if order == 'F':
        raise NotImplementedError('order {} is not supported'.format(order))
    if isinstance(x, numeric_types):
        return _np.reshape(x, -1)
    elif isinstance(x, NDArray):
        return _npi.reshape(x, -1)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


def unravel_index(indices, shape, order='C'): # pylint: disable=redefined-outer-name
    """
    Converts a flat index or array of flat indices into a tuple of coordinate arrays.

    Parameters:
    -------------
    indices : array_like
            An integer array whose elements are indices into the flattened version of an array of dimensions shape.
            Before version 1.6.0, this function accepted just one index value.
    shape : tuple of ints
            The shape of the array to use for unraveling indices.

    Returns:
    -------------
    unraveled_coords : ndarray
            Each row in the ndarray has the same shape as the indices array.
            Each column in the ndarray represents the unravelled index

    Examples:
    -------------
    >>> np.unravel_index([22, 41, 37], (7,6))
    ([3. 6. 6.]
      [4. 5. 1.])
    >>> np.unravel_index(1621, (6,7,8,9))
    (3, 1, 4, 1)
    """
    if order == 'C':
        if isinstance(indices, numeric_types):
            return _np.unravel_index(indices, shape)
        ret = _npi.unravel_index_fallback(indices, shape=shape)
        ret_list = []
        for item in ret:
            ret_list += [item]
        return tuple(ret_list)
    else:
        raise NotImplementedError('Do not support column-major (Fortran-style) order at this moment')


def diag_indices_from(arr):
    """
    This returns a tuple of indices that can be used to access the main diagonal of an array
    a with a.ndim >= 2 dimensions and shape (n, n, ..., n). For a.ndim = 2 this is
    the usual diagonal, for a.ndim > 2 this is the set of indices to access
    a[i, i, ..., i] for i = [0..n-1].

    Parameters:
    -------------
    arr : ndarray
        Input array for acessing the main diagonal. All dimensions
        should have equal length.

    Return:
    -------------
    diag: tuple of ndarray
        indices of the main diagonal.

    Examples:
    -------------
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
    >>> idx = np.diag_indices_from(a)
    >>> idx
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> a[idx] = 100
    >>> a
    array([[100,   1,   2,   3],
        [  4, 100,   6,   7],
        [  8,   9, 100,  11],
        [ 12,  13,  14, 100]])
    """
    return tuple(_npi.diag_indices_from(arr))


@set_module('mxnet.ndarray.numpy')
def hanning(M, dtype=_np.float32, ctx=None):
    r"""Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `float32`. Note that you need
        select numpy.float32 or float64 in this operator.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    blackman, hamming

    Notes
    -----
    The Hanning window is defined as

    .. math::  w(n) = 0.5 - 0.5cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The Hanning was named for Julius von Hann, an Austrian meteorologist.
    It is also known as the Cosine Bell. Some authors prefer that it be
    called a Hann window, to help avoid confusion with the very similar
    Hamming window.

    Most references to the Hanning window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           http://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> np.hanning(12)
    array([0.        , 0.07937324, 0.29229254, 0.5711574 , 0.8274304 ,
           0.9797465 , 0.97974646, 0.82743025, 0.5711573 , 0.29229245,
           0.07937312, 0.        ])

    Plot the window and its frequency response:

    >>> import matplotlib.pyplot as plt
    >>> window = np.hanning(51)
    >>> plt.plot(window.asnumpy())
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("Hann window")
    Text(0.5, 1.0, 'Hann window')
    >>> plt.ylabel("Amplitude")
    Text(0, 0.5, 'Amplitude')
    >>> plt.xlabel("Sample")
    Text(0.5, 0, 'Sample')
    >>> plt.show()
    """
    if ctx is None:
        ctx = current_context()
    return _npi.hanning(M, dtype=dtype, ctx=ctx)


@set_module('mxnet.ndarray.numpy')
def hamming(M, dtype=_np.float32, ctx=None):
    r"""Return the hamming window.

    The hamming window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `float32`. Note that you need
        select numpy.float32 or float64 in this operator.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    blackman, hanning

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey
    and is described in Blackman and Tukey. It was recommended for
    smoothing the truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> np.hamming(12)
    array([0.08000001, 0.15302339, 0.34890914, 0.6054648 , 0.841236  ,
           0.9813669 , 0.9813668 , 0.8412359 , 0.6054647 , 0.34890908,
           0.15302327, 0.08000001])

    Plot the window and its frequency response:

    >>> import matplotlib.pyplot as plt
    >>> window = np.hamming(51)
    >>> plt.plot(window.asnumpy())
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("hamming window")
    Text(0.5, 1.0, 'hamming window')
    >>> plt.ylabel("Amplitude")
    Text(0, 0.5, 'Amplitude')
    >>> plt.xlabel("Sample")
    Text(0.5, 0, 'Sample')
    >>> plt.show()
    """
    if ctx is None:
        ctx = current_context()
    return _npi.hamming(M, dtype=dtype, ctx=ctx)


@set_module('mxnet.ndarray.numpy')
def blackman(M, dtype=_np.float32, ctx=None):
    r"""Return the Blackman window.

    The Blackman window is a taper formed by using the first three
    terms of a summation of cosines. It was designed to have close to the
    minimal leakage possible.  It is close to optimal, only slightly worse
    than a Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `float32`. Note that you need
        select numpy.float32 or float64 in this operator.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    hamming, hanning

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/{M-1}) + 0.08 \cos(4\pi n/{M-1})

    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the kaiser window.

    References
    ----------
    Blackman, R.B. and Tukey, J.W., (1958) The measurement of power spectra,
    Dover Publications, New York.

    Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
    Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.

    Examples
    --------
    >>> np.blackman(12)
    array([-1.4901161e-08,  3.2606423e-02,  1.5990365e-01,  4.1439798e-01,
            7.3604530e-01,  9.6704686e-01,  9.6704674e-01,  7.3604506e-01,
            4.1439781e-01,  1.5990359e-01,  3.2606363e-02, -1.4901161e-08])

    Plot the window and its frequency response:

    >>> import matplotlib.pyplot as plt
    >>> window = np.blackman(51)
    >>> plt.plot(window.asnumpy())
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("blackman window")
    Text(0.5, 1.0, 'blackman window')
    >>> plt.ylabel("Amplitude")
    Text(0, 0.5, 'Amplitude')
    >>> plt.xlabel("Sample")
    Text(0.5, 0, 'Sample')
    >>> plt.show()
    """
    if ctx is None:
        ctx = current_context()
    return _npi.blackman(M, dtype=dtype, ctx=ctx)


@set_module('mxnet.ndarray.numpy')
def flip(m, axis=None, out=None):
    r"""
    flip(m, axis=None, out=None)

    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : ndarray or scalar
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to flip over. The default,
        axis=None, will flip over all of the axes of the input array.
        If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, flipping is performed on all of the axes
        specified in the tuple.
    out : ndarray or scalar, optional
        Alternative output array in which to place the result. It must have
        the same shape and type as the expected output.

    Returns
    -------
    out : ndarray or scalar
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.

    Examples
    --------
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> np.flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> np.flip(A)
    array([[[7, 6],
            [5, 4]],
           [[3, 2],
            [1, 0]]])
    >>> np.flip(A, (0, 2))
    array([[[5, 4],
            [7, 6]],
           [[1, 0],
            [3, 2]]])
    """
    from ...numpy import ndarray
    if isinstance(m, numeric_types):
        return _np.flip(m, axis)
    elif isinstance(m, ndarray):
        return _npi.flip(m, axis, out=out)
    else:
        raise TypeError('type {} not supported'.format(str(type(m))))


@set_module('mxnet.ndarray.numpy')
def flipud(m):
    r"""
    flipud(*args, **kwargs)

    Flip array in the up/down direction.

    Flip the entries in each column in the up/down direction.
    Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array.

    Returns
    -------
    out : array_like
        A view of `m` with the rows reversed.  Since a view is
        returned, this operation is :math:`\mathcal O(1)`.

    See Also
    --------
    fliplr : Flip array in the left/right direction.
    rot90 : Rotate array counterclockwise.

    Notes
    -----
    Equivalent to ``m[::-1,...]``.
    Does not require the array to be two-dimensional.

    Examples
    --------
    >>> A = np.diag(np.array([1.0, 2, 3]))
    >>> A
    array([[1.,  0.,  0.],
           [0.,  2.,  0.],
           [0.,  0.,  3.]])
    >>> np.flipud(A)
    array([[0.,  0.,  3.],
           [0.,  2.,  0.],
           [1.,  0.,  0.]])

    >>> A = np.random.randn(2,3,5)
    >>> np.all(np.flipud(A) == A[::-1,...])
    array(True)

    >>> np.flipud(np.array([1,2]))
    array([2., 1.])
    """
    return flip(m, 0)


@set_module('mxnet.ndarray.numpy')
def fliplr(m):
    r"""
    fliplr(*args, **kwargs)

    Flip array in the left/right direction.

    Flip the entries in each row in the left/right direction.
    Columns are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array, must be at least 2-D.

    Returns
    -------
    f : ndarray
        A view of `m` with the columns reversed.  Since a view
        is returned, this operation is :math:`\mathcal O(1)`.

    See Also
    --------
    flipud : Flip array in the up/down direction.
    rot90 : Rotate array counterclockwise.

    Notes
    -----
    Equivalent to m[:,::-1]. Requires the array to be at least 2-D.

    Examples
    --------
    >>> A = np.diag(np.array([1.,2.,3.]))
    >>> A
    array([[1.,  0.,  0.],
           [0.,  2.,  0.],
           [0.,  0.,  3.]])
    >>> np.fliplr(A)
    array([[0.,  0.,  1.],
           [0.,  2.,  0.],
           [3.,  0.,  0.]])

    >>> A = np.random.randn(2,3,5)
    >>> np.all(np.fliplr(A) == A[:,::-1,...])
    array(True)
    """
    return flip(m, 1)


@set_module('mxnet.ndarray.numpy')
def around(x, decimals=0, out=None, **kwargs):
    r"""
    around(x, decimals=0, out=None)

    Evenly round to the given number of decimals.
    Parameters
    ----------
    x : ndarray or scalar
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).  If
        decimals is negative, it specifies the number of positions to
        the left of the decimal point.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape and type as the expected output.

    Returns
    -------
    rounded_array : ndarray or scalar
        An array of the same type as `x`, containing the rounded values.
        A reference to the result is returned.

    Notes
    -----
    For values exactly halfway between rounded decimal values, NumPy
    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
    -0.5 and 0.5 round to 0.0, etc.

    This function differs from the original numpy.prod in the following aspects:

        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot support complex-valued number.

    Examples
    --------
    >>> np.around([0.37, 1.64])
    array([ 0.,  2.])
    >>> np.around([0.37, 1.64], decimals=1)
    array([ 0.4,  1.6])
    >>> np.around([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
    array([ 0.,  2.,  2.,  4.,  4.])
    >>> np.around([1, 2, 3, 11], decimals=1) # ndarray of ints is returned
    array([ 1,  2,  3, 11])
    >>> np.around([1, 2, 3, 11], decimals=-1)
    array([ 0,  0,  0, 10])
    """
    from ...numpy import ndarray
    if isinstance(x, numeric_types):
        return _np.around(x, decimals, **kwargs)
    elif isinstance(x, ndarray):
        return _npi.around(x, decimals, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.ndarray.numpy')
def round(x, decimals=0, out=None, **kwargs):
    r"""
    round(a, decimals=0, out=None)
    Round an array to the given number of decimals.

    See Also
    --------
    around : equivalent function; see for details.
    """
    from ...numpy import ndarray
    if isinstance(x, numeric_types):
        return _np.around(x, decimals, **kwargs)
    elif isinstance(x, ndarray):
        return _npi.around(x, decimals, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.ndarray.numpy')
def round_(x, decimals=0, out=None, **kwargs):
    r"""
    round_(a, decimals=0, out=None)
    Round an array to the given number of decimals.

    See Also
    --------
    around : equivalent function; see for details.
    """
    from ...numpy import ndarray
    if isinstance(x, numeric_types):
        return _np.around(x, decimals, **kwargs)
    elif isinstance(x, ndarray):
        return _npi.around(x, decimals, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def arctan2(x1, x2, out=None, **kwargs):
    r"""
    Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    The quadrant (i.e., branch) is chosen so that ``arctan2(x1, x2)`` is
    the signed angle in radians between the ray ending at the origin and
    passing through the point (1,0), and the ray ending at the origin and
    passing through the point (`x2`, `x1`).  (Note the role reversal: the
    "`y`-coordinate" is the first function parameter, the "`x`-coordinate"
    is the second.)  By IEEE convention, this function is defined for
    `x2` = +/-0 and for either or both of `x1` and `x2` = +/-inf (see
    Notes for specific values).

    This function is not defined for complex-valued arguments; for the
    so-called argument of complex values, use `angle`.

    Parameters
    ----------
    x1 : ndarray or scalar
        `y`-coordinates.
    x2 : ndarray or scalar
        `x`-coordinates. `x2` must be broadcastable to match the shape of
        `x1` or vice versa.
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray or scalar
        Array of angles in radians, in the range ``[-pi, pi]``. This is a scalar if
        `x1` and `x2` are scalars.

    Notes
    -----
    *arctan2* is identical to the `atan2` function of the underlying
    C library.  The following special values are defined in the C
    standard: [1]_

    ====== ====== ================
    `x1`   `x2`   `arctan2(x1,x2)`
    ====== ====== ================
    +/- 0  +0     +/- 0
    +/- 0  -0     +/- pi
        > 0   +/-inf +0 / +pi
        < 0   +/-inf -0 / -pi
    +/-inf +inf   +/- (pi/4)
    +/-inf -inf   +/- (3*pi/4)
    ====== ====== ================

    Note that +0 and -0 are distinct floating point numbers, as are +inf
    and -inf.

    This function differs from the original numpy.arange in the following aspects:
        - Only support float16, float32 and float64.

    References
    ----------
    .. [1] ISO/IEC standard 9899:1999, "Programming language C."

    Examples
    --------
    Consider four points in different quadrants:

    >>> x = np.array([-1, +1, +1, -1])
    >>> y = np.array([-1, -1, +1, +1])
    >>> np.arctan2(y, x) * 180 / np.pi
    array([-135.,  -45.,   45.,  135.])

    Note the order of the parameters. `arctan2` is defined also when `x2` = 0
    and at several other special points, obtaining values in
    the range ``[-pi, pi]``:

    >>> x = np.array([1, -1])
    >>> y = np.array([0, 0])
    >>> np.arctan2(x, y)
    array([ 1.5707964, -1.5707964])
    """
    return _ufunc_helper(x1, x2, _npi.arctan2, _np.arctan2,
                         _npi.arctan2_scalar, _npi.rarctan2_scalar, out=out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def hypot(x1, x2, out=None, **kwargs):
    r"""
    Given the "legs" of a right triangle, return its hypotenuse.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise.  If `x1` or
    `x2` is scalar_like (i.e., unambiguously cast-able to a scalar type),
    it is broadcast for use with each element of the other argument.

    Parameters
    ----------
    x1, x2 : ndarray
        Leg of the triangle(s).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.

    Returns
    -------
    z : ndarray
        The hypotenuse of the triangle(s).
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    This function differs from the original numpy.arange in the following aspects:
        - Only support float16, float32 and float64.

    Examples
    --------
    >>> np.hypot(3*np.ones((3, 3)), 4*np.ones((3, 3)))
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])

    Example showing broadcast of scalar_like argument:

    >>> np.hypot(3*np.ones((3, 3)), [4])
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])
    """
    return _ufunc_helper(x1, x2, _npi.hypot, _np.hypot, _npi.hypot_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def bitwise_and(x1, x2, out=None, **kwargs):
    r"""
    Compute the bit-wise XOR of two arrays element-wise.

    Parameters
    ----------
    x1, x2 : ndarray or scalar
        Only integer and boolean types are handled. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that the
        inputs broadcast to. If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray
        Result.

    Examples
    --------
    >>> np.bitwise_and(13, 17)
    1

    >>> np.bitwise_and(14, 13)
    12
    >>> np.bitwise_and(np.array([14,3], dtype='int32'), 13)
    array([12,  1], dtype=int32)

    >>> np.bitwise_and(np.array([11,7], dtype='int32'), np.array([4,25], dtype='int32'))
    array([0, 1], dtype=int32)
    >>> np.bitwise_and(np.array([2,5,255], dtype='int32'), np.array([3,14,16], dtype='int32'))
    array([ 2,  4, 16], dtype=int32)
    >>> np.bitwise_and(np.array([True, True], dtype='bool'), np.array([False, True], dtype='bool'))
    array([False,  True])
    """
    return _ufunc_helper(x1, x2, _npi.bitwise_and, _np.bitwise_and, _npi.bitwise_and_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def bitwise_xor(x1, x2, out=None, **kwargs):
    r"""
    Compute the bit-wise XOR of two arrays element-wise.

    Parameters
    ----------
    x1, x2 : ndarray or scalar
        Only integer and boolean types are handled. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that the
        inputs broadcast to. If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray
        Result.

    Examples
    --------
    >>> np.bitwise_xor(13, 17)
    28

    >>> np.bitwise_xor(31, 5)
    26
    >>> np.bitwise_xor(np.array([31,3], dtype='int32'), 5)
    array([26,  6])

    >>> np.bitwise_xor(np.array([31,3], dtype='int32'), np.array([5,6], dtype='int32'))
    array([26,  5])
    >>> np.bitwise_xor(np.array([True, True], dtype='bool'), np.array([False, True], dtype='bool'))
    array([ True, False])
    """
    return _ufunc_helper(x1, x2, _npi.bitwise_xor, _np.bitwise_xor, _npi.bitwise_xor_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def bitwise_or(x1, x2, out=None, **kwargs):
    r"""
    Compute the bit-wise OR of two arrays element-wise.

    Parameters
    ----------
    x1, x2 : ndarray or scalar
        Only integer and boolean types are handled. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have a shape that the
        inputs broadcast to. If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray
        Result.

    Examples
    --------
    >>> np.bitwise_or(13, 17)
    29

    >>> np.bitwise_or(31, 5)
    31
    >>> np.bitwise_or(np.array([31,3], dtype='int32'), 5)
    array([31,  7])

    >>> np.bitwise_or(np.array([31,3], dtype='int32'), np.array([5,6], dtype='int32'))
    array([31,  7])
    >>> np.bitwise_or(np.array([True, True], dtype='bool'), np.array([False, True], dtype='bool'))
    array([ True, True])
    """
    return _ufunc_helper(x1, x2, _npi.bitwise_or, _np.bitwise_or, _npi.bitwise_or_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
@wrap_np_binary_func
def ldexp(x1, x2, out=None, **kwargs):
    """
    Returns x1 * 2**x2, element-wise.
    The mantissas `x1` and twos exponents `x2` are used to construct
    floating point numbers ``x1 * 2**x2``.

    Parameters
    ----------
    x1 : ndarray or scalar
        Array of multipliers.
    x2 : ndarray or scalar, int
        Array of twos exponents.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or scalar
        The result of ``x1 * 2**x2``.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.
    Different from numpy, we allow x2 to be float besides int.
    `ldexp` is useful as the inverse of `frexp`, if used by itself it is
    more clear to simply use the expression ``x1 * 2**x2``.

    Examples
    --------
    >>> np.ldexp(5, np.arange(4))
    array([  5.,  10.,  20.,  40.])
    """
    return _ufunc_helper(x1, x2, _npi.ldexp, _np.ldexp, _npi.ldexp_scalar, _npi.rldexp_scalar, out)


@set_module('mxnet.ndarray.numpy')
def inner(a, b):
    r"""
    Inner product of two arrays.
    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : ndarray
        If `a` and `b` are nonscalar, their last dimensions must match.

    Returns
    -------
    out : ndarray
        `out.shape = a.shape[:-1] + b.shape[:-1]`

    Raises
    ------
    ValueError
        If the last dimension of `a` and `b` has different size.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.
    einsum : Einstein summation convention.

    Notes
    -----
    For vectors (1-D arrays) it computes the ordinary inner-product::
        np.inner(a, b) = sum(a[:]*b[:])
    More generally, if `ndim(a) = r > 0` and `ndim(b) = s > 0`::
        np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))
    or explicitly::
        np.inner(a, b)[i0,...,ir-1,j0,...,js-1]
            = sum(a[i0,...,ir-1,:]*b[j0,...,js-1,:])
    In addition `a` or `b` may be scalars, in which case::
    np.inner(a,b) = a*b

    Examples
    --------
    Ordinary inner product for vectors:
    >>> a = np.array([1,2,3])
    >>> b = np.array([0,1,0])
    >>> np.inner(a, b)
    2
    A multidimensional example:
    >>> a = np.arange(24).reshape((2,3,4))
    >>> b = np.arange(4)
    >>> np.inner(a, b)
    array([[ 14,  38,  62],
           [ 86, 110, 134]])
    """
    return tensordot(a, b, [-1, -1])


@set_module('mxnet.ndarray.numpy')
def outer(a, b):
    r"""
    Compute the outer product of two vectors.
    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::
    [[a0*b0  a0*b1 ... a0*bN ]
    [a1*b0    .
    [ ...          .
    [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : (M,) ndarray
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) ndarray
        Second input vector.  Input is flattened if
        not already 1-dimensional.

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``
    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
    ufunc.outer : A generalization to N dimensions and other operations.
                ``np.multiply.outer(a.ravel(), b.ravel())`` is the equivalent.
    References
    ----------
    .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
            ed., Baltimore, MD, Johns Hopkins University Press, 1996,
            pg. 8.
    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:
    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
        [-2., -1.,  0.,  1.,  2.],
        [-2., -1.,  0.,  1.,  2.],
        [-2., -1.,  0.,  1.,  2.],
        [-2., -1.,  0.,  1.,  2.]])
    """
    return tensordot(a.flatten(), b.flatten(), 0)


@set_module('mxnet.ndarray.numpy')
def vdot(a, b):
    r"""
    Return the dot product of two vectors.
    Note that `vdot` handles multidimensional arrays differently than `dot`:
    it does *not* perform a matrix product, but flattens input arguments
    to 1-D vectors first. Consequently, it should only be used for vectors.

    Parameters
    ----------
    a : ndarray
        First argument to the dot product.
    b : ndarray
        Second argument to the dot product.

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
    >>> np.vdot(a, b)
    30
    >>> np.vdot(b, a)
    30
    >>> 1*4 + 4*1 + 5*2 + 6*2
    30
    """
    return tensordot(a.flatten(), b.flatten(), 1)


@set_module('mxnet.ndarray.numpy')
def equal(x1, x2, out=None):
    """
    Return (x1 == x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less
    Examples
    --------
    >>> np.equal(np.ones(2, 1)), np.zeros(1, 3))
    array([[False, False, False],
           [False, False, False]])
    >>> np.equal(1, np.ones(1))
    array([ True])
    """
    return _ufunc_helper(x1, x2, _npi.equal, _np.equal, _npi.equal_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def not_equal(x1, x2, out=None):
    """
    Return (x1 != x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.not_equal(np.ones(2, 1)), np.zeros(1, 3))
    array([[ True,  True,  True],
           [ True,  True,  True]])
    >>> np.not_equal(1, np.ones(1))
    array([False])
    """
    return _ufunc_helper(x1, x2, _npi.not_equal, _np.not_equal, _npi.not_equal_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def greater(x1, x2, out=None):
    """
    Return the truth value of (x1 > x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.greater(np.ones(2, 1)), np.zeros(1, 3))
    array([[ True,  True,  True],
           [ True,  True,  True]])
    >>> np.greater(1, np.ones(1))
    array([False])
    """
    return _ufunc_helper(x1, x2, _npi.greater, _np.greater, _npi.greater_scalar,
                         _npi.less_scalar, out)


@set_module('mxnet.ndarray.numpy')
def less(x1, x2, out=None):
    """
    Return the truth value of (x1 < x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.less(np.ones(2, 1)), np.zeros(1, 3))
    array([[ True,  True,  True],
           [ True,  True,  True]])
    >>> np.less(1, np.ones(1))
    array([False])
    """
    return _ufunc_helper(x1, x2, _npi.less, _np.less, _npi.less_scalar, _npi.greater_scalar, out)


@set_module('mxnet.ndarray.numpy')
def greater_equal(x1, x2, out=None):
    """
    Return the truth value of (x1 >= x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.greater_equal(np.ones(2, 1)), np.zeros(1, 3))
    array([[ True,  True,  True],
           [ True,  True,  True]])
    >>> np.greater_equal(1, np.ones(1))
    array([True])
    """
    return _ufunc_helper(x1, x2, _npi.greater_equal, _np.greater_equal, _npi.greater_equal_scalar,
                         _npi.less_equal_scalar, out)


@set_module('mxnet.ndarray.numpy')
def less_equal(x1, x2, out=None):
    """
    Return the truth value of (x1 <= x2) element-wise.
    Parameters
    ----------
    x1, x2 : ndarrays or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : ndarray or scalar
        Output array of type bool, element-wise comparison of `x1` and `x2`.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    equal, greater, greater_equal, less, less_equal
    Examples
    --------
    >>> np.less_equal(np.ones(2, 1)), np.zeros(1, 3))
    array([[False, False, False],
           [False, False, False]])
    >>> np.less_equal(1, np.ones(1))
    array([True])
    """
    return _ufunc_helper(x1, x2, _npi.less_equal, _np.less_equal, _npi.less_equal_scalar,
                         _npi.greater_equal_scalar, out)


@set_module('mxnet.ndarray.numpy')
def rot90(m, k=1, axes=(0, 1)):
    """
    Rotate an array by 90 degrees in the plane specified by axes.
    Rotation direction is from the first towards the second axis.
    Parameters
    ----------
    m : ndarray
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes: (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.

    Returns
    -------
    y : ndarray
        A rotated view of `m`.

    -----
    rot90(m, k=1, axes=(1,0)) is the reverse of rot90(m, k=1, axes=(0,1))
    rot90(m, k=1, axes=(1,0)) is equivalent to rot90(m, k=-1, axes=(0,1))
    Examples
    --------
    >>> m = np.array([[1,2],[3,4]], 'int')
    >>> m
    array([[1, 2],
           [3, 4]], dtype=int64)
    >>> np.rot90(m)
    array([[2, 4],
           [1, 3]], dtype=int64)
    >>> np.rot90(m, 2)
    array([[4, 3],
           [2, 1]], dtype=int64)
    >>> m = np.arange(8).reshape((2,2,2))
    >>> np.rot90(m, 1, (1,2))
    array([[[1., 3.],
            [0., 2.]],

           [[5., 7.],
            [4., 6.]]])
    """
    return _npi.rot90(m, k=k, axes=axes)


@set_module('mxnet.ndarray.numpy')
def einsum(*operands, **kwargs):
    r"""
    einsum(subscripts, *operands, out=None, optimize=False)

    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    See the notes and examples for clarification.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list of ndarray
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    optimize : {False, True}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False. Defaults to False.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    Notes
    -----
    The Einstein summation convention can be used to compute
    many multi-dimensional, linear algebraic array operations. `einsum`
    provides a succinct way of representing these.

    A non-exhaustive list of these operations,
    which can be computed by `einsum`, is shown below along with examples:

    * Trace of an array, :py:func:`np.trace`.
    * Return a diagonal, :py:func:`np.diag`.
    * Array axis summations, :py:func:`np.sum`.
    * Transpositions and permutations, :py:func:`np.transpose`.
    * Matrix multiplication and dot product, :py:func:`np.matmul` :py:func:`np.dot`.
    * Vector inner and outer products, :py:func:`np.inner` :py:func:`np.outer`.
    * Broadcasting, element-wise and scalar multiplication, :py:func:`np.multiply`.
    * Tensor contractions, :py:func:`np.tensordot`.

    The subscripts string is a comma-separated list of subscript labels,
    where each label refers to a dimension of the corresponding operand.
    Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``
    is equivalent to :py:func:`np.inner(a,b) <np.inner>`. If a label
    appears only once, it is not summed, so ``np.einsum('i', a)`` produces a
    view of ``a`` with no changes. A further example ``np.einsum('ij,jk', a, b)``
    describes traditional matrix multiplication and is equivalent to
    :py:func:`np.matmul(a,b) <np.matmul>`. Repeated subscript labels in one
    operand take the diagonal. For example, ``np.einsum('ii', a)`` is equivalent
    to :py:func:`np.trace(a) <np.trace>`.

    In *implicit mode*, the chosen subscripts are important
    since the axes of the output are reordered alphabetically.  This
    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
    ``np.einsum('ji', a)`` takes its transpose. Additionally,
    ``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
    ``np.einsum('ij,jh', a, b)`` returns the transpose of the
    multiplication since subscript 'h' precedes subscript 'i'.

    In *explicit mode* the output can be directly controlled by
    specifying output subscript labels.  This requires the
    identifier '->' as well as the list of output subscript labels.
    This feature increases the flexibility of the function since
    summing can be disabled or forced when required. The call
    ``np.einsum('i->', a)`` is like :py:func:`np.sum(a, axis=-1) <np.sum>`,
    and ``np.einsum('ii->i', a)`` is like :py:func:`np.diag(a) <np.diag>`.
    The difference is that `einsum` does not allow broadcasting by default.
    Additionally ``np.einsum('ij,jh->ih', a, b)`` directly specifies the
    order of the output subscript labels and therefore returns matrix
    multiplication, unlike the example above in implicit mode.

    To enable and control broadcasting, use an ellipsis.  Default
    NumPy-style broadcasting is done by adding an ellipsis
    to the left of each term, like ``np.einsum('...ii->...i', a)``.
    To take the trace along the first and last axes,
    you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
    product with the left-most indices instead of rightmost, one can do
    ``np.einsum('ij...,jk...->ik...', a, b)``.

    When there is only one operand, no axes are summed, and no output
    parameter is provided, a view into the operand is returned instead
    of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
    produces a view.

    The ``optimize`` argument which will optimize the contraction order
    of an einsum expression. For a contraction with three or more operands this
    can greatly increase the computational efficiency at the cost of a larger
    memory footprint during computation.

    Typically a 'greedy' algorithm is applied which empirical tests have shown
    returns the optimal path in the majority of cases. 'optimal' is not supported
    for now.

    This function differs from the original `numpy.einsum
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html>`_ in
    the following way(s):

    - Does not support 'optimal' strategy
    - Does not support the alternative subscript like
        `einsum(op0, sublist0, op1, sublist1, ..., [sublistout])`
    - Does not produce view in any cases

    Examples
    --------
    >>> a = np.arange(25).reshape(5,5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2,3)

    Trace of a matrix:

    >>> np.einsum('ii', a)
    array(60.)

    Extract the diagonal (requires explicit form):

    >>> np.einsum('ii->i', a)
    array([ 0.,  6., 12., 18., 24.])

    Sum over an axis (requires explicit form):

    >>> np.einsum('ij->i', a)
    array([ 10.,  35.,  60.,  85., 110.])
    >>> np.sum(a, axis=1)
    array([ 10.,  35.,  60.,  85., 110.])

    For higher dimensional arrays summing a single axis can be done with ellipsis:

    >>> np.einsum('...j->...', a)
    array([ 10.,  35.,  60.,  85., 110.])

    Compute a matrix transpose, or reorder any number of axes:

    >>> np.einsum('ji', c)
    array([[0., 3.],
           [1., 4.],
           [2., 5.]])
    >>> np.einsum('ij->ji', c)
    array([[0., 3.],
           [1., 4.],
           [2., 5.]])
    >>> np.transpose(c)
    array([[0., 3.],
           [1., 4.],
           [2., 5.]])

    Vector inner products:

    >>> np.einsum('i,i', b, b)
    array(30.)

    Matrix vector multiplication:

    >>> np.einsum('ij,j', a, b)
    array([ 30.,  80., 130., 180., 230.])
    >>> np.dot(a, b)
    array([ 30.,  80., 130., 180., 230.])
    >>> np.einsum('...j,j', a, b)
    array([ 30.,  80., 130., 180., 230.])

    Broadcasting and scalar multiplication:

    >>> np.einsum('..., ...', np.array(3), c)
    array([[ 0.,  3.,  6.],
           [ 9., 12., 15.]])
    >>> np.einsum(',ij', np.array(3), c)
    array([[ 0.,  3.,  6.],
           [ 9., 12., 15.]])
    >>> np.multiply(3, c)
    array([[ 0.,  3.,  6.],
           [ 9., 12., 15.]])

    Vector outer product:

    >>> np.einsum('i,j', np.arange(2)+1, b)
    array([[0., 1., 2., 3., 4.],
           [0., 2., 4., 6., 8.]])

    Tensor contraction:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> np.einsum('ijk,jil->kl', a, b)
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    Example of ellipsis use:

    >>> a = np.arange(6).reshape((3,2))
    >>> b = np.arange(12).reshape((4,3))
    >>> np.einsum('ki,jk->ij', a, b)
    array([[10., 28., 46., 64.],
           [13., 40., 67., 94.]])
    >>> np.einsum('ki,...k->i...', a, b)
    array([[10., 28., 46., 64.],
           [13., 40., 67., 94.]])
    >>> np.einsum('k...,jk', a, b)
    array([[10., 28., 46., 64.],
           [13., 40., 67., 94.]])

    Chained array operations. For more complicated contractions, speed ups
    might be achieved by repeatedly computing a 'greedy' path. Performance
    improvements can be particularly significant with larger arrays:

    >>> a = np.ones(64).reshape(2,4,8)
    # Basic `einsum`: ~42.22ms  (benchmarked on 3.4GHz Intel Xeon.)
    >>> for iteration in range(500):
    ...     np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a)
    # Greedy `einsum` (faster optimal path approximation): ~0.117ms
    >>> for iteration in range(500):
    ...     np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=True)
    """
    # Grab non-einsum kwargs; do not optimize by default.
    optimize_arg = kwargs.pop('optimize', False)
    out = kwargs.pop('out', None)

    subscripts = operands[0]
    operands = operands[1:]
    return _npi.einsum(*operands, subscripts=subscripts, out=out, optimize=int(optimize_arg))


@set_module('mxnet.ndarray.numpy')
def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension. The values in `a` are always returned in
    row-major, C-style order.

    To group the indices by element, rather than dimension, use `argwhere`,
    which returns a row for each non-zero element.

    Parameters
    ----------
    a : ndarray
        Input array.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    See Also
    --------
    ndarray.nonzero :
        Equivalent ndarray method.

    Notes
    -----
    While the nonzero values can be obtained with ``a[nonzero(a)]``, it is
    recommended to use ``x[x.astype(bool)]`` or ``x[x != 0]`` instead, which
    will correctly handle 0-d arrays.

    Examples
    --------
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> x
    array([[3, 0, 0],
           [0, 4, 0],
           [5, 6, 0]], dtype=int32)
    >>> np.nonzero(x)
    (array([0, 1, 2, 2], dtype=int64), array([0, 1, 0, 1], dtype=int64))

    >>> x[np.nonzero(x)]
    array([3, 4, 5, 6])
    >>> np.transpose(np.stack(np.nonzero(x)))
    array([[0, 0],
           [1, 1],
           [2, 0],
           [2, 1]], dtype=int64)

    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is True.  Given an array `a`, the condition `a` > 3 is a
    boolean array and since False is interpreted as 0, np.nonzero(a > 3)
    yields the indices of the `a` where the condition is true.

    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    >>> a > 3
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> np.nonzero(a > 3)
    (array([1, 1, 1, 2, 2, 2], dtype=int64), array([0, 1, 2, 0, 1, 2], dtype=int64))

    Using this result to index `a` is equivalent to using the mask directly:

    >>> a[np.nonzero(a > 3)]
    array([4, 5, 6, 7, 8, 9], dtype=int32)
    >>> a[a > 3]
    array([4, 5, 6, 7, 8, 9], dtype=int32)

    ``nonzero`` can also be called as a method of the array.

    >>> (a > 3).nonzero()
    (array([1, 1, 1, 2, 2, 2], dtype=int64), array([0, 1, 2, 0, 1, 2], dtype=int64))
    """
    out = _npi.nonzero(a).transpose()
    return tuple([out[i] for i in range(len(out))])


@set_module('mxnet.ndarray.numpy')
def percentile(a, q, axis=None, out=None, overwrite_input=None, interpolation='linear', keepdims=False): # pylint: disable=too-many-arguments
    """
    Compute the q-th percentile of the data along the specified axis.
    Returns the q-th percentile(s) of the array elements.

    Parameters
    ----------
    a : ndarray
        Input array
    q : ndarray
        Percentile or sequence of percentiles to compute.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The default is to
        compute the percentile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the same
        shape and buffer length as the expected output, but the type (of the output)
        will be cast if necessary.
    overwrite_input : bool, optional (Not supported yet)
        If True, then allow the input array a to be modified by intermediate calculations,
        to save memory. In this case, the contents of the input a after this function
        completes is undefined.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to use when the
        desired percentile lies between two data points i < j:
        'linear': i + (j - i) * fraction, where fraction is the fractional part of the
        index surrounded by i and j.
        'lower': i.
        'higher': j.
        'nearest': i or j, whichever is nearest.
        'midpoint': (i + j) / 2.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the original array a.

    Returns
    -------
    percentile : scalar or ndarray
        Output array.

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
        [ 3,  2,  1]])
    >>> np.percentile(a, np.array(50))
    array(3.5)
    >>> np.percentile(a, np.array(50), axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.percentile(a, np.array(50), axis=1)
    array([7.,  2.])
    >>> np.percentile(a, np.array(50), axis=1, keepdims=True)
    array([[7.],
        [2.]])

    >>> m = np.percentile(a, np.array(50), axis=0)
    >>> out = np.zeros_like(m)
    >>> np.percentile(a, np.array(50), axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])
    """
    if overwrite_input is not None:
        raise NotImplementedError('overwrite_input is not supported yet')
    if isinstance(q, numeric_types):
        return _npi.percentile(a, axis=axis, interpolation=interpolation,
                               keepdims=keepdims, q_scalar=q, out=out)
    return _npi.percentile(a, q, axis=axis, interpolation=interpolation,
                           keepdims=keepdims, q_scalar=None, out=out)


@set_module('mxnet.ndarray.numpy')
def quantile(a, q, axis=None, out=None, overwrite_input=None, interpolation='linear', keepdims=False): # pylint: disable=too-many-arguments
    """
    Compute the q-th quantile of the data along the specified axis.
    New in version 1.15.0.
    Parameters
    ----------
    a : ndarray
        Input array or object that can be converted to an array.
    q : ndarray
        Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed.
        The default is to compute the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result.
        It must have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to use
        when the desired quantile lies between two data points i < j:
            linear: i + (j - i) * fraction, where fraction is the fractional part of the index surrounded by i and j.
            lower: i.
            higher: j.
            nearest: i or j, whichever is nearest.
            midpoint: (i + j) / 2.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original array a.
    Returns
    -------
    quantile : ndarray
        If q is a single quantile and axis=None, then the result is a scalar.
        If multiple quantiles are given, first axis of the result corresponds to the quantiles.
        The other axes are the axes that remain after the reduction of a.
        If out is specified, that array is returned instead.
    See also
    --------
    mean
    Notes
    -----
    Given a vector V of length N, the q-th quantile of V is the value q of the way from the minimum
    to the maximum in a sorted copy of V. The values and distances of the two nearest neighbors
    as well as the interpolation parameter will determine the quantile if the normalized ranking
    does not match the location of q exactly. This function is the same as the median if q=0.5,
    the same as the minimum if q=0.0 and the same as the maximum if q=1.0.
    This function differs from the original `numpy.quantile
    <https://numpy.org/devdocs/reference/generated/numpy.quantile.html>`_ in
    the following aspects:
    - q must be ndarray type even if it is a scalar
    - do not support overwrite_input
    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10., 7., 4.],
           [3., 2., 1.]])
    >>> q = np.array(0.5)
    >>> q
    array(0.5)
    >>> np.quantile(a, q)
    array(3.5)
    >>> np.quantile(a, q, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.quantile(a, q, axis=1)
    array([7., 2.])
    >>> np.quantile(a, q, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.quantile(a, q, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.quantile(a, q, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> out
    array([6.5, 4.5, 2.5])
    """
    if overwrite_input is not None:
        raise NotImplementedError('overwrite_input is not supported yet')
    if isinstance(q, numeric_types):
        return _npi.percentile(a, axis=axis, interpolation=interpolation,
                               keepdims=keepdims, q_scalar=q * 100, out=out)
    return _npi.percentile(a, q * 100, axis=axis, interpolation=interpolation,
                           keepdims=keepdims, q_scalar=None, out=out)


@set_module('mxnet.ndarray.numpy')
def shares_memory(a, b, max_work=None):
    """
    Determine if two arrays share memory

    Parameters
    ----------
    a, b : ndarray
        Input arrays

    Returns
    -------
    out : bool

    See Also
    --------
    may_share_memory

    Examples
    --------
    >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))
    False

    This function differs from the original `numpy.shares_memory
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.shares_memory.html>`_ in
    the following way(s):

    - Does not support `max_work`, it is a dummy argument
    - Actually it is same as `may_share_memory` in MXNet DeepNumPy
    """
    return _npi.share_memory(a, b).item()


@set_module('mxnet.ndarray.numpy')
def may_share_memory(a, b, max_work=None):
    """
    Determine if two arrays might share memory

    A return of True does not necessarily mean that the two arrays
    share any element.  It just means that they *might*.

    Only the memory bounds of a and b are checked by default.

    Parameters
    ----------
    a, b : ndarray
        Input arrays

    Returns
    -------
    out : bool

    See Also
    --------
    shares_memory

    Examples
    --------
    >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))
    False
    >>> x = np.zeros([3, 4])
    >>> np.may_share_memory(x[:,0], x[:,1])
    True

    This function differs from the original `numpy.may_share_memory
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.may_share_memory.html>`_ in
    the following way(s):

    - Does not support `max_work`, it is a dummy argument
    - Actually it is same as `shares_memory` in MXNet DeepNumPy
    """
    return _npi.share_memory(a, b).item()


@set_module('mxnet.ndarray.numpy')
def diff(a, n=1, axis=-1, prepend=None, append=None):  # pylint: disable=redefined-outer-name
    r"""
    Calculate the n-th discrete difference along the given axis.

    Parameters
    ----------
    a : ndarray
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.
    prepend, append : ndarray, optional
        Not supported yet

    Returns
    -------
    diff : ndarray
        The n-th differences.
        The shape of the output is the same as a except along axis where the dimension is smaller by n.
        The type of the output is the same as the type of the difference between any two elements of a.

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.diff(x)
    array([ 1,  2,  3, -7])
    >>> np.diff(x, n=2)
    array([  1,   1, -10])

    >>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> np.diff(x)
    array([[2, 3, 4],
        [5, 1, 2]])
    >>> np.diff(x, axis=0)
    array([[-1,  2,  0, -2]])

    Notes
    -----
    Optional inputs `prepend` and `append` are not supported yet
    """
    if (prepend or append):
        raise NotImplementedError('prepend and append options are not supported yet')
    return _npi.diff(a, n=n, axis=axis)


@set_module('mxnet.ndarray.numpy')
def ediff1d(ary, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    Parameters
    ----------
    ary : ndarray
        If necessary, will be flattened before the differences are taken.
    to_end : ndarray or scalar, optional
        Number(s) to append at the end of the returned differences.
    to_begin : ndarray or scalar, optional
        Number(s) to prepend at the beginning of the returned differences.

    Returns
    -------
    ediff1d : ndarray
        The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.

    Examples
    --------
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.ediff1d(x)
    array([ 1.,  2.,  3., -7.])

    >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
    rray([-99.,   1.,   2.,   3.,  -7.,  88.,  99.])

    The returned array is always 1D.

    >>> y = np.array([[1, 2, 4], [1, 6, 24]])
    >>> np.ediff1d(y)
    array([ 1.,  2., -3.,  5., 18.])

    >>> np.ediff1d(x, to_begin=y)
    array([ 1.,  2.,  4.,  1.,  6., 24.,  1.,  2.,  3., -7.])
    """
    from ...numpy import ndarray as np_ndarray
    input_type = (isinstance(to_begin, np_ndarray), isinstance(to_end, np_ndarray))
    # case 1: when both `to_begin` and `to_end` are arrays
    if input_type == (True, True):
        return _npi.ediff1d(ary, to_begin, to_end, to_begin_arr_given=True, to_end_arr_given=True,
                            to_begin_scalar=None, to_end_scalar=None)
    # case 2: only `to_end` is array but `to_begin` is scalar/None
    elif input_type == (False, True):
        return _npi.ediff1d(ary, to_end, to_begin_arr_given=False, to_end_arr_given=True,
                            to_begin_scalar=to_begin, to_end_scalar=None)
    # case 3: only `to_begin` is array but `to_end` is scalar/None
    elif input_type == (True, False):
        return _npi.ediff1d(ary, to_begin, to_begin_arr_given=True, to_end_arr_given=False,
                            to_begin_scalar=None, to_end_scalar=to_end)
    # case 4: both `to_begin` and `to_end` are scalar/None
    else:
        return _npi.ediff1d(ary, to_begin_arr_given=False, to_end_arr_given=False,
                            to_begin_scalar=to_begin, to_end_scalar=to_end)


@set_module('mxnet.ndarray.numpy')
def resize(a, new_shape):
    """
    Return a new array with the specified shape.
    If the new array is larger than the original array, then the new
    array is filled with repeated copies of `a`.  Note that this behavior
    is different from a.resize(new_shape) which fills with zeros instead
    of repeated copies of `a`.

    Parameters
    ----------
    a : ndarray
        Array to be resized.
    new_shape : int or tuple of int
        Shape of resized array.

    Returns
    -------
    reshaped_array : ndarray
        The new array is formed from the data in the old array, repeated
        if necessary to fill out the required number of elements.  The
        data are repeated in the order that they are stored in memory.

    See Also
    --------
    ndarray.resize : resize an array in-place.

    Notes
    -----
    Warning: This functionality does **not** consider axes separately,
    i.e. it does not apply interpolation/extrapolation.
    It fills the return array with the required number of elements, taken
    from `a` as they are laid out in memory, disregarding strides and axes.
    (This is in case the new shape is smaller. For larger, see above.)
    This functionality is therefore not suitable to resize images,
    or data where each axis represents a separate and distinct entity.

    Examples
    --------
    >>> a = np.array([[0, 1], [2, 3]])
    >>> np.resize(a, (2, 3))
    array([[0., 1., 2.],
           [3., 0., 1.]])
    >>> np.resize(a, (1, 4))
    array([[0., 1., 2., 3.]])
    >>> np.resize(a,(2, 4))
    array([[0., 1., 2., 3.],
           [0., 1., 2., 3.]])
    """
    return _npi.resize_fallback(a, new_shape=new_shape)


@set_module('mxnet.ndarray.numpy')
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None, **kwargs):
    """
    Replace NaN with zero and infinity with large finite numbers (default
    behaviour) or with the numbers defined by the user using the `nan`,
    `posinf` and/or `neginf` keywords.

    If `x` is inexact, NaN is replaced by zero or by the user defined value in
    `nan` keyword, infinity is replaced by the largest finite floating point
    values representable by ``x.dtype`` or by the user defined value in
    `posinf` keyword and -infinity is replaced by the most negative finite
    floating point values representable by ``x.dtype`` or by the user defined
    value in `neginf` keyword.

    For complex dtypes, the above is applied to each of the real and
    imaginary components of `x` separately.

    If `x` is not inexact, then no replacements are made.

    Parameters
    ----------
    x : ndarray
        Input data.
    copy : bool, optional
        Whether to create a copy of `x` (True) or to replace values
        in-place (False). The in-place operation only occurs if
        casting to an array does not require a copy.
        Default is True.
    nan : int, float, optional
        Value to be used to fill NaN values. If no value is passed
        then NaN values will be replaced with 0.0.
    posinf : int, float, optional
        Value to be used to fill positive infinity values. If no value is
        passed then positive infinity values will be replaced with a very
        large number.
    neginf : int, float, optional
        Value to be used to fill negative infinity values. If no value is
        passed then negative infinity values will be replaced with a very
        small (or negative) number.

        .. versionadded:: 1.13

    Returns
    -------
    out : ndarray
        `x`, with the non-finite values replaced. If `copy` is False, this may
        be `x` itself.

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> np.nan_to_num(np.inf)
    1.7976931348623157e+308
    >>> np.nan_to_num(-np.inf)
    -1.7976931348623157e+308
    >>> np.nan_to_num(np.nan)
    0.0
    >>> x = np.array([np.inf, -np.inf, np.nan, -128, 128])
    >>> np.nan_to_num(x)
    array([ 3.4028235e+38, -3.4028235e+38,  0.0000000e+00, -1.2800000e+02,
            1.2800000e+02])
    >>> np.nan_to_num(x, nan=-9999, posinf=33333333, neginf=33333333)
    array([ 3.3333332e+07,  3.3333332e+07, -9.9990000e+03, -1.2800000e+02,
            1.2800000e+02])
    >>> y = np.array([[-1, 0, 1],[9999,234,-14222]],dtype="float64")/0
    array([[-inf,  nan,  inf],
        [ inf,  inf, -inf]], dtype=float64)
    >>> np.nan_to_num(y)
    array([[-1.79769313e+308,  0.00000000e+000,  1.79769313e+308],
        [ 1.79769313e+308,  1.79769313e+308, -1.79769313e+308]], dtype=float64)
    >>> np.nan_to_num(y, nan=111111, posinf=222222)
    array([[-1.79769313e+308,  1.11111000e+005,  2.22222000e+005],
        [ 2.22222000e+005,  2.22222000e+005, -1.79769313e+308]], dtype=float64)
    >>> y
    array([[-inf,  nan,  inf],
       [ inf,  inf, -inf]], dtype=float64)
    >>> np.nan_to_num(y, copy=False, nan=111111, posinf=222222)
    array([[-1.79769313e+308,  1.11111000e+005,  2.22222000e+005],
       [ 2.22222000e+005,  2.22222000e+005, -1.79769313e+308]], dtype=float64)
    >>> y
    array([[-1.79769313e+308,  1.11111000e+005,  2.22222000e+005],
       [ 2.22222000e+005,  2.22222000e+005, -1.79769313e+308]], dtype=float64)
    """
    if isinstance(x, numeric_types):
        return _np.nan_to_num(x, copy, nan, posinf, neginf)
    elif isinstance(x, NDArray):
        if x.dtype in ['int8', 'uint8', 'int32', 'int64']:
            return x
        if not copy:
            return _npi.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=x)
        return _npi.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=None)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def isnan(x, out=None, **kwargs):
    """
    Test element-wise for NaN and return result as a boolean array.

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or bool
        True where x is NaN, false otherwise.
        This is a scalar if x is a scalar.

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).

    This function differs from the original `numpy.isinf
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.isnan.html>`_ in
    the following aspects:
    - Does not support complex number for now
    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    Examples
    --------
    >>> np.isnan(np.nan)
    True
    >>> np.isnan(np.inf)
    False
    >>> np.isnan(np.array([np.log(-1.),1.,np.log(0)]))
    array([ True, False, False])
    """
    return _unary_func_helper(x, _npi.isnan, _np.isnan, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def isinf(x, out=None, **kwargs):
    """
    Test element-wise for positive or negative infinity.

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or bool
        True where x is positive or negative infinity, false otherwise.
        This is a scalar if x is a scalar.

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
    This means that Not a Number is not equivalent to infinity.

    This function differs from the original `numpy.isnan
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.isnan.html>`_ in
    the following aspects:
    - Does not support complex number for now
    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    Examples
    --------
    >>> np.isinf(np.inf)
    True
    >>> np.isinf(np.nan)
    False
    >>> np.isinf(np.array([np.inf, -np.inf, 1.0, np.nan]))
    array([ True,  True, False, False])
    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([True, True, True], dtype=np.bool_)
    >>> np.isinf(x, y)
    array([ True, False,  True])
    >>> y
    array([ True, False,  True])
    """
    return _unary_func_helper(x, _npi.isinf, _np.isinf, out=out, **kwargs)


@wrap_np_unary_func
def isposinf(x, out=None, **kwargs):
    """
    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or bool
        True where x is positive infinity, false otherwise.
        This is a scalar if x is a scalar.

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
    This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> np.isposinf(np.inf)
    True
    >>> np.isposinf(-np.inf)
    False
    >>> np.isposinf(np.nan)
    False
    >>> np.isposinf(np.array([-np.inf, 0., np.inf]))
    array([False, False,  True])
    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([True, True, True], dtype=np.bool)
    >>> np.isposinf(x, y)
    array([False, False,  True])
    >>> y
    array([False, False,  True])
    """
    return _unary_func_helper(x, _npi.isposinf, _np.isposinf, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def isneginf(x, out=None, **kwargs):
    """
    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or bool
        True where x is negative infinity, false otherwise.
        This is a scalar if x is a scalar.

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
    This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> np.isneginf(-np.inf)
    True
    >>> np.isneginf(np.inf)
    False
    >>> np.isneginf(float('-inf'))
    True
    >>> np.isneginf(np.array([-np.inf, 0., np.inf]))
    array([ True, False, False])
    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([True, True, True], dtype=np.bool)
    >>> np.isneginf(x, y)
    array([ True, False, False])
    >>> y
    array([ True, False, False])
    """
    return _unary_func_helper(x, _npi.isneginf, _np.isneginf, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
@wrap_np_unary_func
def isfinite(x, out=None, **kwargs):
    """
    Test element-wise for finiteness (not infinity or not Not a Number).

    Parameters
    ----------
    x : ndarray
        Input array.
    out : ndarray or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : ndarray or bool
        True where x is negative infinity, false otherwise.
        This is a scalar if x is a scalar.

    Notes
    -----
    Not a Number, positive infinity and negative infinity are considered to be non-finite.

    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
    This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity.
    But infinity is equivalent to positive infinity. Errors result if the second argument
    is also supplied when x is a scalar input, or if first and second arguments have different shapes.

    Examples
    --------
    >>> np.isfinite(1)
    True
    >>> np.isfinite(0)
    True
    >>> np.isfinite(np.nan)
    False
    >>> np.isfinite(np.inf)
    False
    >>> np.isfinite(-np.inf)
    False
    >>> np.isfinite(np.array([np.log(-1.),1.,np.log(0)]))
    array([False,  True, False])
    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([True, True, True], dtype=np.bool)
    >>> np.isfinite(x, y)
    array([False,  True, False])
    >>> y
    array([False,  True, False])
    """
    return _unary_func_helper(x, _npi.isfinite, _np.isfinite, out=out, **kwargs)


@set_module('mxnet.ndarray.numpy')
def where(condition, x=None, y=None):  # pylint: disable=too-many-return-statements
    """where(condition, [x, y])
    Return elements chosen from `x` or `y` depending on `condition`.

    .. note::
        When only `condition` is provided, this function is a shorthand for
        ``np.asarray(condition).nonzero()``. The rest of this documentation
        covers only the case where all three arguments are provided.

    Parameters
    ----------
    condition : ndarray
        Where True, yield `x`, otherwise yield `y`.
    x, y : ndarray
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape. `x` and `y` must have the same dtype.

    Returns
    -------
    out : ndarray
        An array with elements from `x` where `condition` is True, and elements
        from `y` elsewhere.

    Notes
    -----
    If all the arrays are 1-D, `where` is equivalent to::

        [xv if c else yv
        for c, xv, yv in zip(condition, x, y)]

    This function differs from the original `numpy.where
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html>`_ in
    the following way(s):

    - If `condition` is a scalar, this operator returns x or y directly without broadcasting.
    - If `condition` is ndarray, while both `x` and `y` are scalars,
        the output dtype will be `float32`.

    Examples
    --------
    >>> a = np.arange(10)
    >>> a
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> np.where(a < 5, a, 10*a)
    array([ 0.,  1.,  2.,  3.,  4., 50., 60., 70., 80., 90.])

    This can be used on multidimensional arrays too:

    >>> cond = np.array([[True, False], [True, True]])
    >>> x = np.array([[1, 2], [3, 4]])
    >>> y = np.array([[9, 8], [7, 6]])
    >>> np.where(cond, x, y)
    array([[1., 8.],
           [3., 4.]])

    The shapes of x, y, and the condition are broadcast together:

    >>> x, y = onp.ogrid[:3, :4]
    >>> x = np.array(x)
    >>> y = np.array(y)
    >>> np.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
    array([[10,  0,  0,  0],
           [10, 11,  1,  1],
           [10, 11, 12,  2]], dtype=int64)

    >>> a = np.array([[0, 1, 2],
    ...               [0, 2, 4],
    ...               [0, 3, 6]])
    >>> np.where(a < 4, a, -1)  # -1 is broadcast
    array([[ 0.,  1.,  2.],
           [ 0.,  2., -1.],
           [ 0.,  3., -1.]])
    """
    if x is None and y is None:
        return nonzero(condition)
    else:
        if isinstance(condition, numeric_types):
            if condition != 0:
                return x
            else:
                return y
        else:
            if isinstance(x, numeric_types) and isinstance(y, numeric_types):
                return _npi.where_scalar2(condition, float(x), float(y), out=None)
            elif isinstance(x, NDArray) and isinstance(y, NDArray):
                return _npi.where(condition, x, y, out=None)
            elif isinstance(y, NDArray):
                return _npi.where_lscalar(condition, y, float(x), out=None)
            elif isinstance(x, NDArray):
                return _npi.where_rscalar(condition, x, float(y), out=None)
            else:
                raise TypeError('type {0} and {1} not supported'.format(str(type(x)), str(type(y))))


@set_module('mxnet.ndarray.numpy')
def polyval(p, x):
    """
    Evaluate a polynomial at specific values.
    If p is of length N, this function returns the value:
    p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
    If x is a sequence, then p(x) is returned for each element of x.
    If x is another polynomial then the composite polynomial p(x(t)) is returned.

    Parameters
    ----------
    p : ndarray
        1D array of polynomial coefficients (including coefficients equal to zero)
        from highest degree to the constant term.
    x : ndarray
        An array of numbers, at which to evaluate p.

    Returns
    -------
    values : ndarray
        Result array of polynomials

    Notes
    -----
    This function differs from the original `numpy.polyval
    <https://numpy.org/devdocs/reference/generated/numpy.polyval.html>`_ in
    the following way(s):
    - Does not support poly1d.
    - X should be ndarray type even if it contains only one element.

    Examples
    --------
    >>> p = np.array([3, 0, 1])
    array([3., 0., 1.])
    >>> x = np.array([5])
    array([5.])
    >>> np.polyval(p, x)  # 3 * 5**2 + 0 * 5**1 + 1
    array([76.])
    >>> x = np.array([5, 4])
    array([5., 4.])
    >>> np.polyval(p, x)
    array([76., 49.])
    """
    from ...numpy import ndarray
    if isinstance(p, ndarray) and isinstance(x, ndarray):
        return _npi.polyval(p, x)
    elif not isinstance(p, ndarray) and not isinstance(x, ndarray):
        return _np.polyval(p, x)
    else:
        raise TypeError('type not supported')


@set_module('mxnet.ndarray.numpy')
def bincount(x, weights=None, minlength=0):
    """
    Count number of occurrences of each value in array of non-negative ints.

    Parameters
    ----------
    x : ndarray
        input array, 1 dimension, nonnegative ints.
    weights: ndarray
        input weigths same shape as x. (Optional)
    minlength: int
        A minimum number of bins for the output. (Optional)

    Returns
    --------
    out : ndarray
        the result of binning the input array. The length of out is equal to amax(x)+1.

    Raises
    --------
    Value Error
        If the input is not 1-dimensional, or contains elements with negative values,
        or if minlength is negative
    TypeError
        If the type of the input is float or complex.

    Examples
    --------
    >>> np.bincount(np.arange(5))
    array([1, 1, 1, 1, 1])
    >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
    array([1, 3, 1, 1, 0, 0, 0, 1])

    >>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
    >>> np.bincount(x).size == np.amax(x)+1
    True

    >>> np.bincount(np.arange(5, dtype=float))
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    TypeError: array cannot be safely cast to required type

    >>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
    >>> x = np.array([0, 1, 1, 2, 2, 2])
    >>> np.bincount(x,  weights=w)
    array([ 0.3,  0.7,  1.1])
    """
    if not isinstance(x, NDArray):
        raise TypeError("Input data should be NDarray")
    if minlength < 0:
        raise ValueError("Minlength value should greater than 0")
    if weights is None:
        return _npi.bincount(x, minlength=minlength, has_weights=False)
    return _npi.bincount(x, weights=weights, minlength=minlength, has_weights=True)


@set_module('mxnet.ndarray.numpy')
def pad(x, pad_width, mode='constant', **kwargs): # pylint: disable=too-many-arguments
    """
    Pad an array.

    Parameters
    ----------
    array : array_like of rank N
        The array to pad.
    pad_width : {sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ((before_1, after_1), ... (before_N, after_N)) unique pad widths
        for each axis.
        ((before, after),) yields same before and after pad for each axis.
        (pad,) or int is a shortcut for before = after = pad width for all
        axes.
    mode : str or function, optional
        One of the following string values or a user supplied function.
        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            not supported yet
        'maximum'
            Pads with the maximum value of all of the
            vector along each axis.
        'mean'
            not supported yet
        'median'
            not supported yet
        'minimum'
            Pads with the minimum value of all of the
            vector along each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        'wrap'
            not supported yet.
        'empty'
            not supported yet.
        <function>
            not supported yet.
    stat_length : not supported yet
    constant_values : scalar, optional
        Used in 'constant'.  The values to set the padded values for each
        axis.
        Default is 0.

    end_values : not supported yet
    reflect_type : {'even', 'odd'}, optional
        only support even now

    Returns
    -------
    pad : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `pad_width`.
    """
    # pylint: disable = too-many-return-statements, inconsistent-return-statements
    if not _np.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('`pad_width` must be of integral type.')
    if not isinstance(pad_width, tuple):
        raise TypeError("`pad_width` must be tuple.")
    if mode == "linear_ramp":
        raise ValueError("mode {'linear_ramp'} is not supported.")
    if mode == "wrap":
        raise ValueError("mode {'wrap'} is not supported.")
    if mode == "median":
        raise ValueError("mode {'median'} is not supported.")
    if mode == "mean":
        raise ValueError("mode {'mean'} is not supported.")
    if mode == "empty":
        raise ValueError("mode {'empty'} is not supported.")
    if callable(mode):
        raise ValueError("mode {'<function>'} is not supported.")

    allowedkwargs = {
        'constant': ['constant_values'],
        'edge': [],
        'linear_ramp': ['end_values'],
        'maximum': ['stat_length'],
        'mean': ['stat_length'],
        'median': ['stat_length'],
        'minimum': ['stat_length'],
        'reflect': ['reflect_type'],
        'symmetric': ['reflect_type'],
        'wrap': [],
        }

    if isinstance(mode, _np.compat.basestring):
        # Make sure have allowed kwargs appropriate for mode
        for key in kwargs:
            if key not in allowedkwargs[mode]:
                raise ValueError('%s keyword not in allowed keywords %s' %(key, allowedkwargs[mode]))

    unsupported_kwargs = set(kwargs) - set(allowedkwargs[mode])
    if unsupported_kwargs:
        raise ValueError("unsupported keyword arguments for mode '{}': {}"
                         .format(mode, unsupported_kwargs))
    if mode == "constant":
        values = kwargs.get("constant_values", 0)
        if isinstance(values, tuple):
            raise TypeError("unsupported constant_values type: {'tuple'}.")
        _npi.pad(x, pad_width, mode='constant', constant_value=values)
    elif mode == "symmetric":
        values = kwargs.get("reflect_type", "even")
        if values != "even" and values is not None:
            raise ValueError("unsupported reflect_type '{}'".format(values))
        return _npi.pad(x, pad_width, mode='symmetric', reflect_type="even")
    elif mode == "edge":
        return _npi.pad(x, pad_width, mode='edge')
    elif mode == "reflect":
        values = kwargs.get("reflect_type", "even")
        if values != "even" and values is not None:
            raise ValueError("unsupported reflect_type '{}'".format(values))
        return _npi.pad(x, pad_width, mode='reflect', reflect_type="even")
    elif mode == "maximum":
        values = kwargs.get("stat_length", None)
        if values is not None:
            raise ValueError("unsupported stat_length '{}'".format(values))
        return _npi.pad(x, pad_width, mode='maximum')
    elif mode == "minimum":
        values = kwargs.get("stat_length", None)
        if values is not None:
            raise ValueError("unsupported stat_length '{}'".format(values))
        return _npi.pad(x, pad_width, mode='minimum')
    return _npi.pad(x, pad_width, mode='constant', constant_value=0)
