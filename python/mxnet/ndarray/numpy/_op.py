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

"""Namespace for numpy operators used in Gluon dispatched by F=ndarray."""

from __future__ import absolute_import
import numpy as _np
from ...base import numeric_types
from ...util import set_module
from ...context import current_context
from . import _internal as _npi
from ..ndarray import NDArray

__all__ = ['zeros', 'ones', 'add', 'subtract', 'multiply', 'divide', 'mod', 'power', 'tensordot',
           'linspace', 'expand_dims', 'tile', 'arange', 'split', 'concatenate']


@set_module('mxnet.ndarray.numpy')
def zeros(shape, dtype=_np.float32, order='C', ctx=None):
    """Return a new array of given shape and type, filled with zeros.
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
        Array of zeros with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.zeros(shape=shape, ctx=ctx, dtype=dtype)


@set_module('mxnet.ndarray.numpy')
def ones(shape, dtype=_np.float32, order='C', ctx=None):
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
        Array of zeros with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.ones(shape=shape, ctx=ctx, dtype=dtype)


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
    elif isinstance(rhs, ndarray):
        return fn_array(lhs, rhs, out=out)
    else:
        raise TypeError('type {} not supported'.format(str(type(rhs))))
#pylint: enable= too-many-arguments, no-member, protected-access


@set_module('mxnet.ndarray.numpy')
def add(x1, x2, out=None):
    """Add arguments element-wise.

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
    """
    return _ufunc_helper(x1, x2, _npi.add, _np.add, _npi.add_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def subtract(x1, x2, out=None):
    """Subtract arguments element-wise.

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
    """
    return _ufunc_helper(x1, x2, _npi.subtract, _np.subtract, _npi.subtract_scalar,
                         _npi.rsubtract_scalar, out)


@set_module('mxnet.ndarray.numpy')
def multiply(x1, x2, out=None):
    """Multiply arguments element-wise.

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
    """
    return _ufunc_helper(x1, x2, _npi.multiply, _np.multiply, _npi.multiply_scalar, None, out)


@set_module('mxnet.ndarray.numpy')
def divide(x1, x2, out=None):
    """Returns a true division of the inputs, element-wise.

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
    return _ufunc_helper(x1, x2, _npi.true_divide, _np.divide, _npi.true_divide_scalar,
                         _npi.rtrue_divide_scalar, out)


@set_module('mxnet.ndarray.numpy')
def mod(x1, x2, out=None):
    """Return element-wise remainder of division.

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
def power(x1, x2, out=None):
    """First array elements raised to powers from second array, element-wise.

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
    if _np.isscalar(axes):
        return _npi.tensordot_int_axes(a, b, axes)

    if len(axes) != 2:
        raise ValueError('Axes must consist of two arrays.')
    a_axes_summed, b_axes_summed = axes
    if _np.isscalar(a_axes_summed):
        a_axes_summed = (a_axes_summed,)
    if _np.isscalar(b_axes_summed):
        b_axes_summed = (b_axes_summed,)

    if len(a_axes_summed) != len(b_axes_summed):
        raise ValueError('Axes length mismatch')

    return _npi.tensordot(a, b, a_axes_summed, b_axes_summed)


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


@set_module('mxnet.ndarray.numpy')
def split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.
    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
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
    indices = []
    axis_size = ary.shape[axis]
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if axis_size % sections:
            raise ValueError('array split does not result in an equal division')
        section_size = int(axis_size / sections)
        indices = [i * section_size for i in range(sections)]
    elif isinstance(indices_or_sections, tuple):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple of ints')
    ret = _npi.split(ary, indices, axis, False)
    if not isinstance(ret, list):
        return [ret]
    return ret


@set_module('mxnet.ndarray.numpy')
def concatenate(seq, axis=0, out=None):
    """Join a sequence of arrays along an existing axis.
    Parameters
    ----------
    a1, a2, ... : sequence of array_like
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
    """
    return _npi.concatenate(*seq, dim=axis, out=out)
