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

# pylint: disable=too-many-lines
"""numpy namespace for operators used in Gluon APIs dispatched by F=symbol module."""

from __future__ import absolute_import
import ctypes
import numpy as _np
from . import _op as _mx_np_op
from ...base import _LIB, SymbolHandle, numeric_types, mx_uint
from ...util import _sanity_check_params, check_call, set_module
from ...context import current_context
from ..symbol import Symbol
from .._internal import _set_np_symbol_class
from . import _internal as _npi

__all__ = ['zeros', 'ones', 'maximum', 'minimum', 'stack', 'concatenate', 'arange', 'argmax',
           'clip', 'add', 'subtract', 'multiply', 'divide', 'mod', 'power', 'split', 'swapaxes',
           'expand_dims', 'tile', 'linspace', 'eye', 'sin', 'cos', 'sinh', 'cosh', 'log10', 'sqrt',
           'abs', 'exp', 'arctan', 'sign', 'log', 'degrees', 'log2', 'rint', 'radians', 'mean',
           'reciprocal', 'square', 'arcsin', 'argsort']


def _num_outputs(sym):
    return len(sym.as_nd_ndarray())


@set_module('mxnet.symbol.numpy')
class _Symbol(Symbol):
    def __getitem__(self, key):
        num_outputs = _num_outputs(self)
        if num_outputs == 1:
            raise NotImplementedError
        if not isinstance(key, int):
            raise NotImplementedError
        if key >= num_outputs:
            # Important, python determines the end by this exception
            raise IndexError
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolGetOutput(
            self.handle, mx_uint(key), ctypes.byref(handle)))
        return _Symbol(handle=handle)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __iter__(self):
        raise AttributeError('_Symbol object has no attribute __iter__')

    def __add__(self, other):
        """x.__add__(y) <=> x + y"""
        return add(self, other)

    def __sub__(self, other):
        """x.__sub__(y) <=> x - y"""
        return subtract(self, other)

    def __rsub__(self, other):
        """x.__rsub__(y) <=> y - x"""
        return subtract(other, self)

    def __mul__(self, other):
        """x.__mul__(y) <=> x * y"""
        return multiply(self, other)

    def __rmul__(self, other):
        """x.__rmul__(y) <=> y * x"""
        return multiply(other, self)

    def __div__(self, other):
        raise AttributeError('_Symbol.__div__ is replaced by __truediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __rdiv__(self, other):
        raise AttributeError('_Symbol.__rdiv__ is replaced by __rtruediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __mod__(self, other):
        """x.__mod__(y) <=> x % y"""
        return mod(self, other)

    def __rmod__(self, other):
        """x.__rmod__(y) <=> y % x"""
        return mod(other, self)

    def __idiv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        """x.__truediv__(y) <=> x / y"""
        return divide(self, other)

    def __rtruediv__(self, other):
        """x.__rtruediv__(y) <=> y / x"""
        return divide(other, self)

    def __itruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        """x.__pow__(y) <=> x ** y"""
        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    def __neg__(self):
        """x.__neg__() <=> - x"""
        return self.__mul__(-1.0)

    def __deepcopy__(self, _):
        return super(_Symbol, self).as_np_ndarray()

    def __eq__(self, other):
        """x.__eq__(y) <=> x == y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.equal_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __ne__(self, other):
        """x.__ne__(y) <=> x != y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.not_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.not_equal_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __gt__(self, other):
        """x.__gt__(y) <=> x > y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.greater(self, other)
        elif isinstance(other, numeric_types):
            return _npi.greater_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __ge__(self, other):
        """x.__ge__(y) <=> x >= y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.greater_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.greater_equal_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __lt__(self, other):
        """x.__lt__(y) <=> x < y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.less(self, other)
        elif isinstance(other, numeric_types):
            return _npi.less_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __le__(self, other):
        """x.__le__(y) <=> x <= y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, _Symbol):
            return _npi.less_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.less_equal_scalar(self, float(other))
        else:
            raise TypeError("_Symbol does not support type {} as operand".format(str(type(other))))

    def __len__(self):
        raise NotImplementedError

    def as_nd_ndarray(self):
        """Convert _Symbol to mxnet.symbol.Symbol to use its convenience fluent methods."""
        hdl = SymbolHandle()
        check_call(_LIB.MXShallowCopySymbol(self.handle, ctypes.byref(hdl)))
        return Symbol(handle=hdl)

    def as_np_ndarray(self):
        """For the convenience of conversion between legacy and np symbols."""
        return self

    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        """Same as self.transpose()."""
        return self.transpose()
    # pylint: enable= invalid-name, undefined-variable

    def astype(self, dtype, **kwargs):  # pylint: disable=arguments-differ
        raise NotImplementedError

    def dot(self, b, out=None):
        return _mx_np_op.dot(self, b, out=out)

    def reshape(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Returns an array containing the same data with a new shape.

        Notes
        -----
        Unlike the free function `numpy.reshape`, this method on `ndarray` allows
        the elements of the shape parameter to be passed in as separate arguments.
        For example, ``a.reshape(10, 11)`` is equivalent to
        ``a.reshape((10, 11))``.
        """
        order = 'C'
        if len(kwargs) > 1:
            raise TypeError('function takes at most 1 keyword argument')
        if len(kwargs) == 1:
            if 'order' not in kwargs:
                raise TypeError('{} is an invalid keyword argument for this function'
                                .format(kwargs.keys()[0]))
            order = kwargs.pop('order', 'C')
            if order != 'C':
                raise NotImplementedError('only supports C-order,'
                                          ' while received {}'.format(order))
        if len(args) == 0:
            raise TypeError('reshape() takes exactly 1 argument (0 given)')
        if len(args) == 1 and isinstance(args[0], tuple):
            return _mx_np_op.reshape(self, newshape=args[0], order=order)
        else:
            return _mx_np_op.reshape(self, newshape=args, order=order)

    def argmax(self, axis=None, out=None):  # pylint: disable=arguments-differ
        return _mx_np_op.argmax(self, axis, out)

    def reshape_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reshape_like`.

        The arguments are the same as for :py:func:`reshape_like`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute reshape_like')

    def zeros_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`zeros_like`.

        The arguments are the same as for :py:func:`zeros_like`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute zeros_like')

    def ones_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ones_like`.

        The arguments are the same as for :py:func:`ones_like`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute ones_like')

    def broadcast_axes(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`broadcast_axes`.

        The arguments are the same as for :py:func:`broadcast_axes`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute broadcast_like')

    def repeat(self, repeats, axis=None):  # pylint: disable=arguments-differ
        """Repeat elements of an array."""
        return _mx_np_op.repeat(self, repeats=repeats, axis=axis)

    def pad(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pad`.

        The arguments are the same as for :py:func:`pad`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute pad')

    def swapaxes(self, axis1, axis2):  # pylint: disable=arguments-differ
        """Return a copy of the array with axis1 and axis2 interchanged.
        Refer to `mxnet.numpy.swapaxes` for full documentation.
        """
        return swapaxes(self, axis1, axis2)

    def split(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split`.

        The arguments are the same as for :py:func:`split`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute split')

    def split_v2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split_v2`.

        The arguments are the same as for :py:func:`split_v2`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute split_v2')

    def slice(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice`.

        The arguments are the same as for :py:func:`slice`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute slice')

    def slice_axis(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_axis`.

        The arguments are the same as for :py:func:`slice_axis`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute slice_axis')

    def slice_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_like`.

        The arguments are the same as for :py:func:`slice_like`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute slice_like')

    def take(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`take`.

        The arguments are the same as for :py:func:`take`, with
        this array as data.
        """
        raise NotImplementedError

    def one_hot(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`one_hot`.

        The arguments are the same as for :py:func:`one_hot`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute one_hot')

    def pick(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pick`.

        The arguments are the same as for :py:func:`pick`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute pick')

    def sort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sort`.

        The arguments are the same as for :py:func:`sort`, with
        this array as data.
        """
        raise NotImplementedError

    def topk(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`topk`.

        The arguments are the same as for :py:func:`topk`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute topk')

    def argsort(self, axis=-1, kind='quicksort', order=None):   # pylint: disable=arguments-differ
        """
        Returns the indices that would sort an input array along the given axis.
        This function performs sorting along the given axis and returns an array
        of indices having same shape as an input array that index data in sorted order.

        Parameters
        ----------
        a : _Symbol
            Input array
        axis : int, optional
            The axis along which to sort teh input tensor.
            If not given, the last, dimension -1 will be used by default.
            If None, the flattened array is used.
        kind: {'quicksort'}
            Currently not supported.
        order: None
            Currently not supported.

        Returns
        -------
        output : ndarray
        Array of indices that sort a along the specified axis.
        If a is one-dimensional, a[index_array] yields a sorted a.
        More generally, np.take_along_axis(a, index_array, axis=a) always yields the sorted a,
        irrespective of dimensionality.

        Examples
        --------
        >>> x = np.array([3, 1, 2])
        >>> np.argsort(x)
        array([1., 2., 0.])
        >>> x = np.array([[0, 3], [2, 2]])
        >>> x
        array([[0., 3.],
            [2., 2.]])
        >>> np.argsort(x, axis=0)  # sorts along first axis (down)
        array([[0., 1.],
            [1., 0.]])
        >>> np.argsort(x, axis=1)  # sorts along last axis (across)
        array([[0., 1.],
            [0., 1.]])

        Notes
        -----
        This function differs from the original `numpy.mean
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html>`_ in
        the following way(s):

        - kind and order are currently not supported
        """
        if kind != 'quicksort':
            raise AttributeError('mxnet.numpy.argsort does not support other sorting methods')
        if order is not None:
            raise AttributeError('mxnet.numpy.argsort does not support sorting with fields ordering')
        return _npi.argsort(self, axis)

    def argmax_channel(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax_channel`.

        The arguments are the same as for :py:func:`argmax_channel`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute argmax_channel')

    def argmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmin`.

        The arguments are the same as for :py:func:`argmin`, with
        this array as data.
        """
        raise NotImplementedError

    def clip(self, min=None, max=None, out=None):  # pylint: disable=arguments-differ
        """Return an array whose values are limited to [min, max].
        One of max or min must be given.
        """
        return clip(self, min, max, out=out)

    def abs(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`abs`.

        The arguments are the same as for :py:func:`abs`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute abs')

    def sign(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sign`.

        The arguments are the same as for :py:func:`sign`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute abs')

    def flatten(self, order='C'):  # pylint: disable=arguments-differ
        """Return a copy of the array collapsed into one dimension."""
        return self.reshape(-1, order=order)

    def shape_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`shape_array`.

        The arguments are the same as for :py:func:`shape_array`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute shape_array')

    def size_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`size_array`.

        The arguments are the same as for :py:func:`size_array`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute size_array')

    def expand_dims(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expand_dims`.

        The arguments are the same as for :py:func:`expand_dims`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute expand_dims')

    def tile(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tile`.

        The arguments are the same as for :py:func:`tile`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute tile')

    def transpose(self, *axes):  # pylint: disable=arguments-differ
        """Convenience fluent method for :py:func:`transpose`.

        The arguments are the same as for :py:func:`transpose`, with
        this array as data.
        """
        return _mx_np_op.transpose(self, axes=axes if len(axes) != 0 else None)

    def flip(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`flip`.

        The arguments are the same as for :py:func:`flip`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute flip')

    def depth_to_space(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`depth_to_space`.

        The arguments are the same as for :py:func:`depth_to_space`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute depth_to_space')

    def space_to_depth(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`space_to_depth`.

        The arguments are the same as for :py:func:`space_to_depth`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute space_to_depth')

    def diag(self, k=0, **kwargs):
        """Convenience fluent method for :py:func:`diag`.

        The arguments are the same as for :py:func:`diag`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute diag')

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Convenience fluent method for :py:func:`sum`.

        The arguments are the same as for :py:func:`sum`, with
        this array as data.
        """
        return _mx_np_op.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def nansum(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nansum`.

        The arguments are the same as for :py:func:`nansum`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute nansum')

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the product of the array elements over the given axis."""
        return _mx_np_op.prod(self, axis=axis, dtype=dtype, keepdims=keepdims, out=out)

    def nanprod(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nanprod`.

        The arguments are the same as for :py:func:`nanprod`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute nanprod')

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Convenience fluent method for :py:func:`mean`.

        The arguments are the same as for :py:func:`mean`, with
        this array as data.
        """
        return _npi.mean(self, axis=axis, dtype=dtype, keepdims=keepdims, out=out)

    def cumsum(self, axis=None, dtype=None, out=None):
        """Return the cumulative sum of the elements along the given axis."""
        return _mx_np_op.cumsum(self, axis=axis, dtype=dtype, out=out)

    def max(self, axis=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the maximum along a given axis."""
        return _mx_np_op.max(self, axis=axis, keepdims=keepdims, out=out)

    def min(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`min`.

        The arguments are the same as for :py:func:`min`, with
        this array as data.
        """
        raise NotImplementedError

    def norm(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`norm`.

        The arguments are the same as for :py:func:`norm`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute norm')

    def round(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`round`.

        The arguments are the same as for :py:func:`round`, with
        this array as data.
        """
        raise NotImplementedError

    def rint(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rint`.

        The arguments are the same as for :py:func:`rint`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute rint')

    def fix(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`fix`.

        The arguments are the same as for :py:func:`fix`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute fix')

    def floor(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`floor`.

        The arguments are the same as for :py:func:`floor`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute floor')

    def ceil(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ceil`.

        The arguments are the same as for :py:func:`ceil`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute ceil')

    def trunc(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`trunc`.

        The arguments are the same as for :py:func:`trunc`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute trunc')

    def sin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sin`.

        The arguments are the same as for :py:func:`sin`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute sin')

    def cos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cos`.

        The arguments are the same as for :py:func:`cos`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute cos')

    def tan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tan`.

        The arguments are the same as for :py:func:`tan`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute tan')

    def arcsin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsin`.

        The arguments are the same as for :py:func:`arcsin`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arcsin')

    def arccos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccos`.

        The arguments are the same as for :py:func:`arccos`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arccos')

    def arctan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctan`.

        The arguments are the same as for :py:func:`arctan`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arctan')

    def degrees(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`degrees`.

        The arguments are the same as for :py:func:`degrees`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute degrees')

    def radians(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`radians`.

        The arguments are the same as for :py:func:`radians`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute radians')

    def sinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sinh`.

        The arguments are the same as for :py:func:`sinh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute sinh')

    def cosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cosh`.

        The arguments are the same as for :py:func:`cosh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute cosh')

    def tanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tanh`.

        The arguments are the same as for :py:func:`tanh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute tanh')

    def arcsinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsinh`.

        The arguments are the same as for :py:func:`arcsinh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arcsinh')

    def arccosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccosh`.

        The arguments are the same as for :py:func:`arccosh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arccosh')

    def arctanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctanh`.

        The arguments are the same as for :py:func:`arctanh`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute arctanh')

    def exp(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`exp`.

        The arguments are the same as for :py:func:`exp`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute exp')

    def expm1(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expm1`.

        The arguments are the same as for :py:func:`expm1`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute expm1')

    def log(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log`.

        The arguments are the same as for :py:func:`log`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log')

    def log10(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log10`.

        The arguments are the same as for :py:func:`log10`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log10')

    def log2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log2`.

        The arguments are the same as for :py:func:`log2`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log2')

    def log1p(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log1p`.

        The arguments are the same as for :py:func:`log1p`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log1p')

    def sqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sqrt`.

        The arguments are the same as for :py:func:`sqrt`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute sqrt')

    def rsqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rsqrt`.

        The arguments are the same as for :py:func:`rsqrt`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute rsqrt')

    def cbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cbrt`.

        The arguments are the same as for :py:func:`cbrt`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute cqrt')

    def rcbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rcbrt`.

        The arguments are the same as for :py:func:`rcbrt`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute rcqrt')

    def square(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`square`.

        The arguments are the same as for :py:func:`square`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute square')

    def reciprocal(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reciprocal`.

        The arguments are the same as for :py:func:`reciprocal`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute reciprocal')

    def relu(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`relu`.

        The arguments are the same as for :py:func:`relu`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute relu')

    def sigmoid(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sigmoid`.

        The arguments are the same as for :py:func:`sigmoid`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute sigmoid')

    def softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmax`.

        The arguments are the same as for :py:func:`softmax`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute softmax')

    def log_softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log_softmax`.

        The arguments are the same as for :py:func:`log_softmax`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute log_softmax')

    def softmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmin`.

        The arguments are the same as for :py:func:`softmin`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute softmin')

    def squeeze(self, axis=None):  # pylint: disable=arguments-differ
        """Remove single-dimensional entries from the shape of a.
        """
        return _mx_np_op.squeeze(self, axis=axis)

    def broadcast_to(self, *args, **kwargs):
        raise AttributeError('_Symbol object has no attribute broadcast_to')

    def broadcast_like(self, *args, **kwargs):
        raise AttributeError('_Symbol object has no attribute broadcast_like')


@set_module('mxnet.symbol.numpy')
def zeros(shape, dtype=_np.float32, **kwargs):
    """Return a new array of given shape and type, filled with zeros.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type. Default is `numpy.float32`. Note that this
        behavior is different from NumPy's `zeros` function  where `float64`
        is the default value, because `float32` is considered as the default
        data type in deep learning.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : Symbol
        Array of zeros with the given shape, dtype, and ctx.
    """
    _sanity_check_params('zeros', ['order'], kwargs)
    ctx = kwargs.get('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.zeros(shape=shape, ctx=ctx, dtype=dtype, **kwargs)


@set_module('mxnet.symbol.numpy')
def ones(shape, dtype=None, **kwargs):
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
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and ctx.
    """
    _sanity_check_params('zeros', ['order'], kwargs)
    ctx = kwargs.get('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    dtype = _np.float32 if dtype is None else dtype
    return _npi.ones(shape=shape, ctx=ctx, dtype=dtype, **kwargs)


#pylint: disable= too-many-arguments, no-member, protected-access
def _ufunc_helper(lhs, rhs, fn_array, fn_scalar, lfn_scalar, rfn_scalar=None, out=None):
    """ Helper function for element-wise operation.
    The function will perform numpy-like broadcasting if needed and call different functions.

    Parameters
    --------
    lhs : Symbol or numeric value
        Left-hand side operand.

    rhs : Symbol or numeric value
        Right-hand operand,

    fn_array : function
        Function to be called if both lhs and rhs are of ``Symbol`` type.

    fn_scalar : function
        Function to be called if both lhs and rhs are numeric values.

    lfn_scalar : function
        Function to be called if lhs is ``Symbol`` while rhs is numeric value

    rfn_scalar : function
        Function to be called if lhs is numeric value while rhs is ``Symbol``;
        if none is provided, then the function is commutative, so rfn_scalar is equal to lfn_scalar

    Returns
    --------
    mxnet.numpy.ndarray
        result array
    """
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
    elif isinstance(rhs, Symbol):
        return fn_array(lhs, rhs, out=out)
    else:
        raise TypeError('type %s not supported' % str(type(rhs)))
#pylint: enable= too-many-arguments, no-member, protected-access


@set_module('mxnet.symbol.numpy')
def maximum(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.maximum, _np.maximum, _npi.maximum_scalar, None, out)


@set_module('mxnet.symbol.numpy')
def minimum(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.minimum, _np.minimum, _npi.minimum_scalar, None, out)


@set_module('mxnet.symbol.numpy')
def add(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.add, _np.add, _npi.add_scalar, None, out)


@set_module('mxnet.symbol.numpy')
def subtract(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.subtract, _np.subtract, _npi.subtract_scalar,
                         _npi.rsubtract_scalar, out)


@set_module('mxnet.symbol.numpy')
def multiply(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.multiply, _np.multiply, _npi.multiply_scalar, None, out)


@set_module('mxnet.symbol.numpy')
def divide(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.true_divide, _np.divide, _npi.true_divide_scalar,
                         _npi.rtrue_divide_scalar, out)


@set_module('mxnet.symbol.numpy')
def mod(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.mod, _np.mod, _npi.mod_scalar, _npi.rmod_scalar, out)


@set_module('mxnet.symbol.numpy')
def power(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.power, _np.power, _npi.power_scalar, _npi.rpower_scalar, out)


@set_module('mxnet.symbol.numpy')
def mean(a, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
    """
    mean(a, axis=None, dtype=None, out=None, keepdims=None)

    Compute the arithmetic mean along the specified axis.
    Returns the average of the array elements.
    The average is taken over the flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : `_Symbol`
        _Symbol containing numbers whose mean is desired.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean. For integer inputs, the default is float32;
        for floating point inputs, it is the same as the input dtype.
    out : _Symbol, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
        If the default value is passed, then keepdims will not be passed through to the mean
        method of sub-classes of _Symbol, however any non-default value will be. If the sub-class
        method does not implement keepdims any exceptions will be raised.

    Returns
    -------
    m : _Symbol, see dtype parameter above
        If out=None, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    Notes
    -----
    This function differs from the original `numpy.mean
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html>`_ in
    the following way(s):

    - only _Symbol is accepted as valid input, python iterables or scalar is not supported
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


@set_module('mxnet.symbol.numpy')
def stack(arrays, axis=0, out=None):
    """
    Join a sequence of arrays along a new axis.

    The axis parameter specifies the index of the new axis in the dimensions of the result.
    For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last dimension.

    Parameters
    ----------
    arrays : sequence of _Symbols
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : _Symbol, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    out : _Symbol
        The stacked array has one more dimension than the input arrays.

    Notes
    -----
    This function differs from the original `numpy.stack
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.stack.html>`_ in
    the following ways:

    - only sequence of _Symbol is accepted as valid input

    Examples
    --------
    >>> arrays = [np.random.uniform(size=(3, 4)) for _ in range(10)]
    >>> np.stack(arrays, axis=0).shape
    (10, 3, 4)
    >>> np.stack(arrays, axis=1).shape
    (3, 10, 4)
    >>> np.stack(arrays, axis=2).shape
    (3, 4, 10)
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> np.stack((a, b))
    array([[1., 2., 3.],
           [2., 3., 4.]])
    >>> np.stack((a, b), axis=-1)
    array([[1., 2.],
           [2., 3.],
           [3., 4.]])
    """
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]

    arrays = get_list(arrays)
    return _npi.stack(*arrays, axis=axis, out=out)


@set_module('mxnet.symbol.numpy')
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


@set_module('mxnet.symbol.numpy')
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


@set_module('mxnet.symbol.numpy')
def argmax(a, axis=None, out=None):
    r"""
    argmax(a, axis=None, out=None)

    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : _Symbol
        Input array. Only support dtype `float16`, `float32`, and `float64`.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    index_array : _Symbol of indices whose dtype is same as the input ndarray.
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
    - Output has dtype that is same as the input ndarray.
    - ``out`` param: cannot perform auto broadcasting. ``out`` symbol's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` symnbol's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    """
    return _npi.argmax(a, axis=axis, keepdims=False, out=out)


@set_module('mxnet.symbol.numpy')
def argsort(a, axis=-1, kind='quicksort', order=None):
    """
    Returns the indices that would sort an input array along the given axis.
    This function performs sorting along the given axis and returns an array
    of indices having same shape as an input array that index data in sorted order.
    Parameters
    ----------
    a : _Symbol
        Input array
    axis : int, optional
        The axis along which to sort teh input tensor.
        If not given, the last, dimension -1 will be used by default.
        If None, the flattened array is used.
    kind: {'quicksort'}
        Currently not supported.
    order: None
        Currently not supported.
    Returns
    -------
    output : _Symbol
        Array of indices that sort a along the specified axis.
        If a is one-dimensional, a[index_array] yields a sorted a.
        More generally, np.take_along_axis(a, index_array, axis=a) always yields the sorted a,
        irrespective of dimensionality.
    Examples
    --------
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1., 2., 0.])
    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0., 3.],
           [2., 2.]])
    >>> np.argsort(x, axis=0)  # sorts along first axis (down)
    array([[0., 1.],
           [1., 0.]])
    >>> np.argsort(x, axis=1)  # sorts along last axis (across)
    array([[0., 1.],
           [0., 1.]])
    Notes
    -----
    This function differs from the original `numpy.argsort
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html>`_ in
    the following way(s):
    - kind and order is currently not supported
    """
    if kind != 'quicksort':
        raise AttributeError('mxnet.numpy.argsort does not support other sorting methods')
    if order is not None:
        raise AttributeError('mxnet.numpy.argsort does not support sorting with fields ordering')
    return _npi.argsort(a, axis)


@set_module('mxnet.symbol.numpy')
def clip(a, a_min, a_max, out=None):
    """clip(a, a_min, a_max, out=None)

    Clip (limit) the values in an array.
    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : _Symbol
        Array containing elements to clip.
    a_min : scalar or `None`
        Minimum value. If `None`, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.
    a_max : scalar or `None`
        Maximum value. If `None`, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        `None`.
    out : _Symbol or `None`
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.

    Returns
    -------
    clipped_array : _Symbol
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    Notes
    -----
    array_like `a_min` and `a_max` are not supported.
    """
    if a_min is None and a_max is None:
        raise ValueError('array_clip: must set either max or min')
    if a_min is None:
        a_min = float('-inf')
    if a_max is None:
        a_max = float('inf')
    return _npi.clip(a, a_min, a_max, out=out)


@set_module('mxnet.symbol.numpy')
def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array.

    Parameters
    ----------
    a : _Symbol
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : _Symbol
        Swapped array symbol.
    """
    return _npi.swapaxes(a, dim1=axis1, dim2=axis2)


@set_module('mxnet.symbol.numpy')
def expand_dims(a, axis):
    """Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded

    Parameters
    ----------
    a : _Symbol
        Input array.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : _Symbol
        Output array. The number of dimensions is one greater than that of
        the input array.
    """
    return _npi.expand_dims(a, axis)


# pylint: disable=line-too-long
@set_module('mxnet.symbol.numpy')
def split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : _Symbol
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

        Index `must be within` the dimension of the array along `axis`.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of _Symbols
        A list of sub-arrays.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division.

    Notes
    -----
    This function differs from the original `numpy.split
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html>`_ in
    the following ways:

    - Index exceeding the dimension the dimension of the array is currently not supported.

    Examples
    --------
    >>> x = np.arange(9.0)
    >>> np.split(x, 3)
    [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7., 8.])]
    >>> np.split(x, (3, 5, 6))
    [array([0., 1., 2.]), array([3., 4.]), array([5.]), array([6., 7.])]
    """
    indices = []
    sections = 0
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, tuple):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple of ints')
    ret = _npi.split(ary, indices, axis, False, sections)
    return ret
# pylint: enable=line-too-long


@set_module('mxnet.symbol.numpy')
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
    A : _Symbol or scalar
        An input array or a scalar to repeat.
    reps : a single integer or tuple of integers
        The number of repetitions of `x` along each axis.

    Returns
    -------
    c : _Symbol
        The tiled output array.
    """
    return _unary_func_helper(A, _npi.tile, _np.tile, reps=reps)


@set_module('mxnet.symbol.numpy')
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, ctx=None): # pylint: disable=too-many-arguments
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
    samples : _Symbol
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
    if isinstance(start, (list, _np.ndarray)) or \
        isinstance(stop, (list, _np.ndarray)):
        raise NotImplementedError('start and stop only support int')
    if axis != 0:
        raise NotImplementedError("the function only support axis 0")
    if ctx is None:
        ctx = current_context()
    if retstep:
        step = (stop - start) / (num - 1)
        return (_npi.linspace(start=start, stop=stop, num=num, endpoint=endpoint, ctx=ctx, dtype=dtype), step)
    else:
        return _npi.linspace(start=start, stop=stop, num=num, endpoint=endpoint, ctx=ctx, dtype=dtype)


@set_module('mxnet.symbol.numpy')
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


def _unary_func_helper(x, fn_array, fn_scalar, out=None, **kwargs):
    """Helper function for unary operators.

    Parameters
    ----------
    x : _Symbol or scalar
        Input of the unary operator.
    fn_array : function
        Function to be called if x is of ``_Symbol`` type.
    fn_scalar : function
        Function to be called if x is a Python scalar.
    out : _Symbol
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    out : _Symbol or scalar
        Result _Symbol or scalar.
    """
    if isinstance(x, numeric_types):
        return fn_scalar(x, **kwargs)
    elif isinstance(x, _Symbol):
        return fn_array(x, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.symbol.numpy')
def sin(x, out=None, **kwargs):
    r"""Trigonometric sine, element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol
        The sine of each element of x.
        This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.sin, _np.sin, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def cos(x, out=None, **kwargs):
    r"""Cosine, element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol
        The corresponding cosine values. This is a scalar if x is a scalar.

    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.cos, _np.cos, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def sinh(x, out=None, **kwargs):
    """Hyperbolic sine, element-wise.

    Equivalent to ``1/2 * (np.exp(x) - np.exp(-x))`` or ``-1j * np.sin(1j*x)``.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array or scalar.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol or scalar
        The corresponding hyperbolic sine values. This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.sinh, _np.sinh, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def cosh(x, out=None, **kwargs):
    """Hyperbolic cosine, element-wise.

    Equivalent to ``1/2 * (np.exp(x) + np.exp(-x))`` and ``np.cos(1j*x)``.


    Parameters
    ----------
    x : _Symbol or scalar
        Input array or scalar.
    out : ndarray or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol or scalar
        The corresponding hyperbolic cosine values. This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.cosh, _np.cosh, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def log10(x, out=None, **kwargs):
    """Return the base 10 logarithm of the input array, element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array or scalar.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol or scalar
        The logarithm to the base 10 of `x`, element-wise. NaNs are
        returned where x is negative. This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.log10, _np.log10, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def sqrt(x, out=None, **kwargs):
    """
    Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        The values whose square-roots are required.
    out : _Symbol, or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol or scalar
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`. This is a scalar if `x` is a scalar.

    Notes
    ----
    This function only supports input type of float.
    """
    return _unary_func_helper(x, _npi.sqrt, _np.sqrt, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def abs(x, out=None, **kwargs):
    r"""abs(x, out=None, **kwargs)

    Calculate the absolute value element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    absolute : _Symbol
        An ndarray containing the absolute value of
        each element in `x`. This is a scalar if `x` is a scalar.
    """
    return _unary_func_helper(x, _npi.abs, _np.abs, out=out, **kwargs)

@set_module('mxnet.symbol.numpy')
def sign(x, out=None, **kwargs):
    r"""
    sign(x, out=None)

    Returns an element-wise indication of the sign of a number.

    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``. Only supports real number.

    Parameters
    ----------
    x : _Symbol or a scalar
        Input values.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol
        The sign of `x`.
        This is a scalar if `x` is a scalar.

    Note
    -------
    - Only supports real number as input elements.
    - Input type does not support Python native iterables(list, tuple, ...)
    - ``out`` param: cannot perform auto broadcasting. ``out`` symbol's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` symbol's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    """
    return _unary_func_helper(x, _npi.sign, _np.sign, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def exp(x, out=None, **kwargs):
    r"""exp(x, out=None, **kwargs)

    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : _Symbol or scalar
        Input values.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    out : _Symbol
        Output array, element-wise exponential of `x`.
        This is a scalar if `x` is a scalar.
    """
    return _unary_func_helper(x, _npi.exp, _np.exp, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def arctan(x, out=None, **kwargs):
    r"""arctan(x, out=None, **kwargs)

    Trigonometric inverse tangent, element-wise.

    The inverse of tan, so that if ``y = tan(x)`` then ``x = arctan(y)``.

    Parameters
    ----------
    x : _Symbol or scalar
        Input values.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    out : _Symbol
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
    """
    return _unary_func_helper(x, _npi.arctan, _np.arctan, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def log(x, out=None, **kwargs):
    """
    log(x, out=None)

    Natural logarithm, element-wise.

    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base
    `e`.

    Parameters
    ----------
    x : _Symbol
        Input value. Elements must be of real value.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol
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
    - Input type does not support Python native iterables(list, tuple, ...). Only ndarray is supported.
    - ``out`` param: cannot perform auto braodcasting. ``out`` symbol's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` symbol's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    """
    return _unary_func_helper(x, _npi.log, _np.log, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def degrees(x, out=None, **kwargs):
    """
    degrees(x, out=None)

    Convert angles from radians to degrees.

    Parameters
    ----------
    x : _Symbol
        Input value. Elements must be of real value.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol of floats
        The corresponding degree values; if `out` was supplied this is a
        reference to it.
        This is a scalar if `x` is a scalar.

    Notes
    -------
    This function differs from the original `numpy.degrees
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.degrees.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...). Only ndarray is supported.
    - ``out`` param: cannot perform auto broadcasting. ``out`` symbol's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` symbol's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    """
    return _unary_func_helper(x, _npi.degrees, _np.degrees, out=out, **kwargs)


def rint(x, out=None, **kwargs):
    """
    Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    out : _Symbol or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.

    Notes
    -----
    This function differs from the original `numpy.rint
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.rint.html>`_ in
    the following way(s):

    - only _Symbol or scalar is accpted as valid input, tuple of _Symbol is not supported
     - broadcasting to `out` of different shape is currently not supported
    - when input is plain python numerics, the result will not be stored in the `out` param

    """
    return _unary_func_helper(x, _npi.rint, _np.rint, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def log2(x, out=None, **kwargs):
    """
    Base-2 logarithm of x.

    Parameters
    ----------
    x : _Symbol
        Input values.
    out : ndarray or None
        A location into which the result is stored.
        If provided, it must have the same shape and type as the input.
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    y : _Symbol
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

    """
    return _unary_func_helper(x, _npi.log2, _np.log2, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def radians(x, out=None, **kwargs):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array in degrees.
    out : _Symbol or None
       Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol
        The corresponding radian values. This is a scalar if x is a scalar.

    Notes
    -----
    This function differs from the original `numpy.radians
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.radians.html>`_ in
    the following way(s):

    - only _Symbol or scalar is accpted as valid input, tuple of _Symbol is not supported
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


@set_module('mxnet.symbol.numpy')
def reciprocal(x, out=None, **kwargs):
    r"""
    reciprocal(x, out=None)

    Return the reciprocal of the argument, element-wise.

    Calculates ``1/x``.

    Parameters
    ----------
    x : _Symbol or scalar
        The values whose reciprocals are required.
    out : _Symbol, or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.

    Notes
    -----
    .. note::
        This function is not designed to work with integers.

    For integer arguments with absolute value larger than 1 the result is
    always zero because of the way Python handles integer division.  For
    integer zero the result is an overflow.

    The output `symbol` has the same `ctx` as the input `symbol`.

    This function differs from the original `numpy.reciprocal
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.reciprocal.html>`_ in
    the following aspects:

    - Only support _Symbol and scalar now.
    - `where` argument is not supported.
    """
    return _unary_func_helper(x, _npi.reciprocal, _np.reciprocal, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def square(x, out=None, **kwargs):
    r"""
    square(x, out=None)

    Return the element-wise square of the input.

    Parameters
    ----------
    x : _Symbol or scalar
        The values whose reciprocals are required.
    out : _Symbol, or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.

    Notes
    -----
    The output `symbol` has the same `ctx` as the input `symbol`.

    This function differs from the original `numpy.square
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html>`_ in
    the following aspects:

    - Only support _Symbol and scalar now.
    - `where` argument is not supported.
    """
    return _unary_func_helper(x, _npi.square, _np.square, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def arcsin(x, out=None, **kwargs):
    r"""
    arcsin(x, out=None)

    Inverse sine, element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        The values whose reciprocals are required.
    out : _Symbol, or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    angle : _Symbol or scalar
        Output array is same shape and type as x. This is a scalar if x is a scalar.

    Notes
    -----
    `arcsin` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that :math:`sin(z) = x`.  The convention is to
    return the angle `z` whose real part lies in [-pi/2, pi/2].

    For real-valued input data types, *arcsin* always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    The inverse sine is also known as `asin` or sin^{-1}.

    The output `symbol` has the same `ctx` as the input `symbol`.

    This function differs from the original `numpy.arcsin
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsin.html>`_ in
    the following aspects:

    - Only support _Symbol or scalar now.
    - `where` argument is not supported.
    - Complex input is not supported.

    References
    ----------
    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79ff.
    http://www.math.sfu.ca/~cbm/aands/
    """
    return _unary_func_helper(x, _npi.arcsin, _np.arcsin, out=out, **kwargs)


_set_np_symbol_class(_Symbol)
