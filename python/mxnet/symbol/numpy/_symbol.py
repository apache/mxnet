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

# pylint: disable=too-many-lines, unused-argument
"""numpy namespace for operators used in Gluon APIs dispatched by F=symbol module."""

import ctypes
import numpy as _np
from . import _op as _mx_np_op
from ...base import _LIB, SymbolHandle, numeric_types, mx_uint, integer_types, string_types
from ...base import c_str
from ...base import py_str
from ...util import check_call, set_module, _sanity_check_params
from ...util import wrap_np_unary_func, wrap_np_binary_func
from ...util import is_np_default_dtype
from ...context import current_context
from ..symbol import Symbol, Group
from .._internal import _set_np_symbol_class
from . import _internal as _npi
try:
    from __builtin__ import slice as py_slice
except ImportError:
    from builtins import slice as py_slice

__all__ = ['zeros', 'zeros_like', 'ones', 'ones_like', 'full', 'full_like', 'empty_like', 'bitwise_not', 'invert',
           'delete', 'add', 'broadcast_to', 'subtract', 'multiply', 'divide', 'mod', 'remainder', 'fmod',
           'power', 'arctan2', 'trace', 'transpose', 'copy', 'moveaxis', 'reshape', 'dot',
           'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'log10', 'sqrt', 'cbrt', 'abs', 'absolute', 'fabs', 'exp',
           'expm1', 'arcsin', 'arccos', 'arctan', 'sign', 'log', 'degrees', 'log2', 'log1p', 'matmul', 'median',
           'rint', 'radians', 'reciprocal', 'square', 'negative', 'fix', 'ceil', 'floor', 'histogram', 'insert',
           'trunc', 'logical_not', 'arcsinh', 'arccosh', 'arctanh', 'argsort', 'sort', 'tensordot', 'eye', 'linspace',
           'logspace', 'expand_dims', 'tile', 'arange', 'array_split', 'split', 'hsplit', 'vsplit', 'dsplit',
           'concatenate', 'append', 'stack', 'vstack', 'row_stack', 'column_stack', 'hstack', 'dstack',
           'average', 'mean', 'maximum', 'fmax', 'minimum', 'fmin', 'any', 'all', 'around', 'round', 'round_',
           'flatnonzero', 'tril_indices', 'amax', 'amin', 'max', 'min', 'logical_and', 'logical_or', 'logical_xor',
           'swapaxes', 'clip', 'argmax', 'argmin', 'std', 'var', 'indices', 'copysign', 'ravel', 'unravel_index',
           'diag_indices_from', 'hanning', 'hamming', 'blackman', 'flip', 'flipud', 'fliplr',
           'hypot', 'bitwise_and', 'bitwise_xor', 'bitwise_or', 'rad2deg', 'deg2rad', 'unique', 'lcm', 'gcd', 'interp',
           'tril', 'triu', 'tri', 'identity', 'take', 'ldexp', 'vdot', 'inner', 'outer', 'cross', 'kron',
           'equal', 'not_equal', 'greater', 'less', 'greater_equal', 'less_equal', 'roll', 'rot90', 'einsum',
           'true_divide', 'quantile', 'percentile', 'shares_memory', 'may_share_memory', 'diff', 'ediff1d',
           'resize', 'polyval', 'nan_to_num', 'isnan', 'isinf', 'isposinf', 'isneginf', 'isfinite',
           'atleast_1d', 'atleast_2d', 'atleast_3d', 'squeeze',
           'where', 'bincount', 'rollaxis', 'diagflat', 'repeat', 'prod', 'pad', 'cumsum', 'sum', 'diag', 'diagonal']


@set_module('mxnet.symbol.numpy')
class _Symbol(Symbol):
    def __getitem__(self, key): # pylint: disable = too-many-return-statements, inconsistent-return-statements
        """Return self[key].

        If the symbol is a symbol list, it returns the i-th symbol or a list of symbols
        selected by key.

        Otherwise, it outputs a symbol that slice the input by the given key. Currently, this
        function supports the following types of key:

        - integer types, e.g., int, long, np.int32, np.int64
        - slice containing integer constants, e.g., slice(0, None, None)
        - tuple contaning the above elements, which is used for multidimensional indexing

        Parameters
        ----------
        key : int, slice, or tuple of all previous types
            Indexing key.

        """
        num_outputs = self.num_outputs
        if num_outputs > 1:
            num_outputs = self.num_outputs
            if isinstance(key, integer_types):
                key = int(key)
                if key < -num_outputs or key >= num_outputs:
                    raise IndexError('list index out of range')
                if key < 0:
                    key += num_outputs
                ret_handle = SymbolHandle()
                check_call(_LIB.MXSymbolGetOutput(self.handle, mx_uint(key),
                                                  ctypes.byref(ret_handle)))
                return _Symbol(handle=ret_handle)
            elif isinstance(key, py_slice):
                start, stop, step = key.indices(num_outputs)
                return Group([self[i] for i in range(start, stop, step)], _Symbol)
            else:
                raise TypeError('indices of symbol group must be integers or slices, not {}'
                                .format(type(key)))
        else:
            all = __builtins__['all']  # pylint: disable=redefined-outer-name
            if isinstance(key, integer_types):
                if key == -1:
                    sliced = _npi.slice(self, [key], [None])
                else:
                    sliced = _npi.slice(self, [key], [key+1])
                return _npi.reshape(sliced, (-3, -4))
            elif isinstance(key, py_slice):
                if key.step is None or key.step != 0:
                    start = [None] if key.start is None else key.start
                    stop = [None] if key.stop is None else key.stop
                    return _npi.slice(self, start, stop, key.step)
                else:
                    raise ValueError("slice step cannot be zero")
            elif isinstance(key, Symbol):
                return _npi.advanced_indexing(self, key)
            elif isinstance(key, tuple) and len(key) == 0:
                return self
            elif isinstance(key, tuple) and all(isinstance(k, Symbol) for k in key):
                key = _npi.stack(*[i for i in key])
                sliced = _npi.advanced_indexing_multiple(self, key)
                return sliced
            elif isinstance(key, tuple):
                begin = []
                end = []
                step = []
                new_shape = ()
                assert len(key)  # len(key) == 0 handled above
                for index in key:
                    if isinstance(index, py_slice):
                        if index.step is not None and index.step == 0:
                            raise ValueError("slice step cannot be zero")
                        begin.append(index.start)
                        end.append(index.stop)
                        step.append(index.step)
                        new_shape += (-2,)
                    elif isinstance(index, integer_types):
                        if index >= 0:
                            begin.append(index)
                            end.append(index+1)
                            step.append(1)
                        else:
                            begin.append(index)
                            end.append(index - 1)
                            step.append(-1)
                        new_shape += (-3,)
                    else:
                        raise IndexError('Only integer, slice, symbol or tuple of these types'
                                         ' are supported! Received key={}'.format(key))
                new_shape += (-4,)
                sliced = _npi.slice(self, begin, end, step)
                return _npi.reshape(sliced, new_shape)
            else:
                raise IndexError('Only integer, slice, tuple or Symbol of these types are supported! '
                                 'Received key={}'.format(key))

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __repr__(self):
        """Gets a string representation of the symbol."""
        if self._alive:
            if self.num_outputs > 1:
                name = ', '.join([str(ele_sym) for ele_sym in self])
                return f'<{self.__class__.__name__} group [{name}]>'
            else:
                return f'<{self.__class__.__name__} {self.name}>'
        else:
            return '<FREED {}>'.format(self.__class__.__name__)

    @property
    def name(self):
        """Gets name string from the symbol, this function only works for symbols
         that are not a list (grouped symbols).

        Returns
        -------
        value : str
            The name of this symbol, returns ``None`` for list symbol.
        """
        if self.num_outputs > 1:
            raise AttributeError('This is a Group Symbol that contains {} elements and'
                                 ' does not have a name. Use str(sym) to print the name of '
                                 'all the elements instead.'.format(self.num_outputs))
        ret = ctypes.c_char_p()
        success = ctypes.c_int()
        check_call(_LIB.MXSymbolGetName(
            self.handle, ctypes.byref(ret), ctypes.byref(success)))
        assert success.value != 0,\
            'Fail to infer the name of a symbol that is not a list!'
        return py_str(ret.value)

    def __iter__(self):
        if self.num_outputs == 1:
            raise TypeError("'{}' is not iterable.".format(self))
        return iter((self[i] for i in range(self.num_outputs)))

    def __add__(self, other):
        """x.__add__(y) <=> x + y"""
        return add(self, other)

    def __invert__(self):
        """x.__invert__() <=> ~x"""
        return invert(self)

    def __and__(self, other):
        """x.__and__(y) <=> x & y"""
        return bitwise_and(self, other)

    def __or__(self, other):
        """x.__or__(y) <=> x | y"""
        return bitwise_or(self, other)

    def __xor__(self, other):
        """x.__xor__(y) <=> x ^ y"""
        return bitwise_xor(self, other)

    def __round__(self, n=0):
        """x.__round__(n)"""
        return round(self, decimals=n)

    def __abs__(self):
        """x.__abs__()"""
        return absolute(self)

    def __ceil__(self):
        """x.__ceil__()"""
        return ceil(self)

    def __floor__(self):
        """x.__floor__()"""
        return floor(self)

    def __trunc__(self):
        """x.__trunc__()"""
        return trunc(self)

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
        """x.__truediv__(y) <=> x / y"""
        return divide(self, other)

    def __rdiv__(self, other):
        """x.__rdiv__(y) <=> y / x"""
        return divide(other, self)

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
        return negative(self)

    def __deepcopy__(self, _):
        return super().__deepcopy__(_).as_np_ndarray()

    def __eq__(self, other):
        """x.__eq__(y) <=> x == y"""
        return equal(self, other)

    def __ne__(self, other):
        """x.__ne__(y) <=> x != y"""
        return not_equal(self, other)

    def __gt__(self, other):
        """x.__gt__(y) <=> x > y"""
        return greater(self, other)

    def __ge__(self, other):
        """x.__ge__(y) <=> x >= y"""
        return greater_equal(self, other)

    def __lt__(self, other):
        """x.__lt__(y) <=> x < y"""
        return less(self, other)

    def __le__(self, other):
        """x.__le__(y) <=> x <= y"""
        return less_equal(self, other)

    def __len__(self):
        if self.num_outputs == 1:
            raise TypeError('{} is not a list and does not support len().'.format(self))
        return self.num_outputs

    @property
    def num_outputs(self):
        """The number of outputs of a symbol. If the symbol is not a symbollist, it returns 1.
        Otherwise, it returns the number of elements of the list."""
        output_count = mx_uint()
        check_call(_LIB.MXSymbolGetNumOutputs(self.handle, ctypes.byref(output_count)))
        return output_count.value

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

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):  # pylint: disable=arguments-differ,unused-argument,too-many-arguments,redefined-outer-name
        """
        Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout order of the result.
            'C' means C order, 'F' means Fortran order, 'A'
            means 'F' order if all the arrays are Fortran contiguous,
            'C' order otherwise, and 'K' means as close to the
            order the array elements appear in memory as possible.
            Default is 'K'.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur. Defaults to 'unsafe'
            for backwards compatibility.

              * 'no' means the data types should not be cast at all.
              * 'equiv' means only byte-order changes are allowed.
              * 'safe' means only casts which can preserve values are allowed.
              * 'same_kind' means only safe casts or casts within a kind,
                like float64 to float32, are allowed.
              * 'unsafe' means any data conversions may be done.
        subok : bool, optional
            If True, then sub-classes will be passed-through (default), otherwise
            the returned array will be forced to be a base-class array.
        copy : bool, optional
            Default `True`. By default, astype always returns a newly
            allocated ndarray on the same context. If this is set to
            `False`, and the dtype requested is the same as the ndarray's
            dtype, the ndarray is returned instead of a copy.

        Returns
        -------
        arr_t : ndarray
            Unless `copy` is False and the other conditions for returning the input
            array are satisfied (see description for `copy` input parameter), `arr_t`
            is a new array of the same shape as the input array with `dtype`.

        Notes
        -----
        This function differs from the official `ndarray`'s ``astype`` function in the following
        aspects:
            - `order` only supports 'C' and 'K'.
            - `casting` only supports 'unsafe'.
            - `subok` only supports ``True``.
        """
        if order is not None and order != 'K' and order != 'C':
            raise ValueError('order must be either \'K\' or \'C\'')
        if casting != 'unsafe':
            raise ValueError('casting must be equal to \'unsafe\'')
        if not subok:
            raise ValueError('subok must be equal to True')
        return _npi.cast(self, dtype=dtype)

    def dot(self, b, out=None):
        """Dot product of two arrays.
        Refer to ``numpy.dot`` for full documentation."""
        return _npi.dot(self, b, out=out)

    def reshape(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Returns a copy of the array with a new shape.

        Notes
        -----
        Unlike the free function `mxnet.numpy.reshape`, this method on `ndarray` allows
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
        """Return indices of the maximum values along the given axis.
        Refer to `mxnet.numpy.argmax` for full documentation."""
        return argmax(self, axis, out)

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
        return repeat(self, repeats=repeats, axis=axis)

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

    def take(self, indices, axis=None, mode='raise'):  # pylint: disable=arguments-differ, redefined-outer-name
        """Convenience fluent method for :py:func:`take`.

        The arguments are the same as for :py:func:`take`, with
        this array as data.
        """
        return take(self, indices, axis, mode=mode)

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

    def sort(self, axis=-1, kind=None, order=None):  # pylint: disable=arguments-differ
        """Convenience fluent method for :py:func:`sort`.

        The arguments are the same as for :py:func:`sort`, with
        this array as data.
        """
        raise sort(self, axis=axis, kind=kind, order=order)

    def topk(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`topk`.

        The arguments are the same as for :py:func:`topk`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute topk')

    def argsort(self, axis=-1, kind=None, order=None):  # pylint: disable=arguments-differ
        """Convenience fluent method for :py:func:`argsort`.

        The arguments are the same as for :py:func:`argsort`, with
        this array as data.
        """
        return argsort(self, axis=axis, kind=kind, order=order)

    def argmax_channel(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax_channel`.

        The arguments are the same as for :py:func:`argmax_channel`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute argmax_channel')

    def argmin(self, axis=None, out=None):  # pylint: disable=arguments-differ
        """Return indices of the minimum values along the given axis.
        Refer to `mxnet.numpy.argmax` for full documentation."""
        return argmin(self, axis, out)

    def clip(self, min=None, max=None, out=None):  # pylint: disable=arguments-differ, redefined-outer-name
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

    def expand_dims(self, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
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
        """The arguments are the same as for :py:func:`transpose`, with
        this array as data.
        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)):
                axes = axes[0]
            elif axes[0] is None:
                axes = None
        return transpose(self, axes=axes)

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

    def diagonal(self, offset=0, axis1=0, axis2=1):  # pylint: disable=arguments-differ
        """Return the diagonal with the given offset.

        If array has more than two dimensions, then the axes specified by axis1 and
        axis2 are used to determine the 2-D sub-array whose diagonal is returned.

        Refer to `mxnet.symbol.numpy.diagonal` for full documents.
        """
        return diagonal(self, offset=offset, axis1=axis1, axis2=axis2)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the sum of the array elements over the given axis."""
        return _npi.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

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
        """Returns the average of the array elements along given axis."""
        return mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):  # pylint: disable=arguments-differ,too-many-arguments
        """Returns the standard deviation of the array elements along given axis."""
        return std(self, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, out=out)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False):  # pylint: disable=arguments-differ,too-many-arguments
        """Returns the variance of the array elements, along given axis."""
        return var(self, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)

    def cumsum(self, axis=None, dtype=None, out=None):
        """Return the cumulative sum of the elements along the given axis."""
        return _npi.cumsum(self, axis=axis, dtype=dtype, out=out)

    def max(self, axis=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the maximum along a given axis."""
        return _npi.max(self, axis=axis, keepdims=keepdims, out=out)

    def min(self, axis=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Return the minimum along a given axis."""
        return _npi.min(self, axis=axis, keepdims=keepdims, out=out)

    def norm(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`norm`.

        The arguments are the same as for :py:func:`norm`, with
        this array as data.
        """
        raise AttributeError('_Symbol object has no attribute norm')

    def round(self, decimals=0, out=None, **kwargs): # pylint: disable=arguments-differ
        """Convenience fluent method for :py:func:`round`.

        The arguments are the same as for :py:func:`round`, with
        this array as data.
        """
        return round(self, decimals=decimals, out=out, **kwargs)

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
        """Remove single-dimensional entries from the shape of a."""
        return squeeze(self, axis=axis)

    def broadcast_to(self, *args, **kwargs):
        raise AttributeError('_Symbol object has no attribute broadcast_to')

    def broadcast_like(self, *args, **kwargs):
        raise AttributeError('_Symbol object has no attribute broadcast_like')

    # pylint: disable=too-many-arguments
    def optimize_for(self, backend, args=None, aux=None, ctx=None,
                     shape_dict=None, type_dict=None, stype_dict=None, skip_infer=False, **kwargs):
        """Partitions current symbol and optimizes it for a given backend."""
        new_sym = super().optimize_for(backend, args, aux, ctx, shape_dict, type_dict,
                                       stype_dict, skip_infer, **kwargs)
        new_sym = new_sym.as_np_ndarray()
        return new_sym

@set_module('mxnet.symbol.numpy')
def zeros(shape, dtype=float, order='C', ctx=None):
    """Return a new array of given shape and type, filled with zeros.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type .
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
        Note that this behavior is different from NumPy's `zeros` function where `float64`
        is the default value, here we can set 'float32' or 'float64' as your default dtype,
        because `float32` is considered as the default data type in deep learning.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : Symbol
        Array of zeros with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    if dtype is None or dtype is float:
        dtype = _np.float64 if is_np_default_dtype() else _np.float32
    return _npi.zeros(shape=shape, ctx=ctx, dtype=dtype)


@set_module('mxnet.symbol.numpy')
def ones(shape, dtype=None, order='C', ctx=None):
    """Return a new array of given shape and type, filled with ones.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type.
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
        Note that this behavior is different from NumPy's `ones` function where
        `float64` is the default value.
    order : {'C'}, optional, default: 'C'
        How to store multi-dimensional data in memory, currently only row-major
        (C-style) is supported.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : _Symbol
        Array of ones with the given shape, dtype, and ctx.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    return _npi.ones(shape=shape, ctx=ctx, dtype=dtype)


@set_module('mxnet.symbol.numpy')
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


@set_module('mxnet.symbol.numpy')
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


@set_module('mxnet.symbol.numpy')
def broadcast_to(array, shape):
    """
    Broadcast an array to a new shape.

    Parameters
    ----------
    array : _Symbol or scalar
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


@set_module('mxnet.symbol.numpy')
def full(shape, fill_value, dtype=None, order='C', ctx=None, out=None):  # pylint: disable=too-many-arguments
    """
    Return a new array of given shape and type, filled with `fill_value`.
    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar or _Symbol
        Fill value.
    dtype : data-type, optional
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
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
    if isinstance(fill_value, Symbol):
        if dtype is None:
            ret = broadcast_to(fill_value, shape)
        else:
            ret = broadcast_to(fill_value, shape).astype(dtype)
        return ret
    if isinstance(fill_value, bool):
        fill_value = int(fill_value)
        dtype = _np.bool if dtype is None else dtype
    return _npi.full(shape=shape, value=fill_value, ctx=ctx, dtype=dtype, out=out)


@set_module('mxnet.symbol.numpy')
def full_like(a, fill_value, dtype=None, order='C', ctx=None, out=None):  # pylint: disable=too-many-arguments
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : _Symbol
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
    out : _Symbol
          Array `fill_value` with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full : Return a new array of given shape filled with value.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    if isinstance(fill_value, bool):
        fill_value = int(fill_value)
    return _npi.full_like(a, fill_value=fill_value, ctx=ctx, dtype=dtype, out=out)


@set_module('mxnet.symbol.numpy')
def zeros_like(a, dtype=None, order='C', ctx=None, out=None):  # pylint: disable=too-many-arguments
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : _Symbol
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
    out : _Symbol
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    zeros : Return a new array of given shape filled with zeros.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    return _npi.full_like(a, fill_value=0, ctx=ctx, dtype=dtype, out=out)


@set_module('mxnet.symbol.numpy')
def ones_like(a, dtype=None, order='C', ctx=None, out=None):  # pylint: disable=too-many-arguments
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : _Symbol
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
    out : _Symbol
          Array of ones with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    zeros : Return a new array of given shape filled with zeros.
    """
    if order != 'C':
        raise NotImplementedError
    if ctx is None:
        ctx = current_context()
    return _npi.full_like(a, fill_value=1, ctx=ctx, dtype=dtype, out=out)


@set_module('mxnet.symbol.numpy')
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
        Data-type of the output.
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : _Symbol
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.
    """
    if not isinstance(n, int):
        raise TypeError("Input 'n' should be an integer")
    if n < 0:
        raise ValueError("Input 'n' cannot be negative")
    if ctx is None:
        ctx = current_context()
    return _npi.identity(shape=(n, n), ctx=ctx, dtype=dtype)


# pylint: disable=redefined-outer-name
@set_module('mxnet.symbol.numpy')
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
    a : _Symbol
        The source array.
    indices : _Symbol
        The indices of the values to extract. Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.
    mode : {'clip', 'wrap'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'clip' -- clip to the range (default)
        * 'wrap' -- wrap around

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    Returns
    -------
    out : _Symbol
        The returned array has the same type as `a`.

    Notes
    -----

    This function differs from the original `numpy.take
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html>`_ in
    the following way(s):

    - Only ndarray or scalar ndarray is accepted as valid input.
    """
    if mode not in ('wrap', 'clip', 'raise'):
        raise NotImplementedError(
            "function take does not support mode '{}'".format(mode))
    if axis is None:
        return _npi.take(_npi.reshape(a, -1), indices, 0, mode, out)
    else:
        return _npi.take(a, indices, axis, mode, out)
# pylint: enable=redefined-outer-name


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
            is_int = isinstance(rhs, integer_types)
            if rfn_scalar is None:
                # commutative function
                return lfn_scalar(rhs, scalar=float(lhs), is_int=is_int, out=out)
            else:
                return rfn_scalar(rhs, scalar=float(lhs), is_int=is_int, out=out)
    elif isinstance(rhs, numeric_types):
        is_int = isinstance(rhs, integer_types)
        return lfn_scalar(lhs, scalar=float(rhs), is_int=is_int, out=out)
    elif isinstance(rhs, Symbol):
        return fn_array(lhs, rhs, out=out)
    else:
        raise TypeError(f'type {str(type(rhs))} not supported')
#pylint: enable= too-many-arguments, no-member, protected-access


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def add(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.add, _np.add, _npi.add_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def subtract(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.subtract, _np.subtract, _npi.subtract_scalar,
                         _npi.rsubtract_scalar, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def multiply(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.multiply, _np.multiply, _npi.multiply_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def divide(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.true_divide, _np.divide, _npi.true_divide_scalar,
                         _npi.rtrue_divide_scalar, out)


@set_module('mxnet.symbol.numpy')
def true_divide(x1, x2, out=None):
    return _ufunc_helper(x1, x2, _npi.true_divide, _np.divide, _npi.true_divide_scalar,
                         _npi.rtrue_divide_scalar, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def mod(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.mod, _np.mod, _npi.mod_scalar, _npi.rmod_scalar, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def fmod(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.fmod, _np.fmod, _npi.fmod_scalar, _npi.rfmod_scalar, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def remainder(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.mod, _np.mod, _npi.mod_scalar, _npi.rmod_scalar, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def power(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.power, _np.power, _npi.power_scalar, _npi.rpower_scalar, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def gcd(x1, x2, out=None, **kwargs):
    """
    Returns the greatest common divisor of ``|x1|`` and ``|x2|``

    Parameters
    ----------
    x1, x2 : ndarrays or scalar values
        The arrays for computing greatest common divisor. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which may be the shape of
        one or the other).

    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    y : ndarray or scalar
        The greatest common divisor of the absolute value of the inputs
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    lcm : The lowest common multiple
    """
    return _ufunc_helper(x1, x2, _npi.gcd, _np.gcd, _npi.gcd_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def matmul(a, b, out=None, **kwargs):
    """
    Matrix product of two arrays.

    Parameters
    ----------
    a, b : _Symbol.
    out : _Symbol, optional
        A location into which the result is stored.
        If provided, it must have a shape that matches the signature (n,k),(k,m)->(n,m).
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    y : _Symbol
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
      respecting the signature (n,k),(k,m)->(n,m).
    """
    return _npi.matmul(a, b, out=out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def lcm(x1, x2, out=None, **kwargs):
    """
    Returns the lowest common multiple of ``|x1|`` and ``|x2|``

    Parameters
    ----------
    x1, x2 : _Symbols or scalar values
        The arrays for computing lowest common multiple. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which may be the shape of
        one or the other).

    out : _Symbol or None, optional
        A location into which the result is stored. If provided, it must have a shape
        that the inputs broadcast to. If not provided or None, a freshly-allocated array
        is returned.

    Returns
    -------
    y : _Symbol or scalar
        The lowest common multiple of the absolute value of the inputs
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    gcd : The greatest common divisor
    """
    return _ufunc_helper(x1, x2, _npi.lcm, _np.lcm, _npi.lcm_scalar, None, out)


@set_module('mxnet.symbol.numpy')
def argsort(a, axis=-1, kind=None, order=None):
    """
    Returns the indices that would sort an array.
    Perform an indirect sort along the given axis using the algorithm specified
    by the `kind` keyword. It returns an array of indices of the same shape as
    `a` that index data along the given axis in sorted order.

    Parameters
    ----------
    a : _Symbol
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
    index_array : _Symbol, int
        Array of indices that sort `a` along the specified `axis`.
        If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.
        More generally, ``np.take_along_axis(a, index_array, axis=axis)``
        always yields the sorted `a`, irrespective of dimensionality.

    Notes
    -----
    This operator does not support different sorting algorithms.
    """
    if order is not None:
        raise NotImplementedError("order is not supported yet...")

    return _npi.argsort(data=a, axis=axis, is_ascend=True, dtype='int64')


@set_module('mxnet.symbol.numpy')
def sort(a, axis=-1, kind=None, order=None):
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    a : _Symbol
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
    """
    if order is not None:
        raise NotImplementedError("order is not supported yet...")

    return _npi.sort(data=a, axis=axis, is_ascend=True)

@set_module('mxnet.symbol.numpy')
def dot(a, b, out=None):
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
    a : _Symbol
        First argument.
    b : _Symbol
        Second argument.

    out : _Symbol, optional
        Output argument. It must have the same shape and type as the expected output.

    Returns
    -------
    output : _Symbol
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
    return _npi.dot(a, b, out=out)

@set_module('mxnet.symbol.numpy')
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
    a, b : _Symbol
        Tensors to "dot".
    axes : int or (2,) ndarray
        * integer_like
        If an int N, sum over the last N axes of `a` and the first N axes
        of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
        Or, a list of axes to be summed over, first sequence applying to `a`,
        second to `b`. Both elements array_like must be of the same length.
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


@set_module('mxnet.symbol.numpy')
def histogram(a, bins=10, range=None, normed=None, weights=None, density=None):  # pylint: disable= too-many-arguments
    """
    Compute the histogram of a set of data.

    Parameters
    ----------
    a : Symbol
        Input data. The histogram is computed over the flattened array.
    bins : int or Symbol
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
            raise NotImplementedError("automatic range is not avaialble yet...")
        return _npi.histogram(a, bin_cnt=bins, range=range)
    if isinstance(bins, (list, tuple)):
        raise NotImplementedError("array_like bins is not supported yet...")
    if isinstance(bins, str):
        raise NotImplementedError("string bins is not supported yet...")
    if isinstance(bins, Symbol):
        return _npi.histogram(a, bins)
    raise ValueError("histogram fails with", locals())


@set_module('mxnet.symbol.numpy')
def eye(N, M=None, k=0, dtype=float, **kwargs):
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
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.

    Returns
    -------
    I : _Symbol of shape (N,M)
        An array where all elements are equal to zero,
        except for the k-th diagonal, whose values are equal to one.
    """
    _sanity_check_params('eye', ['order'], kwargs)
    ctx = kwargs.pop('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    if dtype is None or dtype is float:
        dtype = _np.float64 if is_np_default_dtype() else _np.float32
    return _npi.eye(N, M, k, ctx, dtype)


@set_module('mxnet.symbol.numpy')
def empty_like(prototype, dtype=None, order='C', subok=False, shape=None): # pylint: disable=W0621
    """
    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : _Symbol
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. Currently only supports C order.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to False.
        (Only support False at this moment)
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
        (This parameter is not supported at this moment)

    Returns
    -------
    out : _Symbol
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
    """
    dtype_list = {None:'None', _np.int8:'int8', _np.uint8:'uint8', _np.int32:'int32',
                  _np.int64:'int64', _np.float16:'float16', _np.float32:'float32',
                  _np.float64:'float64', _np.bool_:'bool_', bool:'bool', int:'int64', float:'float64'}
    if order != 'C':
        raise NotImplementedError("Only support C order at this moment")
    if subok:
        raise NotImplementedError("Creating array by using sub-class is not supported at this moment")
    if shape is not None:
        raise NotImplementedError("Parameter 'shape' is not supported at this moment")
    try:
        dtype = dtype if isinstance(dtype, str) else dtype_list[dtype]
    except:
        raise NotImplementedError("Do not support this dtype at this moment")
    return _npi.empty_like_fallback(prototype, dtype=dtype, order=order, subok=subok, shape=shape)


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
    if isinstance(start, (list, _np.ndarray)) or isinstance(stop, (list, _np.ndarray)):
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


@set_module('mxnet.symbol.numpy')
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0, ctx=None): # pylint: disable=too-many-arguments
    r"""Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

        Non-scalar `start` and `stop` are now supported.

    Parameters
    ----------
    start : scalar
        ``base ** start`` is the starting value of the sequence.
    stop : scalar
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : scalar, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : scalar, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : scalar, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Now, axis only support axis = 0.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    samples : _Symbol
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
    Logspace is equivalent to the code

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
    if isinstance(start, (list, _np.ndarray)) or \
       isinstance(stop, (list, _np.ndarray)):
        raise NotImplementedError('start and stop only support int')
    if axis != 0:
        raise NotImplementedError("the function only support axis 0")
    if ctx is None:
        ctx = current_context()
    return _npi.logspace(start=start, stop=stop, num=num, endpoint=endpoint, base=base, ctx=ctx, dtype=dtype)


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


@set_module('mxnet.symbol.numpy')
def tril(m, k=0):
    r"""
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : _Symbol, shape (M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril : _Symbol, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : same thing, only for the upper triangle
    """
    return _npi.tril(m, k)


@set_module('mxnet.symbol.numpy')
def triu(m, k=0):
    r"""
    Upper triangle of an array.

    Return a copy of an array with elements under the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : _Symbol, shape (M, N)
        Input array.
    k : int, optional
        Diagonal under which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is under.

    Returns
    -------
    triu : _Symbol, shape (M, N)
        Upper triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    tril : same thing, only for the lower triangle
    """
    return _npi.triu(m, k)


def tril_indices(n, k=0, m=None):
    """
    Return the indices for the lower-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The row dimension of the arrays for which the returned
        indices will be valid.
    k : int, optional
        Diagonal offset (see `tril` for details).
    m : int, optional
        .. versionadded:: 1.9.0

        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.


    Returns
    -------
    inds : tuple of _Symbol
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.

    See also
    --------
    triu_indices : similar function, for upper-triangular.
    mask_indices : generic function accepting an arbitrary mask function.
    tril, triu

    Notes
    -----
    .. versionadded:: 1.4.0

    Examples
    --------
    Compute two different sets of indices to access 4x4 arrays, one for the
    lower triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> il1 = np.tril_indices(4)
    >>> il2 = np.tril_indices(4, 2)

    Here is how they can be used with a sample array:

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Both for indexing:

    >>> a[il1]
    array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])

    And for assigning values:

    >>> a[il1] = -1
    >>> a
    array([[-1,  1,  2,  3],
           [-1, -1,  6,  7],
           [-1, -1, -1, 11],
           [-1, -1, -1, -1]])

    These cover almost the whole array (two diagonals right of the main one):

    >>> a[il2] = -10
    >>> a
    array([[-10, -10, -10,   3],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10]])

    """
    if m is None:
        m = n
    return _npi.tril_indices(n, k, m)


@set_module('mxnet.symbol.numpy')
def trace(a, offset=0, axis1=0, axis2=1, out=None):
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
    a : _Symbol
        Input array, from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both positive
        and negative. Defaults to 0.
    axis1, axis2 : int, optional
        Axes to be used as the first and second axis of the 2-D sub-arrays
        from which the diagonals should be taken. Defaults are the first two
        axes of `a`.
    out : _Symbol
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    sum_along_diagonals : _Symbol
        If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
        larger dimensions, then an array of sums along diagonals is returned.
    """
    return _npi.trace(a, offset=offset, axis1=axis1, axis2=axis2, out=out)


@set_module('mxnet.symbol.numpy')
def transpose(a, axes=None):
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    a : _Symbol
        Input array.
    axes : list of ints, optional
        By default, reverse the dimensions,
        otherwise permute the axes according to the values given.

    Returns
    -------
    p : _Symbol
        a with its axes permuted.
    """
    return _npi.transpose(a, axes=axes)


@set_module('mxnet.symbol.numpy')
def tri(N, M=None, k=0, dtype=None, ctx=None):
    r"""
    An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is float.

    Returns
    -------
    tri : Symbol of shape (N, M)
        Array with its lower triangle filled with ones and zero elsewhere;
        in other words ``T[i,j] == 1`` for ``i <= j + k``, 0 otherwise.
    """
    if dtype is None:
        dtype = 'float32'
    if M is None:
        M = N
    if ctx is None:
        ctx = current_context()
    return _npi.tri(N, M, k, dtype, ctx)


def repeat(a, repeats, axis=None):
    """
    Repeat elements of an array.

    Parameters
    ----------
    a : array_like
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

    See Also
    --------
    tile : Tile an array.

    Examples
    --------
    >>> np.repeat(3, 4)
    array([3, 3, 3, 3])
    >>> x = np.array([[1,2],[3,4]])
    >>> np.repeat(x, 2)
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> np.repeat(x, 3, axis=1)
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> np.repeat(x, [1, 2], axis=0)
    array([[1, 2],
           [3, 4],
           [3, 4]])
    """
    if isinstance(repeats, numeric_types):
        repeats = [repeats]
    if axis is not None:
        tmp = swapaxes(a, 0, axis)
        res = _npi.repeats(tmp, repeats=repeats, axis=0)
        return swapaxes(res, 0, axis)
    return _npi.repeats(a, repeats=repeats, axis=axis)


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
@wrap_np_unary_func
def sin(x, out=None, **kwargs):
    r"""
    Trigonometric sine, element-wise.

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
@wrap_np_unary_func
def cos(x, out=None, **kwargs):
    r"""
    Cosine, element-wise.

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
@wrap_np_unary_func
def sinh(x, out=None, **kwargs):
    """
    Hyperbolic sine, element-wise.
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
@wrap_np_unary_func
def cosh(x, out=None, **kwargs):
    """
    Hyperbolic cosine, element-wise.
    Equivalent to ``1/2 * (np.exp(x) + np.exp(-x))`` and ``np.cos(1j*x)``.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array or scalar.
    out : _Symbol or None
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
@wrap_np_unary_func
def tanh(x, out=None, **kwargs):
    """
    Compute hyperbolic tangent element-wise.
    Equivalent to ``np.sinh(x)/np.cosh(x)``.

    Parameters
    ----------
    x : _Symbol
        Input array.
    out : _Symbol or None
          Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol
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
    >>> # Example of ValueError due to provision of shape mis-matched `out`
    >>> np.tanh(np.zeros((3,3)),np.zeros((2,2)))
    mxnet.base.MXNetError:
    [07:17:36] ../src/ndarray/./../operator/tensor/../elemwise_op_common.h:135:
    Check failed: assign(&dattr, vec.at(i)): Incompatible attr in node
    at 0-th output: expected [3,3], got [2,2]
    """
    return _unary_func_helper(x, _npi.tanh, _np.tanh, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def log10(x, out=None, **kwargs):
    """
    Return the base 10 logarithm of the input array, element-wise.

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
@wrap_np_unary_func
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
@wrap_np_unary_func
def cbrt(x, out=None, **kwargs):
    r"""
    Return the cube-root of an array, element-wise.

    Parameters
    ----------
    x : _Symbol
        The values whose cube-roots are required.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    ----------
    y : _Symbol
        An array of the same shape as x, containing the cube cube-root of each element in x.
        If out was provided, y is a reference to it. This is a scalar if x is a scalar.
    """
    return _unary_func_helper(x, _npi.cbrt, _np.cbrt, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def abs(x, out=None, **kwargs):
    r"""
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
@wrap_np_unary_func
def fabs(x, out=None, **kwargs):
    r"""
    Calculate the absolute value element-wise.

    This function returns the absolute values (positive magnitude) of the
    data in `x`. Complex values are not handled, use `absolute` to find the
    absolute values of complex data.

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
@wrap_np_unary_func
def absolute(x, out=None, **kwargs):
    r"""
    Calculate the absolute value element-wise.
    np.abs is a shorthand for this function.

    Parameters
    ----------
    x : _Symbol
        Input array.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    ----------
    absolute : _Symbol
        An ndarray containing the absolute value of each element in x.
    """
    return _unary_func_helper(x, _npi.absolute, _np.absolute, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def sign(x, out=None, **kwargs):
    r"""
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
@wrap_np_unary_func
def exp(x, out=None, **kwargs):
    r"""
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
@wrap_np_unary_func
def expm1(x, out=None, **kwargs):
    r"""
    Calculate `exp(x) - 1` for all elements in the array.

    Parameters
    ----------
    x : _Symbol or scalar
        Input values.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    out : _Symbol
        Output array, .
        This is a scalar if `x` is a scalar.
    """
    return _unary_func_helper(x, _npi.expm1, _np.expm1, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def arcsin(x, out=None, **kwargs):
    r"""
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
    <https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html>`_ in
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


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def arccos(x, out=None, **kwargs):
    r"""
    Trigonometric inverse cosine, element-wise.
    The inverse of cos so that, if y = cos(x), then x = arccos(y).

    Parameters
    ----------
    x : _Symbol
        x-coordinate on the unit circle. For real arguments, the domain is [-1, 1].
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    ----------
    angle : _Symbol
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
    """
    return _unary_func_helper(x, _npi.arccos, _np.arccos, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def arctan(x, out=None, **kwargs):
    r"""
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
@wrap_np_unary_func
def log(x, out=None, **kwargs):
    """
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
@wrap_np_unary_func
def degrees(x, out=None, **kwargs):
    """
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


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def rad2deg(x, out=None, **kwargs):
    r"""
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : _Symbol or scalar
        Angles in degrees.
    out : _Symbol or None, optional
        A location into which the result is stored.

    Returns
    -------
    y : _Symbol or scalar
        The corresponding angle in radians.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    "rad2deg(x)" is "x * 180 / pi".

    This function differs from the original numpy.arange in the following aspects:
        - Only support float32 and float64.
        - `out` must be in the same size of input.
    """
    return _unary_func_helper(x, _npi.rad2deg, _np.rad2deg, out=out)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
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
@wrap_np_unary_func
def log2(x, out=None, **kwargs):
    """
    Base-2 logarithm of x.
    Parameters
    ----------
    x : _Symbol
        Input values.
    out : _Symbol or None
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
@wrap_np_unary_func
def log1p(x, out=None, **kwargs):
    """
    Return the natural logarithm of one plus the input array, element-wise.
    Calculates ``log(1 + x)``.
    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None
          Dummy parameter to keep the consistency with the ndarray counterpart.
    Returns
    -------
    y : _Symbol or scalar
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


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
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
@wrap_np_unary_func
def deg2rad(x, out=None, **kwargs):
    r"""
    deg2rad(x, out=None)

    Convert angles from degrees to radians.

    Parameters
    ----------
    x : _Symbol or scalar
        Angles in degrees.
    out : _Symbol or None, optional
        A location into which the result is stored.

    Returns
    -------
    y : _Symbol or scalar
        The corresponding angle in radians.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    "deg2rad(x)" is "x * pi / 180".

    This function differs from the original numpy.arange in the following aspects:
        - Only support float32 and float64.
        - `out` must be in the same size of input.
    """
    return _unary_func_helper(x, _npi.deg2rad, _np.deg2rad, out=out)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def reciprocal(x, out=None, **kwargs):
    r"""
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
@wrap_np_unary_func
def square(x, out=None, **kwargs):
    r"""
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
@wrap_np_unary_func
def negative(x, out=None, **kwargs):
    r"""
    Numerical negative, element-wise.

    Parameters:
    ------------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None, optional
          A location into which the result is stored.
          If provided, it must have a shape that the inputs broadcast to.
          If not provided or None, a freshly-allocated array is returned.
          A tuple (possible only as a keyword argument) must have length
          equal to the number of outputs.

    Returns:
    -------
    y : _Symbol or scalar
        Returned array or scalar: y = -x. This is a scalar if x is a scalar.

    Examples:
    ---------
    >>> np.negative(1)
    -1
    """
    return _unary_func_helper(x, _npi.negative, _np.negative, out=out)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def fix(x, out=None, **kwargs):
    """
    Round to nearest integer towards zero.

    Round an array of floats element-wise to nearest integer towards zero. The rounded values are returned as floats.

    Parameters:
    ----------
    x : _Symbol or scalar
        An array of floats to be rounded
    out : _Symbol or scalar, optional
          Output array

    Returns:
    ---------
    y : _Symbol or scalar

    Examples:
    ----------
    >>> np.fix(3.14)
    3
    """
    return _unary_func_helper(x, _npi.fix, _np.fix, out=out)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def tan(x, out=None, **kwargs):
    r"""
    Compute tangent element-wise.
    Equivalent to np.sin(x)/np.cos(x) element-wise.

    Parameters:
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or scalar or None.
        A location into which the result is stored. If provided,
        it must have a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a keyword argument)
        must have length equal to the number of outputs.

    Returns:
    -------
    y : _Symbol or scalar
        The corresponding tangent values. This is a scalar if x is a scalar.
    """

    return _unary_func_helper(x, _npi.tan, _np.tan, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def ceil(x, out=None, **kwargs):
    r"""
    Return the ceiling of the input, element-wise.
    The ceil of the ndarray `x` is the smallest integer `i`, such that
    `i >= x`.  It is often denoted as :math:`\lceil x \rceil`.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None
          Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol or scalar
        The ceiling of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.ceil(a)
    array([-1., -1., -0.,  1.,  2.,  2.,  2.])
    >>> #if you use parameter out, x and out must be ndarray. if not, you will get an error!
    >>> a = np.array(1)
    >>> np.ceil(np.array(3.5), a)
    array(4.)
    >>> a
    array(4.)
    """
    return _unary_func_helper(x, _npi.ceil, _np.ceil, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def insert(arr, obj, values, axis=None):
    """
    Insert values along the given axis before the given indices.

    Parameters
    ----------
    arr : _Symbol
        Input array.
    obj : int, slice or ndarray of int64
        Object that defines the index or indices before which `values` is
        inserted.
        Support for multiple insertions when `obj` is a single scalar or a
        sequence with one element (only support int32 and int64 element).
    values : _Symbol
        Values to insert into `arr`.
        If the type of values is different from that of arr, values is converted
        to the type of arr.
    axis : int, optional
        Axis along which to insert `values`.  If `axis` is None then `arr`
        is flattened first.

    Returns
    -------
    out : _Symbol
        A copy of `arr` with `values` inserted.  Note that `insert`
        does not occur in-place: a new array is returned. If
        `axis` is None, `out` is a flattened array.

    Notes
    -----
    - Note that for higher dimensional inserts `obj=0` behaves very different
    from `obj=[0]` just like `arr[:,0,:] = values` is different from
    `arr[:,[0],:] = values`.
    - If obj is a ndarray, it's dtype only supports int64
    """
    if isinstance(values, numeric_types):
        if isinstance(obj, slice):
            start = obj.start
            stop = obj.stop
            step = 1 if obj.step is None else obj.step
            return _npi.insert_slice(arr, val=values, start=start, stop=stop, step=step, axis=axis)
        elif isinstance(obj, integer_types):
            return _npi.insert_scalar(arr, val=values, int_ind=obj, axis=axis)
        elif isinstance(obj, Symbol):
            return _npi.insert_tensor(arr, obj, val=values, axis=axis)
    if not isinstance(arr, Symbol): # pylint: disable= undefined-variable
        raise TypeError("'arr' can not support type {}".format(str(type(arr))))
    if not isinstance(values, Symbol): # pylint: disable= undefined-variable
        raise TypeError("'values' can not support type {}".format(str(type(values))))
    if isinstance(obj, slice):
        start = obj.start
        stop = obj.stop
        step = 1 if obj.step is None else obj.step
        return _npi.insert_slice(arr, values, start=start, stop=stop, step=step, axis=axis)
    elif isinstance(obj, integer_types):
        return _npi.insert_scalar(arr, values, int_ind=obj, axis=axis)
    elif isinstance(obj, Symbol):
        return _npi.insert_tensor(arr, values, obj, axis=axis)
    else:
        raise TypeError("'obj' can not support type {}".format(str(type(obj))))


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def floor(x, out=None, **kwargs):
    r"""
    Return the floor of the input, element-wise.
    The floor of the ndarray `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None
          Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol or scalar
        The floor of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.

    Examples
    --------
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.floor(a)
    array([-2., -2., -1.,  0.,  1.,  1.,  2.])
    >>> # if you use parameter out, x and out must be ndarray. if not, you will get an error!
    >>> a = np.array(1)
    >>> np.floor(np.array(3.5), a)
    array(3.)
    >>> a
    array(3.)
    """
    return _unary_func_helper(x, _npi.floor, _np.floor, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def trunc(x, out=None, **kwargs):
    r"""
    Return the truncated value of the input, element-wise.
    The truncated value of the scalar `x` is the nearest integer `i` which
    is closer to zero than `x` is. In short, the fractional part of the
    signed number `x` is discarded.

    Parameters
    ----------
    x : _Symbol or scalar
        Input data.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol or scalar
        The truncated value of each element in `x`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    This function differs from the original numpy.trunc in the following aspects:
        - Do not support `where`, a parameter in numpy which indicates where to calculate.
        - Cannot cast type automatically. Dtype of `out` must be same as the expected one.
        - Cannot broadcast automatically. Shape of `out` must be same as the expected one.
        - If `x` is plain python numeric, the result won't be stored in out.
    """
    return _unary_func_helper(x, _npi.trunc, _np.trunc, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def logical_not(x, out=None, **kwargs):
    r"""
    Compute the truth value of NOT x element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        Logical NOT is applied to the elements of `x`.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : bool or _Symbol
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
    """
    return _unary_func_helper(x, _npi.logical_not, _np.logical_not, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def arcsinh(x, out=None, **kwargs):
    r"""
    Inverse hyperbolic sine, element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    arcsinh : _Symbol
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
    """
    return _unary_func_helper(x, _npi.arcsinh, _np.arcsinh, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def arccosh(x, out=None, **kwargs):
    r"""
    Inverse hyperbolic cosine, element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    arccosh : _Symbol
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
    """
    return _unary_func_helper(x, _npi.arccosh, _np.arccosh, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def arctanh(x, out=None, **kwargs):
    r"""
    Inverse hyperbolic tangent, element-wise.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None, optional
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    arctanh : _Symbol
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
    """
    return _unary_func_helper(x, _npi.arctanh, _np.arctanh, out=out, **kwargs)


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
        The type of the output array.
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.

    Returns
    -------
    arange : _Symbol
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.
    """
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
def delete(arr, obj, axis=None):
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by
    `arr[obj]`.

    Parameters
    ----------
    arr : _Symbol
      Input array.
    obj : slice, scaler or _Symbol of ints
      Indicate indices of sub-arrays to remove along the specified axis.
    axis : scaler, optional
      The axis along which to delete the subarray defined by `obj`.
      If `axis` is None, `obj` is applied to the flattened array.

    Returns
    -------
    out : _Symbol
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is None, `out` is
        a flattened array.
    """
    if not isinstance(arr, Symbol):
        raise TypeError("'arr' can not support type {}".format(str(type(arr))))
    if isinstance(obj, slice):
        start = obj.start
        stop = obj.stop
        step = 1 if obj.step is None else obj.step
        return _npi.delete(arr, start=start, stop=stop, step=step, axis=axis)
    elif isinstance(obj, integer_types):
        return _npi.delete(arr, int_ind=obj, axis=axis)
    elif isinstance(obj, Symbol):
        return _npi.delete(arr, obj, axis=axis)
    else:
        raise TypeError("'obj' can not support type {}".format(str(type(obj))))


# pylint: disable=redefined-outer-name
@set_module('mxnet.symbol.numpy')
def split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : _Symbol
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
    sub-arrays : _Symbol
        A list of sub-arrays.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division."""
    indices = []
    sections = 0
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, (list, set, tuple)):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple / list / set of ints')
    return _npi.split(ary, indices, axis, False, sections)
# pylint: enable=redefined-outer-name


# pylint: disable=redefined-outer-name
@set_module('mxnet.symbol.numpy')
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
    ary : _Symbol
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D Python tuple, list or set.
        Param used to determine the number and size of the subarray.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list of ndarrays
        A list of sub-arrays.
    """
    indices = []
    sections = 0
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, (list, set, tuple)):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple / list / set of ints')
    ret = _npi.array_split(ary, indices, axis, False, sections)
    if not isinstance(ret, list):
        return [ret]
    return ret
# pylint: enable=redefined-outer-name


# pylint: disable=redefined-outer-name
@set_module('mxnet.symbol.numpy')
def hsplit(ary, indices_or_sections):
    """Split an array into multiple sub-arrays horizontally (column-wise).

    This is equivalent to ``split`` with ``axis=0`` if ``ary`` has one
    dimension, and otherwise that with ``axis=1``.

    Parameters
    ----------
    ary : _Symbol
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
    sub-arrays : _Symbol
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
    indices = []
    sections = 0
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, (list, set, tuple)):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple of ints')
    return _npi.hsplit(ary, indices, 1, False, sections)
# pylint: enable=redefined-outer-name


# pylint: disable=redefined-outer-name
@set_module('mxnet.symbol.numpy')
def vsplit(ary, indices_or_sections):
    r"""
    vsplit(ary, indices_or_sections)

    Split an array into multiple sub-arrays vertically (row-wise).

    ``vsplit`` is equivalent to ``split`` with `axis=0` (default): the array is always split
    along the first axis regardless of the array dimension.

    Parameters
    ----------
    ary : _Symbol
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
    sub-arrays : list of _Symbols
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
    tuple and list
    - In ``indices_or_sections``, if an index exceeds the dimension of the array along axis 0,
    an error will be thrown.

    """
    indices = []
    sections = 0
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, (list, set, tuple)):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple of ints')
    return _npi.split(ary, indices, 0, False, sections)
# pylint: enable=redefined-outer-name


# pylint: disable=redefined-outer-name
@set_module('mxnet.symbol.numpy')
def dsplit(ary, indices_or_sections):
    """
    Split array into multiple sub-arrays along the 3rd axis (depth).

    Please refer to the `split` documentation.  `dsplit` is equivalent
    to `split` with ``axis=2``, the array is always split along the third
    axis provided the array dimension is greater than or equal to 3.

    Parameters
    ----------
    ary : _Symbol
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D Python tuple, list or set.
        If `indices_or_sections` is an integer, N, the array will be divided into N equal arrays
        along axis 2.  If such a split is not possible, an error is raised.

        If `indices_or_sections` is a 1-D array of sorted integers, the entries indicate where
        along axis 2 the array is split.  For example, ``[2, 3]`` would result in

          - ary[:, :, :2]
          - ary[:, :, 2:3]
          - ary[:, :, 3:]

        If an index exceeds the dimension of the array along axis 2, an error will be thrown.
    """
    indices = []
    sections = 0
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, (list, set, tuple)):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple of ints')
    return _npi.dsplit(ary, indices, 2, False, sections)
# pylint: enable=redefined-outer-name


@set_module('mxnet.symbol.numpy')
def concatenate(seq, axis=0, out=None):
    """Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : sequence of _Symbols
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
    res : _Symbol
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


@set_module('mxnet.symbol.numpy')
def append(arr, values, axis=None):  # pylint: disable=redefined-outer-name
    """
    Append values to the end of an array.

    Parameters
    ----------
    arr : _Symbol
        Values are appended to a copy of this array.
    values : _Symbol
        These values are appended to a copy of `arr`.  It must be of the
        correct shape (the same shape as `arr`, excluding `axis`).  If
        `axis` is not specified, `values` can be any shape and will be
        flattened before use.
    axis : int, optional
        The axis along which `values` are appended.  If `axis` is not
        given, both `arr` and `values` are flattened before use.

    Returns
    -------
    append : _Symbol
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


@set_module('mxnet.symbol.numpy')
def stack(arrays, axis=0, out=None):
    """Join a sequence of arrays along a new axis.
        The axis parameter specifies the index of the new axis in the dimensions of the result.
        For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last dimension.
    Parameters
    ----------
    arrays : sequence of _Symbols
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : _Symbol, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what stack would have returned if no out argument were specified.
    Returns
    -------
    stacked : _Symbol
        The stacked array has one more dimension than the input arrays."""
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]
    arrays = get_list(arrays)
    return _npi.stack(*arrays, axis=axis, out=out)


@set_module('mxnet.symbol.numpy')
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
    tup : sequence of _Symbol
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : _Symbol
        The array formed by stacking the given arrays, will be at least 2-D.
    """
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]
    arrays = get_list(arrays)
    return _npi.vstack(*arrays)


@set_module('mxnet.symbol.numpy')
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
    tup : sequence of _Symbol
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.
    Returns
    -------
    stacked : _Symbol
        The array formed by stacking the given arrays, will be at least 2-D.
    """
    def get_list(arrays):
        if not hasattr(arrays, '__getitem__') and hasattr(arrays, '__iter__'):
            raise ValueError("expected iterable for arrays but got {}".format(type(arrays)))
        return [arr for arr in arrays]

    arrays = get_list(arrays)
    return _npi.vstack(*arrays)


@set_module('mxnet.symbol.numpy')
def column_stack(tup):
    """
    Stack 1-D arrays as columns into a 2-D array.

    Take a sequence of 1-D arrays and stack them as columns
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `hstack`.  1-D arrays are turned into 2-D columns
    first.

    Parameters
    ----------
    tup : sequence of 1-D or 2-D arrays.
        Arrays to stack. All of them must have the same first dimension.

    Returns
    -------
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


@set_module('mxnet.symbol.numpy')
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
    tup : _Symbol
        The arrays must have the same shape along all but the second axis, except 1-D arrays which can be any length.

    Returns
    -------
    stacked : _Symbol
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


@set_module('mxnet.symbol.numpy')
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
    tup : sequence of _Symbol
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : _Symbol
        The array formed by stacking the given arrays, will be at least 2-D.
    """
    return _npi.dstack(*arrays)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def maximum(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.maximum, _np.maximum, _npi.maximum_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def fmax(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.fmax, _np.fmax, _npi.fmax_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def minimum(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.minimum, _np.minimum, _npi.minimum_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def fmin(x1, x2, out=None, **kwargs):
    return _ufunc_helper(x1, x2, _npi.fmin, _np.fmin, _npi.fmin_scalar, None, out)


@set_module('mxnet.symbol.numpy')
def max(a, axis=None, out=None, keepdims=False):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : _Symbol
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
    max : _Symbol
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
    """
    return _npi.max(a, axis=axis, keepdims=keepdims, out=out)


@set_module('mxnet.symbol.numpy')
def min(a, axis=None, out=None, keepdims=False):
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
    """
    return _npi.min(a, axis=axis, keepdims=keepdims, out=out)


@set_module('mxnet.symbol.numpy')
def amax(a, axis=None, out=None, keepdims=False):
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
    """
    return _npi.amax(a, axis=axis, keepdims=keepdims, out=out)


@set_module('mxnet.symbol.numpy')
def amin(a, axis=None, out=None, keepdims=False):
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
    """
    return _npi.amin(a, axis=axis, keepdims=keepdims, out=out)


@set_module('mxnet.symbol.numpy')
def all(a, axis=None, out=None, keepdims=False):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : _Symbol
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (axis = None) is to perform a logical AND over
        all the dimensions of the input array.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
    out : ndarray, optional
        Alternate output array in which to place the result. It must have
        the same shape as the expected output and its type is preserved

    Returns
    --------
    all : _Symbol, bool
        A new boolean or array is returned unless out is specified,
        in which case a reference to out is returned.
    """
    return _npi.all(a, axis=axis, keepdims=keepdims, out=out)


@set_module('mxnet.symbol.numpy')
def any(a, axis=None, out=None, keepdims=False):
    """
    Test whether any array element along a given axis evaluates to True.
    Returns single boolean unless axis is not None

    Parameters
    ----------
    a : _Symbol
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (axis = None) is to perform a logical AND over
        all the dimensions of the input array.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
    out : ndarray, optional
        Alternate output array in which to place the result. It must have
        the same shape as the expected output and its type is preserved

    Returns
    --------
    any : bool or _Symbol
        A new boolean or ndarray is returned unless out is specified,
        in which case a reference to out is returned.
    """
    return _npi.any(a, axis=axis, keepdims=keepdims, out=out)


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
def argmax(a, axis=None, out=None):
    r"""
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
    <https://numpy.org/doc/stable/reference/generated/numpy.argmax.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` symbol's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` symnbol's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    """
    return _npi.argmax(a, axis=axis, keepdims=False, out=out)


@set_module('mxnet.symbol.numpy')
def argmin(a, axis=None, out=None):
    r"""
    Returns the indices of the minimum values along an axis.

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
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.

    This function differs from the original `numpy.argmin
    <https://numpy.org/doc/stable/reference/generated/numpy.argmin.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` symbol's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` symnbol's dtype must be the same as the expected output.
    - ``out`` param does not support scalar input case.

    """
    return _npi.argmin(a, axis=axis, keepdims=False, out=out)


def average(a, axis=None, weights=None, returned=False, out=None):
    """
    Compute the weighted average along the specified axis.

    Parameters
    --------
    a : _Symbol
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
    weights : _Symbol, optional
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
    out : _Symbol, optional
        If provided, the calculation is done into this array.

    Returns
    --------
    retval, [sum_of_weights] : _Symbol
        Return the average along the specified axis.
        When returned is True, return a tuple with the average as the first element
        and the sum of the weights as the second element. sum_of_weights is of the same type as retval.
        If a is integral, the result dtype will beyour current default dtype,
        When npx.is_np_default_dtype() returns False, default dtype is float32,
        When npx.is_np_default_dtype() returns True, default dtype is float64;
        otherwise it will be the same as dtype of a.

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
    - Integral a results in float32 or float64 returned dtype, which depends on your current default dtype


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
        Type to use in computing the mean.
        For integer inputs, When npx.is_np_default_dtype() returns False, default dtype is float32,
        When npx.is_np_default_dtype() returns True, default dtype is float64;
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
    - default data type for integer input is float32 or float64, which depends on your current default dtype

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
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):  # pylint: disable=too-many-arguments
    """
    Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : `_Symbol`
        _Symbol containing numbers whose standard deviation is desired.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviations are computed.
        The default is to compute the standard deviation of the flattened array.
        If this is a tuple of ints, computation is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the standard deviation. For integer inputs, the default is float32;
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
        If out=None, returns a new array containing the standard deviation values,
        otherwise a reference to the output array is returned.

    Notes
    -----
    This function differs from the original `numpy.std
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html>`_ in
    the following way(s):

    - only _Symbol is accepted as valid input, python iterables or scalar is not supported
    - default output data type for integer input is float32

    """
    return _npi.std(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, out=out)


@set_module('mxnet.symbol.numpy')
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):  # pylint: disable=too-many-arguments
    """
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : `_Symbol`
        _Symbol containing numbers whose variance is desired.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.
        The default is to compute the variance of the flattened array.
        If this is a tuple of ints, computation is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the variance.
        For arrays of integer type,
        When npx.is_np_default_dtype() returns False, default dtype is float32,
        When npx.is_np_default_dtype() returns True, default dtype is float64;
        For arrays of float types it is the same as the array type.
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
        If out=None, returns a new array containing the variance values,
        otherwise a reference to the output array is returned.

    Notes
    -----
    This function differs from the original `numpy.var
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html>`_ in
    the following way(s):

    - only _Symbol is accepted as valid input, python iterables or scalar is not supported
    - default output data type for integer input is float32

    """
    return _npi.var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, out=out)


# pylint: disable=redefined-outer-name
@set_module('mxnet.symbol.numpy')
def indices(dimensions, dtype=None, ctx=None):
    """Return an array representing the indices of a grid.

    Compute an array where the subarrays contain index values 0,1,...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : data-type, optional
        The desired data-type for the array. Default is `int64`.
    ctx : device context, optional
        Device context on which the memory is allocated. Default is
        `mxnet.context.current_context()`.

    Returns
    -------
    grid : _Symbol
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
           [1, 1, 1]], dtype=int64)
    >>> grid[1]        # column indices
    array([[0, 0, 0],
           [1, 1, 1]], dtype=int64)

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


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def copysign(x1, x2, out=None, **kwargs):
    r"""
    Change the sign of x1 to that of x2, element-wise.

    If `x2` is a scalar, its sign will be copied to all elements of `x1`.

    Parameters
    ----------
    x1 : _Symbol or scalar
        Values to change the sign of.
    x2 : _Symbol or scalar
        The sign of `x2` is copied to `x1`.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    out : _Symbol
        The values of `x1` with the sign of `x2`.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -------
    This function differs from the original `numpy.copysign
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.copysign.html>`_ in
    the following aspects:

    - ``where`` param is not supported.
    """
    return _ufunc_helper(x1, x2, _npi.copysign, _np.copysign, _npi.copysign_scalar, _npi.rcopysign_scalar, out)


@set_module('mxnet.symbol.numpy')
def ravel(x, order='C'):
    r"""
    ravel(x)

    Return a contiguous flattened array.
    A 1-D array, containing the elements of the input, is returned.  A copy is
    made only if needed.

    Parameters
    ----------
    x : _Symbol
        Input array.  The elements in `x` are read in row-major, C-style order and
        packed as a 1-D array.
    order : `C`, optional
        Only support row-major, C-style order.

    Returns
    -------
    y : _Symbol
        y is an array of the same subtype as `x`, with shape ``(x.size,)``.
        Note that matrices are special cased for backward compatibility, if `x`
        is a matrix, then y is a 1-D ndarray.

    Notes
    -----
    This function differs from the original numpy.arange in the following aspects:
        - Only support row-major, C-style order.
    """
    if order == 'F':
        raise NotImplementedError('order {} is not supported'.format(order))
    if isinstance(x, numeric_types):
        return _np.reshape(x, -1)
    elif isinstance(x, _Symbol):
        return reshape(x, -1)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


def unravel_index(indices, shape, order='C'): # pylint: disable=redefined-outer-name
    """
    Converts a flat index or array of flat indices into a tuple of coordinate arrays.

    Parameters:
    -------------
    indices : _Symbol
            An integer array whose elements are indices into the flattened version of an array of dimensions shape.
            Before version 1.6.0, this function accepted just one index value.
    shape : tuple of ints
            The shape of the array to use for unraveling indices.

    Returns:
    -------------
    unraveled_coords : _Symbol
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
        return _npi.unravel_index_fallback(indices, shape=shape)
    else:
        raise NotImplementedError('Don not support column-major (Fortran-style) order at this moment')


def flatnonzero(a):
    r"""
    Return indices that are non-zero in the flattened version of a.

    This is equivalent to np.nonzero(np.ravel(a))[0].

    Parameters
    ----------
    a : _Symbol
        Input data.

    Returns
    -------
    res : _Symbol
        Output array, containing the indices of the elements of `a.ravel()`
        that are non-zero.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    ravel : Return a 1-D array containing the elements of the input array.
    """
    out = _npi.nonzero(ravel(a))
    return out.reshape(-1,)


def diag_indices_from(arr):
    """
    This returns a tuple of indices that can be used to access the main diagonal of an array
    a with a.ndim >= 2 dimensions and shape (n, n, ..., n). For a.ndim = 2 this is
    the usual diagonal, for a.ndim > 2 this is the set of indices to access
    a[i, i, ..., i] for i = [0..n-1].

    Parameters:
    -------------
    arr : _Symbol
        Input array for acessing the main diagonal. All dimensions
        should have equal length.

    Return:
    -------------
    diag: _Symbol
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
    return _npi.diag_indices_from(arr)


@set_module('mxnet.symbol.numpy')
def hanning(M, dtype=None, ctx=None):
    r"""Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : _Symbol, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
        Note that you need select numpy.float32 or float64 in this operator.

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


@set_module('mxnet.symbol.numpy')
def hamming(M, dtype=None, ctx=None):
    r"""Return the hamming window.

    The hamming window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : _Symbol, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
        Note that you need select numpy.float32 or float64 in this operator.

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


@set_module('mxnet.symbol.numpy')
def blackman(M, dtype=None, ctx=None):
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
    ctx : Context, optional
        An optional device context (default is the current default context).

    Returns
    -------
    out : _Symbol
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).
        When npx.is_np_default_dtype() returns False, default dtype is float32;
        When npx.is_np_default_dtype() returns True, default dtype is float64.
        Note that you need select numpy.float32 or float64 in this operator.

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


@set_module('mxnet.symbol.numpy')
def flip(m, axis=None, out=None):
    r"""
    flip(m, axis=None, out=None)

    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : _Symbol or scalar
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to flip over. The default,
        axis=None, will flip over all of the axes of the input array.
        If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, flipping is performed on all of the axes
        specified in the tuple.
    out : _Symbol or scalar, optional
        Alternative output array in which to place the result. It must have
        the same shape and type as the expected output.

    Returns
    -------
    out : _Symbol or scalar
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.
    """
    if isinstance(m, numeric_types):
        return _np.flip(m, axis)
    elif isinstance(m, _Symbol):
        return _npi.flip(m, axis, out=out)
    else:
        raise TypeError('type {} not supported'.format(str(type(m))))


@set_module('mxnet.symbol.numpy')
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
    """
    return flip(m, 0)


@set_module('mxnet.symbol.numpy')
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
    """
    return flip(m, 1)


@set_module('mxnet.symbol.numpy')
def around(x, decimals=0, out=None, **kwargs):
    r"""
    around(x, decimals=0, out=None)

    Evenly round to the given number of decimals.
    Parameters
    ----------
    x : _Symbol or scalar
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).  If
        decimals is negative, it specifies the number of positions to
        the left of the decimal point.
    out : _Symbol, optional
        Alternative output array in which to place the result. It must have
        the same shape and type as the expected output.

    Returns
    -------
    rounded_array : _Symbol or scalar
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
    """
    if isinstance(x, numeric_types):
        return _np.around(x, decimals, **kwargs)
    elif isinstance(x, _Symbol):
        return _npi.around(x, decimals, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.symbol.numpy')
def round(x, decimals=0, out=None, **kwargs):
    r"""
    round(a, decimals=0, out=None)
    Round an array to the given number of decimals.

    See Also
    --------
    around : equivalent function; see for details.
    """
    if isinstance(x, numeric_types):
        return _np.around(x, decimals, **kwargs)
    elif isinstance(x, _Symbol):
        return _npi.around(x, decimals, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.symbol.numpy')
def round_(x, decimals=0, out=None, **kwargs):
    r"""
    round_(a, decimals=0, out=None)
    Round an array to the given number of decimals.

    See Also
    --------
    around : equivalent function; see for details.
    """
    if isinstance(x, numeric_types):
        return _np.around(x, decimals, **kwargs)
    elif isinstance(x, _Symbol):
        return _npi.around(x, decimals, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.symbol.numpy')
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
    x1 : _Symbol or scalar
        `y`-coordinates.
    x2 : _Symbol or scalar
        `x`-coordinates. `x2` must be broadcastable to match the shape of
        `x1` or vice versa.
    out : _Symbol or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    out : _Symbol or scalar
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
    """
    return _ufunc_helper(x1, x2, _npi.arctan2, _np.arctan2,
                         _npi.arctan2_scalar, _npi.rarctan2_scalar, out=out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def hypot(x1, x2, out=None, **kwargs):
    r"""
    Given the "legs" of a right triangle, return its hypotenuse.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise.  If `x1` or
    `x2` is scalar_like (i.e., unambiguously cast-able to a scalar type),
    it is broadcast for use with each element of the other argument.

    Parameters
    ----------
    x1, x2 : _Symbol or scalar
        Leg of the triangle(s).
    out : _Symbol or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    z : _Symbol or scalar
        The hypotenuse of the triangle(s).
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    This function differs from the original numpy.arange in the following aspects:
        - Only support float16, float32 and float64.
    """
    return _ufunc_helper(x1, x2, _npi.hypot, _np.hypot, _npi.hypot_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def bitwise_and(x1, x2, out=None, **kwargs):
    r"""
    Compute the bit-wise XOR of two arrays element-wise.

    Parameters
    ----------
    x1, x2 : _Symbol or scalar
        Only integer and boolean types are handled. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which becomes the shape of the output).
    out : _Symbol or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    out : _Symbol or scalar
        Result.
    """
    return _ufunc_helper(x1, x2, _npi.bitwise_and, _np.bitwise_and, _npi.bitwise_and_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def bitwise_xor(x1, x2, out=None, **kwargs):
    r"""
    Compute the bit-wise XOR of two arrays element-wise.

    Parameters
    ----------
    x1, x2 : _Symbol or scalar
        Only integer and boolean types are handled. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which becomes the shape of the output).
    out : _Symbol or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    out : _Symbol or scalar
        Result.
    """
    return _ufunc_helper(x1, x2, _npi.bitwise_xor, _np.bitwise_xor, _npi.bitwise_xor_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def bitwise_or(x1, x2, out=None, **kwargs):
    r"""
    Compute the bit-wise OR of two arrays element-wise.

    Parameters
    ----------
    x1, x2 : _Symbol or scalar
        Only integer and boolean types are handled. If x1.shape != x2.shape,
        they must be broadcastable to a common shape (which becomes the shape of the output).
    out : _Symbol or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.

    Returns
    -------
    out : _Symbol or scalar
        Result.
    """
    return _ufunc_helper(x1, x2, _npi.bitwise_or, _np.bitwise_or, _npi.bitwise_or_scalar, None, out)


@set_module('mxnet.symbol.numpy')
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
    ar : _Symbol
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
    unique : _Symbol
        The sorted unique values.
    unique_indices : _Symbol, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : _Symbol, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : _Symbol, optional
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
    """
    return _npi.unique(ar, return_index, return_inverse, return_counts, axis)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def ldexp(x1, x2, out=None, **kwargs):
    """
    Returns x1 * 2**x2, element-wise.
    The mantissas `x1` and twos exponents `x2` are used to construct
    floating point numbers ``x1 * 2**x2``.

    Parameters
    ----------
    x1 : _Symbol
        Array of multipliers.
    x2 : _Symbol
        Array of twos exponents.
    out : _Symbol or None
        Dummy parameter to keep the consistency with the ndarray counterpart.

    Returns
    -------
    y : _Symbol
        The result of ``x1 * 2**x2``.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.
    Different from numpy, we allow x2 to be float besides int.
    `ldexp` is useful as the inverse of `frexp`, if used by itself it is
    more clear to simply use the expression ``x1 * 2**x2``.
    """
    return _ufunc_helper(x1, x2, _npi.ldexp, _np.ldexp, _npi.ldexp_scalar, _npi.rldexp_scalar, out)


@set_module('mxnet.symbol.numpy')
def vdot(a, b):
    r"""
    Return the dot product of two vectors.
    Note that `vdot` handles multidimensional arrays differently than `dot`:
    it does *not* perform a matrix product, but flattens input arguments
    to 1-D vectors first. Consequently, it should only be used for vectors.

    Parameters
    ----------
    a : _Symbol
        First argument to the dot product.
    b : _Symbol
        Second argument to the dot product.

    Returns
    -------
    output : _Symbol
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


@set_module('mxnet.symbol.numpy')
def inner(a, b):
    r"""Inner product of two arrays.
    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : _Symbol
        If `a` and `b` are nonscalar, their last dimensions must match.

    Returns
    -------
    out : _Symbol
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


@set_module('mxnet.symbol.numpy')
def outer(a, b):
    r"""Compute the outer product of two vectors.
    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::
    [[a0*b0  a0*b1 ... a0*bN ]
    [a1*b0    .
    [ ...          .
    [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : (M,) _Symbol
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) _Symbol
        Second input vector.  Input is flattened if
        not already 1-dimensional.

    Returns
    -------
    out : (M, N) _Symbol
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


@set_module('mxnet.symbol.numpy')
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None): # pylint: disable=too-many-arguments
    """
    Return the cross product of two (arrays of) vectors.

    The cross product of `a` and `b` in :math:`R^3` is a vector perpendicular
    to both `a` and `b`.  If `a` and `b` are arrays of vectors, the vectors
    are defined by the last axis of `a` and `b` by default, and these axes
    can have dimensions 2 or 3.  Where the dimension of either `a` or `b` is
    2, the third component of the input vector is assumed to be zero and the
    cross product calculated accordingly.  In cases where both input vectors
    have dimension 2, the z-component of the cross product is returned.

    Parameters
    ----------
    a : _Symbol
        Components of the first vector(s).
    b : _Symbol
        Components of the second vector(s).
    axisa : int, optional
        Axis of `a` that defines the vector(s).  By default, the last axis.
    axisb : int, optional
        Axis of `b` that defines the vector(s).  By default, the last axis.
    axisc : int, optional
        Axis of `c` containing the cross product vector(s).  Ignored if
        both input vectors have dimension 2, as the return is scalar.
        By default, the last axis.
    axis : int, optional
        If defined, the axis of `a`, `b` and `c` that defines the vector(s)
        and cross product(s).  Overrides `axisa`, `axisb` and `axisc`.

    Returns
    -------
    c : _Symbol
        Vector cross product(s).

    Raises
    ------
    ValueError
        When the dimension of the vector(s) in `a` and/or `b` does not
        equal 2 or 3.

    Notes
    -----
    Supports full broadcasting of the inputs.
    """
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3

    return _npi.cross(a, b, axisa, axisb, axisc)


@set_module('mxnet.symbol.numpy')
def kron(a, b):
    r"""
    kron(a, b)
    Kronecker product of two arrays.
    Computes the Kronecker product, a composite array made of blocks of the
    second array scaled by the first.
    Parameters
    ----------
    a, b : ndarray
    Returns
    -------
    out : ndarray
    See Also
    --------
    outer : The outer product
    Notes
    -----
    The function assumes that the number of dimensions of `a` and `b`
    are the same, if necessary prepending the smallest with ones.
    If `a.shape = (r0,r1,..,rN)` and `b.shape = (s0,s1,...,sN)`,
    the Kronecker product has shape `(r0*s0, r1*s1, ..., rN*SN)`.
    The elements are products of elements from `a` and `b`, organized
    explicitly by::
        kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]
    where::
        kt = it * st + jt,  t = 0,...,N
    In the common 2-D case (N=1), the block structure can be visualized::
        [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],
        [  ...                              ...   ],
        [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]
    Examples
    --------
    >>> np.kron([1,10,100], [5,6,7])
    array([  5,   6,   7,  50,  60,  70, 500, 600, 700])
    >>> np.kron([5,6,7], [1,10,100])
    array([  5,  50, 500,   6,  60, 600,   7,  70, 700])
    """
    return _npi.kron(a, b)


@set_module('mxnet.symbol.numpy')
def equal(x1, x2, out=None):
    """
    Return (x1 == x2) element-wise.
    Parameters
    ----------
    x1, x2 : _Symbol or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : Dummy parameter, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : _Symbol or scalar
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


@set_module('mxnet.symbol.numpy')
def not_equal(x1, x2, out=None):
    """
    Return (x1 != x2) element-wise.
    Parameters
    ----------
    x1, x2 : _Symbol or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : Dummy parameter, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : _Symbol or scalar
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


@set_module('mxnet.symbol.numpy')
def greater(x1, x2, out=None):
    """
    Return the truth value of (x1 > x2) element-wise.
    Parameters
    ----------
    x1, x2 : _Symbol or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : Dummy parameter, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : _Symbol or scalar
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


@set_module('mxnet.symbol.numpy')
def less(x1, x2, out=None):
    """
    Return the truth value of (x1 < x2) element-wise.
    Parameters
    ----------
    x1, x2 : _Symbol or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : Dummy parameter, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : _Symbol or scalar
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


@set_module('mxnet.symbol.numpy')
def greater_equal(x1, x2, out=None):
    """
    Return the truth value of (x1 >= x2) element-wise.
    Parameters
    ----------
    x1, x2 : _Symbol or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : Dummy parameter, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : _Symbol or scalar
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


@set_module('mxnet.symbol.numpy')
def less_equal(x1, x2, out=None):
    """
    Return the truth value of (x1 <= x2) element-wise.
    Parameters
    ----------
    x1, x2 : _Symbol or scalars
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : Dummy parameter, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned.
    Returns
    -------
    out : _Symbol or scalar
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


@set_module('mxnet.symbol.numpy')
def roll(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at
    the first.

    Parameters
    ----------
    a : _Symbol
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
    res : _Symbol
        Output array, with the same shape as `a`.

    Notes
    -----
    Supports rolling over multiple dimensions simultaneously.
    """
    return _npi.roll(a, shift, axis=axis)


@wrap_np_binary_func
def logical_and(x1, x2, out=None):
    r"""
    Compute the truth value of x1 AND x2 element-wise.
    Parameters
    ----------
    x1, x2 : array_like
        Logical AND is applied to the elements of `x1` and `x2`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical AND operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    logical_or, logical_not, logical_xor, bitwise_or
    Examples
    --------
    >>> np.logical_and(True, False)
    False
    >>> np.logical_and(np.array([True, True], dtype='bool'), np.array([False, True], dtype='bool'))
    array([False,  True])
    """
    return _ufunc_helper(x1, x2, _npi.logical_and, _np.logical_and, _npi.logical_and_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def logical_or(x1, x2, out=None):
    r"""
    Compute the truth value of x1 OR x2 element-wise.
    Parameters
    ----------
    x1, x2 : array_like
        Logical OR is applied to the elements of `x1` and `x2`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical OR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    logical_and, logical_not, logical_xor, bitwise_or
    Examples
    --------
    >>> np.logical_or(True, False)
    True
    >>> np.logical_or(np.array([True, True], dtype='bool'), np.array([False, True], dtype='bool'))
    array([True,  True])
    """
    return _ufunc_helper(x1, x2, _npi.logical_or, _np.logical_or, _npi.logical_or_scalar, None, out)


@set_module('mxnet.symbol.numpy')
@wrap_np_binary_func
def logical_xor(x1, x2, out=None):
    r"""
    Compute the truth value of x1 XOR x2 element-wise.
    Parameters
    ----------
    x1, x2 : array_like
        Logical XOR is applied to the elements of `x1` and `x2`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical XOR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        This is a scalar if both `x1` and `x2` are scalars.
    See Also
    --------
    logical_and, logical_not, logical_or, bitwise_or
    Examples
    --------
    >>> np.logical_xor(True, False)
    True
    >>> np.logical_xor(np.array([True, True], dtype='bool'), np.array([False, True], dtype='bool'))
    array([ True, False])
    """
    return _ufunc_helper(x1, x2, _npi.logical_xor, _np.logical_xor, _npi.logical_xor_scalar, None, out)


@set_module('mxnet.symbol.numpy')
def rot90(m, k=1, axes=(0, 1)):
    """
    Rotate an array by 90 degrees in the plane specified by axes.
    Rotation direction is from the first towards the second axis.
    Parameters
    ----------
    m : _Symbol
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes: (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.
    Returns
    -------
    y : _Symbol
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


@set_module('mxnet.symbol.numpy')
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
    operands : list of _Symbol
        These are the arrays for the operation.
    out : _Symbol, optional
        If provided, the calculation is done into this array.
    optimize : {False, True}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False. Defaults to False.

    Returns
    -------
    output : _Symbol
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
    """
    # Grab non-einsum kwargs; do not optimize by default.
    optimize_arg = kwargs.pop('optimize', False)
    out = kwargs.pop('out', None)

    subscripts = operands[0]
    operands = operands[1:]
    return _npi.einsum(*operands, subscripts=subscripts, out=out, optimize=int(optimize_arg))


@set_module('mxnet.symbol.numpy')
def percentile(a, q, axis=None, out=None, overwrite_input=None, interpolation='linear', keepdims=False): # pylint: disable=too-many-arguments
    """
    Compute the q-th percentile of the data along the specified axis.
    Returns the q-th percentile(s) of the array elements.

    Parameters
    ----------
    a : _Symbol
        Input array
    q : _Symbol
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
    percentile : _Symbol
        Output array.
    """
    if overwrite_input is not None:
        raise NotImplementedError('overwrite_input is not supported yet')
    if isinstance(q, numeric_types):
        return _npi.percentile(a, axis=axis, interpolation=interpolation,
                               keepdims=keepdims, q_scalar=q, out=out)
    return _npi.percentile(a, q, axis=axis, interpolation=interpolation,
                           keepdims=keepdims, q_scalar=None, out=out)


@set_module('mxnet.symbol.numpy')
def median(a, axis=None, out=None, overwrite_input=None, keepdims=False):
    r"""
    Compute the median along the specified axis.
    Returns the median of the array elements.
    Parameters
    ----------
    a : _Symbol
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default
        is to compute the median along a flattened version of the array.
        A sequence of axes is supported since version 1.9.0.
    out :  _Symbol, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.
    Returns
    -------
    median :  _Symbol
        A new array holding the result. If the input contains integers
        or floats smaller than ``float32``, then the output data-type is
        ``np.float32``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.
    See Also
    --------
    mean, percentile
    """
    return quantile(a=a, q=0.5, axis=axis, out=out, overwrite_input=overwrite_input,
                    interpolation='midpoint', keepdims=keepdims)


@set_module('mxnet.symbol.numpy')
def quantile(a, q, axis=None, out=None, overwrite_input=None, interpolation='linear', keepdims=False): # pylint: disable=too-many-arguments
    """
    Compute the q-th quantile of the data along the specified axis.
    New in version 1.15.0.
    Parameters
    ----------
    a : _Symbol
        Input array or object that can be converted to an array.
    q : _Symbol
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
    quantile : _Symbol
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
    - q must be _Symbol type even if it is a scalar
    - do not support overwrite_input
    """
    if overwrite_input is not None:
        raise NotImplementedError('overwrite_input is not supported yet')
    if isinstance(q, numeric_types):
        return _npi.percentile(a, axis=axis, interpolation=interpolation,
                               keepdims=keepdims, q_scalar=q * 100, out=out)
    return _npi.percentile(a, q * 100, axis=axis, interpolation=interpolation,
                           keepdims=keepdims, q_scalar=None, out=out)


@set_module('mxnet.symbol.numpy')
def shares_memory(a, b, max_work=None):
    """
    Determine if two arrays share memory

    Parameters
    ----------
    a, b : _Symbol
        Input arrays

    Returns
    -------
    out : _Symbol
    """
    return _npi.share_memory(a, b)


@set_module('mxnet.symbol.numpy')
def may_share_memory(a, b, max_work=None):
    """
    Determine if two arrays might share memory

    A return of True does not necessarily mean that the two arrays
    share any element.  It just means that they *might*.

    Only the memory bounds of a and b are checked by default.

    Parameters
    ----------
    a, b : _Symbol
        Input arrays

    Returns
    -------
    out : _Symbol
    """
    return _npi.share_memory(a, b)


@set_module('mxnet.symbol.numpy')
def diff(a, n=1, axis=-1, prepend=None, append=None):  # pylint: disable=redefined-outer-name
    r"""
    Calculate the n-th discrete difference along the given axis.

    Parameters
    ----------
    a : _Symbol
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.
    prepend, append : _Symbol, optional
        Not supported yet

    Returns
    -------
    diff : _Symbol
        The n-th differences.
        The shape of the output is the same as a except along axis where the dimension is smaller by n.
        The type of the output is the same as the type of the difference between any two elements of a.
        This is the same as the type of a in most cases.

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


@set_module('mxnet.symbol.numpy')
def ediff1d(ary, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    Parameters
    ----------
    ary : _Symbol
        If necessary, will be flattened before the differences are taken.
    to_end : _Symbol or scalar, optional
        Number(s) to append at the end of the returned differences.
    to_begin : _Symbol or scalar, optional
        Number(s) to prepend at the beginning of the returned differences.

    Returns
    -------
    ediff1d : _Symbol
        The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.
    """
    input_type = (isinstance(to_begin, _Symbol), isinstance(to_end, _Symbol))
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


@set_module('mxnet.symbol.numpy')
def interp(x, xp, fp, left=None, right=None, period=None):  # pylint: disable=too-many-arguments
    """
    One-dimensional linear interpolation.
    Returns the one-dimensional piecewise linear interpolant to a function
    with given values at discrete data-points.

    Parameters
    ----------
    x : _Symbol
        The x-coordinates of the interpolated values.
    xp : _Symbol
        The x-coordinates of the data points, must be increasing if argument
        `period` is not specified. Otherwise, `xp` is internally sorted after
        normalizing the periodic boundaries with ``xp = xp % period``.
    fp : _Symbol
        The y-coordinates of the data points, same length as `xp`.
    left : optional float corresponding to fp
        Value to return for `x < xp[0]`, default is `fp[0]`.
    right : optional float corresponding to fp
        Value to return for `x > xp[-1]`, default is `fp[-1]`.
    period : None or float, optional
        A period for the x-coordinates. This parameter allows the proper
        interpolation of angular x-coordinates. Parameters `left` and `right`
        are ignored if `period` is specified.
        .. versionadded:: 1.10.0

    Returns
    -------
    y : _Symbol
        The interpolated values, same shape as `x`.

    Raises
    ------
    ValueError
        If `xp` and `fp` have different length
        If `xp` or `fp` are not 1-D sequences
        If `period == 0`

    Notes
    -----
    Does not check that the x-coordinate sequence `xp` is increasing.
    If `xp` is not increasing, the results are nonsense.
    A simple check for increasing is::
        np.all(np.diff(xp) > 0)
    """
    if isinstance(x, numeric_types):
        return _npi.interp(xp.astype(float), fp.astype(float), left=left,
                           right=right, period=period, x_scalar=x, x_is_scalar=True)
    return _npi.interp(xp.astype(float), fp.astype(float), x.astype(float), left=left,
                       right=right, period=period, x_scalar=0.0, x_is_scalar=False)


@set_module('mxnet.symbol.numpy')
def resize(a, new_shape):
    """
    Return a new array with the specified shape.
    If the new array is larger than the original array, then the new
    array is filled with repeated copies of `a`.  Note that this behavior
    is different from a.resize(new_shape) which fills with zeros instead
    of repeated copies of `a`.

    Parameters
    ----------
    a : _Symbol
        Array to be resized.
    new_shape : int or tuple of int
        Shape of resized array.

    Returns
    -------
    reshaped_array : _Symbol
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

# pylint: disable=redefined-outer-name
@set_module('mxnet.symbol.numpy')
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
    x : _Symbol
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
    out : _Symbol
        `x`, with the non-finite values replaced. If `copy` is False, this may
        be `x` itself.

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    """
    if isinstance(x, numeric_types):
        return _np.nan_to_num(x, copy, nan, posinf, neginf)
    elif isinstance(x, _Symbol):
        if not copy:
            return _npi.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=x)
        return _npi.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=None)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))


@set_module('mxnet.symbol.numpy')
def squeeze(x, axis=None):
    """
    Remove single-dimensional entries from the shape of an array.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        .. versionadded:: 1.7.0
        Selects a subset of the single-dimensional entries in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`.

    Raises
    ------
    ValueError
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
    ValueError: cannot select an axis to squeeze out which has size not equal to one
    >>> np.squeeze(x, axis=2).shape
    (1, 3)
    """
    return _npi.squeeze(x, axis=axis)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def isnan(x, out=None, **kwargs):
    """
    Test element-wise for NaN and return result as a boolean array.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : _Symbol or bool
        True where x is NaN, false otherwise.
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
    """
    return _unary_func_helper(x, _npi.isnan, _np.isnan, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def isinf(x, out=None, **kwargs):
    """
    Test element-wise for positive or negative infinity.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : _Symbol or bool
        True where x is positive or negative infinity, false otherwise.
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
    """
    return _unary_func_helper(x, _npi.isinf, _np.isinf, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def isposinf(x, out=None, **kwargs):
    """
    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : _Symbol or bool
        True where x is positive infinity, false otherwise.
        This is a scalar if x is a scalar.

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
    This means that Not a Number is not equivalent to infinity.
    """
    return _unary_func_helper(x, _npi.isposinf, _np.isposinf, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def isneginf(x, out=None, **kwargs):
    """
    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : _Symbol or bool
        True where x is negative infinity, false otherwise.
        This is a scalar if x is a scalar.

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic (IEEE 754).
    This means that Not a Number is not equivalent to infinity.
    """
    return _unary_func_helper(x, _npi.isneginf, _np.isneginf, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
@wrap_np_unary_func
def isfinite(x, out=None, **kwargs):
    """
    Test element-wise for finiteness (not infinity or not Not a Number).

    Parameters
    ----------
    x : _Symbol or scalar
        Input array.
    out : _Symbol or None, optional
        A location into which the result is stored.
        If provided, it must have the same shape and dtype as input ndarray.
        If not provided or `None`, a freshly-allocated array is returned.

    Returns
    -------
    y : _Symbol or bool
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
    """
    return _unary_func_helper(x, _npi.isfinite, _np.isfinite, out=out, **kwargs)


@set_module('mxnet.symbol.numpy')
def atleast_1d(*arys):
    """
    Convert inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.

    Parameters
    ----------
    arys1, arys2, ... : _Symbol
        One or more input arrays.

    Returns
    -------
    ret : _Symbol
        An array, or list of arrays, each with a.ndim >= 1. Copies are made only if necessary.

    See also
    --------
    atleast_2d, atleast_3d
    """
    return _npi.atleast_1d(*arys)


@set_module('mxnet.symbol.numpy')
def atleast_2d(*arys):
    """
    Convert inputs to arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : _Symbol
        One or more input arrays.

    Returns
    -------
    ret : _Symbol
        An array, or list of arrays, each with a.ndim >= 2. Copies are made only if necessary.

    See also
    --------
    atleast_1d, atleast_3d
    """
    return _npi.atleast_2d(*arys)


@set_module('mxnet.symbol.numpy')
def atleast_3d(*arys):
    """
    Convert inputs to arrays with at least three dimension.

    Parameters
    ----------
    arys1, arys2, ... : _Symbol
        One or more input arrays.

    Returns
    -------
    ret : _Symbol
        An array, or list of arrays, each with a.ndim >= 3.
        For example, a 1-D array of shape (N,) becomes a view of shape (1, N, 1),
        and a 2-D array of shape (M, N) becomes a view of shape (M, N, 1).

    See also
    --------
    atleast_1d, atleast_2d
    """
    return _npi.atleast_3d(*arys)


@set_module('mxnet.symbol.numpy')
def where(condition, x, y):
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    Parameters
    ----------
    condition : _Symbol
        Where True, yield `x`, otherwise yield `y`.
    x, y : _Symbol
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape. `x` and `y` must have the same dtype.

    Returns
    -------
    out : _Symbol
        An array with elements from `x` where `condition` is True, and elements
        from `y` elsewhere.

    """
    if isinstance(condition, numeric_types):
        if condition != 0:
            return x
        else:
            return y
    else:
        if isinstance(x, numeric_types) and isinstance(y, numeric_types):
            return _npi.where_scalar2(condition, float(x), float(y), out=None)
        elif isinstance(x, Symbol) and isinstance(y, Symbol):
            return _npi.where(condition, x, y, out=None)
        elif isinstance(y, Symbol):
            return _npi.where_lscalar(condition, y, float(x), out=None)
        elif isinstance(x, Symbol):
            return _npi.where_rscalar(condition, x, float(y), out=None)
        else:
            raise TypeError('type {0} and {1} not supported'.format(str(type(x)), str(type(y))))


@set_module('mxnet.symbol.numpy')
def load(fname):
    """Loads symbol from a JSON file.
    You can also use pickle to do the job if you only work on python.
    The advantage of load/save is the file is language agnostic.
    This means the file saved using save can be loaded by other language binding of mxnet.
    You also get the benefit being able to directly load/save from cloud storage(S3, HDFS).

    Parameters
    ----------
    fname : str
        The name of the file, examples:
        - `s3://my-bucket/path/my-s3-symbol`
        - `hdfs://my-bucket/path/my-hdfs-symbol`
        - `/path-to/my-local-symbol`

    Returns
    -------
    sym : _Symbol
        The loaded symbol.

    See Also
    --------
    _Symbol.save : Used to save symbol into file.
    """
    if not isinstance(fname, string_types):
        raise TypeError('fname needs to be string')
    handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateFromFile(c_str(fname), ctypes.byref(handle)))
    return _Symbol(handle)


@set_module('mxnet.symbol.numpy')
def load_json(json_str):
    """Loads symbol from json string.

    Parameters
    ----------
    json_str : str
        A JSON string.

    Returns
    -------
    sym : Symbol
        The loaded symbol.

    See Also
    --------
    _Symbol.tojson : Used to save symbol into json string.
    """
    if not isinstance(json_str, string_types):
        raise TypeError('json_str needs to be string')
    handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateFromJSON(c_str(json_str), ctypes.byref(handle)))
    return _Symbol(handle)


@set_module('mxnet.symbol.numpy')
def polyval(p, x):
    """
    Evaluate a polynomial at specific values.
    If p is of length N, this function returns the value:
    p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
    If x is a sequence, then p(x) is returned for each element of x.
    If x is another polynomial then the composite polynomial p(x(t)) is returned.

    Parameters
    ----------
    p : _Symbol
        1D array of polynomial coefficients (including coefficients equal to zero)
        from highest degree to the constant term.
    x : _Symbol
        An array of numbers, at which to evaluate p.

    Returns
    -------
    values : _Symbol
        Result array of polynomials

    Notes
    -----
    This function differs from the original `numpy.polyval
    <https://numpy.org/devdocs/reference/generated/numpy.polyval.html>`_ in
    the following way(s):
    - Does not support poly1d.
    - X should be ndarray type even if it contains only one element.
    """
    if isinstance(p, Symbol) and isinstance(x, Symbol):
        return _npi.polyval(p, x)
    elif not isinstance(p, Symbol) and not isinstance(x, Symbol):
        return _np.polyval(p, x)
    else:
        raise TypeError('type not supported')


@set_module('mxnet.symbol.numpy')
def bincount(x, weights=None, minlength=0):
    """
    Count number of occurrences of each value in array of non-negative ints.

    Parameters
    ----------
    x : _Symbol
        input data
    weights: _Symbol
        input weigths same shape as x. (Optional)
    minlength: int
        A minimum number of bins for the output. (Optional)

    Returns
    --------
    out : _Symbol
        the result of binning the input data. The length of out is equal to amax(x)+1.

    Raises:
    --------
    Value Error
        If the input is not 1-dimensional, or contains elements with negative values,
        or if minlength is negative
    TypeError
        If the type of the input is float or complex.
    """
    if minlength < 0:
        raise ValueError("Minlength value should greater than 0")
    if weights is None:
        return _npi.bincount(x, minlength=minlength, has_weights=False)
    return _npi.bincount(x, weights=weights, minlength=minlength, has_weights=True)


@set_module('mxnet.symbol.numpy')
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
                raise ValueError(f'{key} keyword not in allowed keywords {allowedkwargs[mode]}')

    unsupported_kwargs = set(kwargs) - set(allowedkwargs[mode])
    if unsupported_kwargs:
        raise ValueError("unsupported keyword arguments for mode '{}': {}"
                         .format(mode, unsupported_kwargs))
    if mode == "constant":
        values = kwargs.get("constant_values", 0)
        if isinstance(values, tuple):
            raise TypeError("unsupported constant_values type: {'tuple'}.")
        return _npi.pad(x, pad_width, mode='constant', constant_values=values)
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
    return _npi.pad(x, pad_width, mode='constant', constant_values=0)


@set_module('mxnet.symbol.numpy')
def prod(a, axis=None, dtype=None, keepdims=False, initial=None, output=None): # pylint: disable=too-many-arguments
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed.  The default,
        axis=None, will calculate the product of all the elements in the
        input array. If axis is negative it counts from the last to the
        first axis.
        .. versionadded:: 1.7.0
        If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the
        axes as before.
    dtype : dtype, optional
        The type of the returned array, as well as of the accumulator in
        which the elements are multiplied.  The dtype of `a` is used by
        default unless `a` has an integer dtype of less precision than the
        default platform integer.  In that case, if `a` is signed then the
        platform integer is used while if `a` is unsigned then an unsigned
        integer of the same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `prod` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The starting value for this product. See `~numpy.ufunc.reduce` for details.
    where : not supported

    Returns
    -------
    product_along_axis : ndarray, see `dtype` parameter above.
        An array shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    Examples
    --------
    By default, calculate the product of all elements:
    >>> np.prod([1.,2.])
    2.0
    Even when the input array is two-dimensional:
    >>> np.prod([[1.,2.],[3.,4.]])
    24.0
    But we can also specify the axis over which to multiply:
    >>> np.prod([[1.,2.],[3.,4.]], axis=1)
    array([  2.,  12.])
    Or select specific elements to include:
    >>> np.prod([1., np.nan, 3.], where=[True, False, True])
    3.0
    If the type of `x` is unsigned, then the output type is
    the unsigned platform integer:
    >>> x = np.array([1, 2, 3], dtype=np.uint8)
    >>> np.prod(x).dtype == np.uint
    True
    If `x` is of a signed integer type, then the output type
    is the default platform integer:
    >>> x = np.array([1, 2, 3], dtype=np.int8)
    >>> np.prod(x).dtype == int
    True
    You can also start the product with a value other than one:
    >>> np.prod([1, 2], initial=5)
    10
    """
    return _npi.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial)

@set_module('mxnet.symbol.numpy')
def cumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : _Symbol
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
    out : _Symbol, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See `doc.ufuncs`
        (Section "Output arguments") for more details.

    Returns
    -------
    cumsum_along_axis : _Symbol.
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned. The
        result has the same size as `a`, and the same shape as `a` if
        `axis` is not None or `a` is a 1-d array.
    """
    return _npi.cumsum(a, axis=axis, dtype=dtype, out=out)

@set_module('mxnet.symbol.numpy')
def reshape(a, newshape, reverse=False, order='C'):
    """
    Gives a new shape to an array without changing its data.
    This function always returns a copy of the input array if
    ``out`` is not provided.

    Parameters
    ----------
    a : _Symbol
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
    reshaped_array : _Symbol
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
    return _npi.reshape(a, newshape, reverse, order)

@set_module('mxnet.symbol.numpy')
def moveaxis(a, source, destination):
    """Move axes of an array to new positions.
    Other axes remain in their original order.

    Parameters
    ----------
    a : _Symbol
        The array whose axes should be reordered.
        source : int or sequence of int
        Original positions of the axes to move. These must be unique.
        destination : int or sequence of int
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : _Symbol
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
    return _npi.moveaxis(a, source, destination)

@set_module('mxnet.symbol.numpy')
def copy(a):  # pylint: disable=redefined-outer-name
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : _Symbol
        Input array.

    Returns
    -------
    arr : _Symbol
        Array interpretation of a.

    -----
    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)
    >>> x[0] = 10
    >>> x[0] == y[0]
        True
    >>> x[0] == z[0]
        False
    """
    return _npi.copy(a)

@set_module('mxnet.symbol.numpy')
def rollaxis(a, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    Parameters
    ----------
    a : _Symbol
        Input array.
    axis : integer
        The axis to roll backwards. The positions of the other axes do not
        change relative to one another.
    start: int, optional
        The axis is rolled until it lies before this position.
        The default, 0, results in a “complete” roll.

    Returns
    -------
    res : _Symbol
        A view after applying rollaxis to `a` is returned.

    -----
    Examples
    --------
    >>> a = np.ones((3,4,5,6))
    >>> np.rollaxis(a, 3, 1).shape
    (3, 6, 4, 5)
    >>> np.rollaxis(a, 2).shape
    (5, 3, 4, 6)
    >>> np.rollaxis(a, 1, 4).shape
    (3, 5, 6, 4)
    """
    return _npi.rollaxis(a, axis, start)


@set_module('mxnet.symbol.numpy')
def diag(v, k=0):
    """
    Extracts a diagonal or constructs a diagonal array.
    - 1-D arrays: constructs a 2-D array with the input as its diagonal, all other elements are zero.
    - 2-D arrays: extracts the k-th Diagonal

    Parameters
    ----------
    array : _Symbol
        The array to apply diag method.
    k : offset
        extracts or constructs kth diagonal given input array

    Returns
    ----------
    out : _Symbol
    The extracted diagonal or constructed diagonal array.
    """
    return _npi.diag(v, k=k)


@set_module('mxnet.symbol.numpy')
def diagflat(v, k=0):
    """
    Create a two-dimensional array with the flattened input as a diagonal.

    Parameters
    ----------
    v : array_like
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
    return _npi.diagflat(v, k=k)


@set_module('mxnet.symbol.numpy')
def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    If a is 2-D, returns the diagonal of a with the given offset, i.e., the collection of elements of
    the form a[i, i+offset]. If a has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-D sub-array whose diagonal is returned. The shape of the
    resulting array can be determined by removing axis1 and axis2 and appending an index to the
    right equal to the size of the resulting diagonals.

    Parameters
    ----------
    a : _Symbol
        Input data from which diagonal are taken.
    offset: int, Optional
        Offset of the diagonal from the main diagonal
    axis1: int, Optional
        Axis to be used as the first axis of the 2-D sub-arrays
    axis2: int, Optional
        Axis to be used as the second axis of the 2-D sub-arrays

    Returns
    -------
    out : _Symbol
        Output result

    Raises
    -------
    ValueError:  If the dimension of a is less than 2.
    """
    return _npi.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


# pylint:disable=redefined-outer-name, too-many-arguments
@set_module('mxnet.symbol.numpy')
def sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None):
    r"""
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : _Symbol
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
    sum_along_axis : _Symbol
        An ndarray with the same shape as `a`, with the specified
        axis removed. If an output array is specified, a reference to
        `out` is returned.
    """
    if where is not None and where is not True:
        raise ValueError("only where=None or where=True cases are supported for now")
    return _npi.sum(a, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial, out=out)
# pylint:enable=redefined-outer-name, too-many-arguments


_set_np_symbol_class(_Symbol)
