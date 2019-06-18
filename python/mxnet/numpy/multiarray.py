#!/usr/bin/env python

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
"""numpy ndarray and util functions."""

from __future__ import absolute_import
from __future__ import division

try:
    from __builtin__ import slice as py_slice
except ImportError:
    from builtins import slice as py_slice

from array import array as native_array
import sys
import ctypes
import warnings
import numpy as _np
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _GRAD_REQ_MAP
from ..ndarray._internal import _set_np_ndarray_class
from . import _op as _mx_np_op
from ..base import check_call, _LIB, NDArrayHandle
from ..base import mx_real_t, c_array_buf, mx_uint, numeric_types, integer_types
from ..util import _sanity_check_params, set_module
from ..context import current_context
from ..ndarray import numpy as _mx_nd_np
from ..ndarray.numpy import _internal as _npi

__all__ = ['ndarray', 'empty', 'array', 'zeros', 'ones', 'maximum', 'minimum', 'stack', 'arange',
           'argmax', 'add', 'subtract', 'multiply', 'divide', 'mod', 'power', 'concatenate',
           'clip', 'split', 'swapaxes', 'expand_dims', 'tile']


# This function is copied from ndarray.py since pylint
# keeps giving false alarm error of undefined-all-variable
def _new_alloc_handle(shape, ctx, delay_alloc, dtype=mx_real_t):
    """Return a new handle with specified shape and context.

    Empty handle is only used to hold results.

    Returns
    -------
    handle
        A new empty `ndarray` handle.
    """
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayCreateEx(
        c_array_buf(mx_uint, native_array('I', shape)),
        mx_uint(len(shape)),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(delay_alloc)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[_np.dtype(dtype).type])),
        ctypes.byref(hdl)))
    return hdl


# Have to use 0 as default value for stype since plylint does not allow
# importing _STORAGE_TYPE_DEFAULT from ndarray.py.
def _np_ndarray_cls(handle, writable=True, stype=0):
    if stype != 0:
        raise ValueError('_np_ndarray_cls currently only supports default storage '
                         'type, while received stype = {}'.format(stype))
    return ndarray(handle, writable=writable)


_set_np_ndarray_class(_np_ndarray_cls)


def _get_index(idx):
    if isinstance(idx, NDArray) and not isinstance(idx, ndarray):
        raise TypeError('Cannot have mx.nd.NDArray as index')
    if isinstance(idx, ndarray):
        return idx._as_nd_ndarray()
    elif sys.version_info[0] > 2 and isinstance(idx, range):
        return arange(idx.start, idx.stop, idx.step, dtype='int32')._as_nd_ndarray()
    else:
        return idx


@set_module('mxnet.numpy')  # pylint: disable=invalid-name
class ndarray(NDArray):
    """An array object represents a multidimensional, homogeneous array of fixed-size items.
    An associated data-type object describes the format of each element in the array
    (its byte-order, how many bytes it occupies in memory, whether it is an integer, a
    floating point number, or something else, etc.). Arrays should be constructed using
    `array`, `zeros` or `empty`. Currently, only c-contiguous arrays are supported."""

    # pylint: disable=too-many-return-statements
    def __getitem__(self, key):
        # TODO(junwu): calling base class __getitem__ is a temp solution
        ndim = self.ndim
        shape = self.shape
        if ndim == 0:
            if key != ():
                raise IndexError('scalar tensor can only accept `()` as index')
        if isinstance(key, tuple) and len(key) == 0:
            return self
        elif isinstance(key, tuple) and len(key) == ndim\
                and all(isinstance(idx, integer_types) for idx in key):
            out = self
            for idx in key:
                out = out[idx]
            return out
        elif isinstance(key, integer_types):
            if key > shape[0] - 1:
                raise IndexError(
                    'index {} is out of bounds for axis 0 with size {}'.format(
                        key, shape[0]))
            return self._at(key)
        elif isinstance(key, py_slice):
            if key.step is not None and key.step != 1:
                if key.step == 0:
                    raise ValueError("slice step cannot be zero")
                return self.as_nd_ndarray()._get_nd_basic_indexing(key).as_np_ndarray()
            elif key.start is not None or key.stop is not None:
                return self._slice(key.start, key.stop)
            else:
                return self

        if isinstance(key, ndarray):
            key = key._as_nd_ndarray()
        elif isinstance(key, tuple):
            key = [_get_index(idx) for idx in key]
            key = tuple(key)
        elif isinstance(key, list):
            key = [_get_index(idx) for idx in key]
        elif sys.version_info[0] > 2 and isinstance(key, range):
            key = _get_index(key)
        return self._as_nd_ndarray().__getitem__(key).as_np_ndarray()
    # pylint: enable=too-many-return-statements

    def __setitem__(self, key, value):
        # TODO(junwu): calling base class __setitem__ is a temp solution
        if isinstance(value, NDArray) and not isinstance(value, ndarray):
            raise TypeError('Cannot assign mx.nd.NDArray to mxnet.numpy.ndarray')
        if self.ndim == 0:
            if not isinstance(key, tuple) or len(key) != 0:
                raise IndexError('scalar tensor can only accept `()` as index')
        if isinstance(value, ndarray):
            value = value._as_nd_ndarray()
        # TODO(junwu): Better handling of this situation
        if isinstance(key, tuple) and len(key) == 0:
            self._as_nd_ndarray().__setitem__(slice(None), value)
            return

        if isinstance(key, ndarray):
            key = key._as_nd_ndarray()
        elif isinstance(key, tuple):
            key = [_get_index(idx) for idx in key]
            key = tuple(key)
        elif isinstance(key, list):
            key = [_get_index(idx) for idx in key]
        elif sys.version_info[0] > 2 and isinstance(key, range):
            key = _get_index(key)
        self._as_nd_ndarray().__setitem__(key, value)

    def __add__(self, other):
        """x.__add__(y) <=> x + y"""
        return add(self, other)

    def __iadd__(self, other):
        """x.__iadd__(y) <=> x += y"""
        if not self.writable:
            raise ValueError('trying to add to a readonly ndarray')
        return add(self, other, out=self)

    def __sub__(self, other):
        """x.__sub__(y) <=> x - y"""
        return subtract(self, other)

    def __isub__(self, other):
        """x.__isub__(y) <=> x -= y"""
        if not self.writable:
            raise ValueError('trying to subtract from a readonly ndarray')
        return subtract(self, other, out=self)

    def __rsub__(self, other):
        """x.__rsub__(y) <=> y - x"""
        return subtract(other, self)

    def __mul__(self, other):
        """x.__mul__(y) <=> x * y"""
        return multiply(self, other)

    def __neg__(self):
        return self.__mul__(-1.0)

    def __imul__(self, other):
        """x.__imul__(y) <=> x *= y"""
        if not self.writable:
            raise ValueError('trying to add to a readonly ndarray')
        return multiply(self, other, out=self)

    def __rmul__(self, other):
        """x.__rmul__(y) <=> y * x"""
        return self.__mul__(other)

    def __div__(self, other):
        raise AttributeError('ndarray.__div__ is replaced by __truediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __rdiv__(self, other):
        raise AttributeError('ndarray.__rdiv__ is replaced by __rtruediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __idiv__(self, other):
        raise AttributeError('ndarray.__idiv__ is replaced by __irtruediv__. If you are using'
                             ' Python2, please use the statement from __future__ import division'
                             ' to change the / operator to mean true division throughout the'
                             ' module. If you are using Python3, this error should not have'
                             ' been encountered.')

    def __truediv__(self, other):
        """x.__truediv__(y) <=> x / y"""
        return divide(self, other)

    def __rtruediv__(self, other):
        """x.__rtruediv__(y) <=> y / x"""
        return divide(other, self)

    def __itruediv__(self, other):
        return divide(self, other, out=self)

    def __mod__(self, other):
        """x.__mod__(y) <=> x % y"""
        return mod(self, other)

    def __rmod__(self, other):
        """x.__rmod__(y) <=> y % x"""
        return mod(other, self)

    def __imod__(self, other):
        """x.__imod__(y) <=> x %= y"""
        return mod(self, other, out=self)

    def __pow__(self, other):
        """x.__pow__(y) <=> x ** y"""
        return power(self, other)

    def __rpow__(self, other):
        """x.__rpow__(y) <=> y ** x"""
        return power(other, self)

    def __eq__(self, other):
        """x.__eq__(y) <=> x == y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.equal_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __hash__(self):
        raise NotImplementedError

    def __ne__(self, other):
        """x.__ne__(y) <=> x != y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.not_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.not_equal_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __gt__(self, other):
        """x.__gt__(y) <=> x > y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.greater(self, other)
        elif isinstance(other, numeric_types):
            return _npi.greater_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __ge__(self, other):
        """x.__ge__(y) <=> x >= y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.greater_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.greater_equal_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __lt__(self, other):
        """x.__lt__(y) <=> x < y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.less(self, other)
        elif isinstance(other, numeric_types):
            return _npi.less_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __le__(self, other):
        """x.__le__(y) <=> x <= y"""
        # TODO(junwu): Return boolean ndarray when dtype=bool_ is supported
        if isinstance(other, ndarray):
            return _npi.less_equal(self, other)
        elif isinstance(other, numeric_types):
            return _npi.less_equal_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __bool__(self):
        num_elements = self.size
        if num_elements == 0:
            warnings.simplefilter('default')
            warnings.warn('The truth value of an empty array is ambiguous. Returning False, but in'
                          ' future this will result in an error.', DeprecationWarning)
            return False
        elif num_elements == 1:
            return bool(self.item())
        else:
            raise ValueError("The truth value of an ndarray with multiple elements is ambiguous.")

    __nonzero__ = __bool__

    def __float__(self):
        num_elements = self.size
        if num_elements != 1:
            raise TypeError('only size-1 arrays can be converted to Python scalars')
        return float(self.item())

    def __int__(self):
        num_elements = self.size
        if num_elements != 1:
            raise TypeError('only size-1 arrays can be converted to Python scalars')
        return int(self.item())

    def __len__(self):
        """Number of elements along the first axis."""
        return self.shape[0]

    def __reduce__(self):
        return ndarray, (None,), self.__getstate__()

    def item(self, *args):
        """Copy an element of an array to a standard Python scalar and return it.

        Parameters
        ----------
        *args : Arguments (variable number and type)
            none: in this case, the method only works for arrays with one element (a.size == 1),
            which element is copied into a standard Python scalar object and returned.

            int_type: this argument is interpreted as a flat index into the array, specifying which
            element to copy and return.

            tuple of int_types: functions as does a single int_type argument, except that the
            argument is interpreted as an nd-index into the array.

        Returns
        -------
        z : Standard Python scalar object
            A copy of the specified element of the array as a suitable Python scalar.
        """
        # TODO(junwu): no need to call asnumpy() on the whole array.
        return self.asnumpy().item(*args)

    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        """Same as self.transpose(). This always returns a copy of self."""
        return self.transpose()
    # pylint: enable= invalid-name, undefined-variable

    def all(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def any(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def _as_nd_ndarray(self):
        """This is not a user-facing API."""
        hdl = NDArrayHandle()
        check_call(_LIB.MXShallowCopyNDArray(self.handle, ctypes.byref(hdl)))
        return NDArray(handle=hdl, writable=self.writable)

    def as_nd_ndarray(self):
        """Convert mxnet.numpy.ndarray to mxnet.ndarray.NDArray to use its fluent methods."""
        # TODO(junwu): Uncomment the following lines
        # if self.ndim == 0:  # TODO(junwu): this costs ~10ns, can be moved to backend
        #     raise ValueError('cannot convert a scalar np.ndarray to mx.nd.NDArray')
        # if self.size == 0:  # TODO(junwu): this costs ~10ns, can be moved to backend
        #     raise ValueError('cannot convert a zero-size np.ndarray to mx.nd.NDArray')
        return self._as_nd_ndarray()

    def as_np_ndarray(self):
        """A convenience function for creating a numpy ndarray from the current ndarray
        with zero copy. For this class, it just returns itself since it's already a
        numpy ndarray."""
        return self

    def __repr__(self):
        """Returns a string representation of the array using the following rules:
        1. If the `ndarray` is a scalar tensor, only the string of the scalar is returned.
        2. Else if the `ndarray` is allocated on cpu, the string of its numpy form, class name,
        and shape is returned.
        3. Else (the `ndarray` is allocated on gpu), the string of its numpy form, class name,
        shape, and context is returned."""
        array_str = str(self.asnumpy())
        if self.ndim == 0:  # scalar tensor
            return array_str
        context = self.context
        if context.device_type == 'gpu':
            return '%s\n<%s shape=%s ctx=%s>' % (array_str, self.__class__.__name__, self.shape,
                                                 context)
        else:
            return '%s\n<%s shape=%s>' % (array_str, self.__class__.__name__, self.shape)

    def attach_grad(self, grad_req='write'):  # pylint: disable=arguments-differ
        """Attach a gradient buffer to this ndarray, so that `backward`
        can compute gradient with respect to it.

        Parameters
        ----------
        grad_req : {'write', 'add', 'null'}
            How gradient will be accumulated.
            - 'write': gradient will be overwritten on every backward.
            - 'add': gradient will be added to existing value on every backward.
            - 'null': do not compute gradient for this NDArray.
        """
        grad = _mx_np_op.zeros_like(self)  # pylint: disable=undefined-variable
        grad_req = _GRAD_REQ_MAP[grad_req]
        check_call(_LIB.MXAutogradMarkVariables(
            1, ctypes.pointer(self.handle),
            ctypes.pointer(mx_uint(grad_req)),
            ctypes.pointer(grad.handle)))

    @property
    def grad(self):
        """Returns gradient buffer attached to this ndarray."""
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetGrad(self.handle, ctypes.byref(hdl)))
        if hdl.value is None:
            return None
        return _np_ndarray_cls(hdl)

    def detach(self):
        """Returns a new ndarray, detached from the current graph."""
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayDetach(self.handle, ctypes.byref(hdl)))
        return _np_ndarray_cls(hdl)

    def astype(self, dtype, *args, **kwargs):  # pylint: disable=arguments-differ,unused-argument
        """
        Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
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
        """
        _sanity_check_params('astype', ['order', 'casting', 'subok'], kwargs)
        copy = kwargs.get('copy', True)
        if not copy and _np.dtype(dtype) == self.dtype:
            return self

        res = empty(self.shape, dtype=dtype, ctx=self.context)
        self.copyto(res)
        return res

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``ndarray`` object, then ``other.shape`` and
        ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``NDArray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : ndarray or Context
            The destination array or context.

        Returns
        -------
        ndarray
            The copied array. If ``other`` is an ``ndarray``, then the return value
            and ``other`` will point to the same ``ndarray``.

        Examples
        --------
        >>> x = np.ones((2,3))
        >>> y = np.zeros((2,3), mx.gpu(0))
        >>> z = x.copyto(y)
        >>> z is y
        True
        >>> y.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        """
        if isinstance(other, ndarray):
            other = other._as_nd_ndarray()
        return self._as_nd_ndarray().copyto(other).as_np_ndarray()

    def asscalar(self):
        raise AttributeError('mxnet.numpy.ndarray object has no attribute asscalar')

    def argmax(self, axis=None, out=None):  # pylint: disable=arguments-differ
        return _mx_nd_np.argmax(self, axis, out)

    def as_in_context(self, context):
        """Returns an array on the target device with the same value as this array.

        If the target context is the same as ``self.context``, then ``self`` is
        returned.  Otherwise, a copy is made.

        Parameters
        ----------
        context : Context
            The target context.

        Returns
        -------
        ndarray
            The target array.
        """
        if self.context == context:
            return self
        return self.copyto(context)

    def copy(self, order='C'):  # pylint: disable=arguments-differ
        if order != 'C':
            raise NotImplementedError('ndarray.copy only supports order=\'C\', while '
                                      'received {}'.format(str(order)))
        return super(ndarray, self).copy().as_np_ndarray()

    def dot(self, b, out=None):
        return _mx_np_op.dot(self, b, out=out)

    def reshape(self, shape, order='C'):  # pylint: disable=arguments-differ
        """Returns an array containing the same data with a new shape."""
        if order != 'C':
            raise NotImplementedError('reshape only supports C-order,'
                                      ' while received {}'.format(order))
        return _mx_np_op.reshape(self, newshape=shape, order=order)

    def reshape_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reshape_like`.

        The arguments are the same as for :py:func:`reshape_like`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute reshape_like')

    def zeros_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`zeros_like`.

        The arguments are the same as for :py:func:`zeros_like`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute zeros_like')

    def ones_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ones_like`.

        The arguments are the same as for :py:func:`ones_like`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute ones_like')

    def broadcast_axes(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`broadcast_axes`.

        The arguments are the same as for :py:func:`broadcast_axes`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute broadcast_like')

    def repeat(self, repeats, axis=None):  # pylint: disable=arguments-differ
        """Repeat elements of an array."""
        return _mx_np_op.repeat(self, repeats=repeats, axis=axis)

    def pad(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pad`.

        The arguments are the same as for :py:func:`pad`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute pad')

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
        raise AttributeError('mxnet.numpy.ndarray object has no attribute split')

    def split_v2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split_v2`.

        The arguments are the same as for :py:func:`split_v2`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute split_v2')

    def slice(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice`.

        The arguments are the same as for :py:func:`slice`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute slice')

    def slice_axis(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_axis`.

        The arguments are the same as for :py:func:`slice_axis`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute slice_axis')

    def slice_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_like`.

        The arguments are the same as for :py:func:`slice_like`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute slice_like')

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
        raise AttributeError('mxnet.numpy.ndarray object has no attribute one_hot')

    def pick(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pick`.

        The arguments are the same as for :py:func:`pick`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute pick')

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
        raise AttributeError('mxnet.numpy.ndarray object has no attribute topk')

    def argsort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argsort`.

        The arguments are the same as for :py:func:`argsort`, with
        this array as data.
        """
        raise NotImplementedError

    def argmax_channel(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax_channel`.

        The arguments are the same as for :py:func:`argmax_channel`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute argmax_channel')

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
        raise AttributeError('mxnet.numpy.ndarray object has no attribute abs')

    def sign(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sign`.

        The arguments are the same as for :py:func:`sign`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute abs')

    def flatten(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`flatten`.

        The arguments are the same as for :py:func:`flatten`, with
        this array as data.
        """
        raise NotImplementedError

    def shape_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`shape_array`.

        The arguments are the same as for :py:func:`shape_array`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute shape_array')

    def size_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`size_array`.

        The arguments are the same as for :py:func:`size_array`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute size_array')

    def expand_dims(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expand_dims`.

        The arguments are the same as for :py:func:`expand_dims`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute expand_dims')

    def tile(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tile`.

        The arguments are the same as for :py:func:`tile`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute tile')

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
        raise AttributeError('mxnet.numpy.ndarray object has no attribute flip')

    def depth_to_space(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`depth_to_space`.

        The arguments are the same as for :py:func:`depth_to_space`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute depth_to_space')

    def space_to_depth(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`space_to_depth`.

        The arguments are the same as for :py:func:`space_to_depth`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute space_to_depth')

    def diag(self, k=0, **kwargs):
        """Convenience fluent method for :py:func:`diag`.

        The arguments are the same as for :py:func:`diag`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute diag')

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
        raise AttributeError('mxnet.numpy.ndarray object has no attribute nansum')

    def prod(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`prod`.

        The arguments are the same as for :py:func:`prod`, with
        this array as data.
        """
        raise NotImplementedError

    def nanprod(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nanprod`.

        The arguments are the same as for :py:func:`nanprod`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute nanprod')

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):  # pylint: disable=arguments-differ
        """Convenience fluent method for :py:func:`mean`.

        The arguments are the same as for :py:func:`mean`, with
        this array as data.
        """
        return _mx_nd_np.mean(self, axis=axis, dtype=dtype, keepdims=keepdims, out=out)

    def max(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`max`.

        The arguments are the same as for :py:func:`max`, with
        this array as data.
        """
        raise NotImplementedError

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
        raise AttributeError('mxnet.numpy.ndarray object has no attribute norm')

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
        raise AttributeError('mxnet.numpy.ndarray object has no attribute rint')

    def fix(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`fix`.

        The arguments are the same as for :py:func:`fix`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute fix')

    def floor(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`floor`.

        The arguments are the same as for :py:func:`floor`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute floor')

    def ceil(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ceil`.

        The arguments are the same as for :py:func:`ceil`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute ceil')

    def trunc(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`trunc`.

        The arguments are the same as for :py:func:`trunc`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute trunc')

    def sin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sin`.

        The arguments are the same as for :py:func:`sin`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute sin')

    def cos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cos`.

        The arguments are the same as for :py:func:`cos`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute cos')

    def tan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tan`.

        The arguments are the same as for :py:func:`tan`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute tan')

    def arcsin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsin`.

        The arguments are the same as for :py:func:`arcsin`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arcsin')

    def arccos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccos`.

        The arguments are the same as for :py:func:`arccos`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arccos')

    def arctan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctan`.

        The arguments are the same as for :py:func:`arctan`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arctan')

    def degrees(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`degrees`.

        The arguments are the same as for :py:func:`degrees`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute degrees')

    def radians(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`radians`.

        The arguments are the same as for :py:func:`radians`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute radians')

    def sinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sinh`.

        The arguments are the same as for :py:func:`sinh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute sinh')

    def cosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cosh`.

        The arguments are the same as for :py:func:`cosh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute cosh')

    def tanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tanh`.

        The arguments are the same as for :py:func:`tanh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute tanh')

    def arcsinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsinh`.

        The arguments are the same as for :py:func:`arcsinh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arcsinh')

    def arccosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccosh`.

        The arguments are the same as for :py:func:`arccosh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arccosh')

    def arctanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctanh`.

        The arguments are the same as for :py:func:`arctanh`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute arctanh')

    def exp(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`exp`.

        The arguments are the same as for :py:func:`exp`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute exp')

    def expm1(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expm1`.

        The arguments are the same as for :py:func:`expm1`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute expm1')

    def log(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log`.

        The arguments are the same as for :py:func:`log`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log')

    def log10(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log10`.

        The arguments are the same as for :py:func:`log10`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log10')

    def log2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log2`.

        The arguments are the same as for :py:func:`log2`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log2')

    def log1p(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log1p`.

        The arguments are the same as for :py:func:`log1p`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log1p')

    def sqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sqrt`.

        The arguments are the same as for :py:func:`sqrt`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute sqrt')

    def rsqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rsqrt`.

        The arguments are the same as for :py:func:`rsqrt`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute rsqrt')

    def cbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cbrt`.

        The arguments are the same as for :py:func:`cbrt`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute cqrt')

    def rcbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rcbrt`.

        The arguments are the same as for :py:func:`rcbrt`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute rcqrt')

    def square(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`square`.

        The arguments are the same as for :py:func:`square`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute square')

    def reciprocal(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reciprocal`.

        The arguments are the same as for :py:func:`reciprocal`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute reciprocal')

    def relu(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`relu`.

        The arguments are the same as for :py:func:`relu`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute relu')

    def sigmoid(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sigmoid`.

        The arguments are the same as for :py:func:`sigmoid`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute sigmoid')

    def softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmax`.

        The arguments are the same as for :py:func:`softmax`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute softmax')

    def log_softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log_softmax`.

        The arguments are the same as for :py:func:`log_softmax`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute log_softmax')

    def softmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmin`.

        The arguments are the same as for :py:func:`softmin`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute softmin')

    def squeeze(self, axis=None):  # pylint: disable=arguments-differ
        """Remove single-dimensional entries from the shape of a.
        """
        return _mx_np_op.squeeze(self, axis=axis)

    def broadcast_to(self, shape):
        raise AttributeError('mxnet.numpy.ndarray object has no attribute broadcast_to')

    def broadcast_like(self, other):
        raise AttributeError('mxnet.numpy.ndarray object has no attribute broadcast_like')

    @property
    def shape(self):
        return super(ndarray, self).shape

    @property
    def ndim(self):
        """Number of array dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Number of elements in the array."""
        return super(ndarray, self).size

    def tostype(self, stype):
        raise AttributeError('mxnet.numpy.ndarray object has no attribute tostype')


@set_module('mxnet.numpy')
def empty(shape, dtype=None, **kwargs):
    """Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Desired output data-type for the array, e.g, `numpy.int8`. Default is
        `numpy.float32`. Note that this behavior is different from NumPy's `empty`
        function where `float64` is the default value, because `float32` is
        considered as the default data type in deep learning.
    ctx : device context, optional
        Device context on which the memory is allocated. Default is
        `mxnet.context.current_context()`.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape, dtype, and order.
    """
    _sanity_check_params('emtpy', ['order'], kwargs)
    ctx = kwargs.get('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    if dtype is None:
        dtype = _np.float32
    if isinstance(shape, int):
        shape = (shape,)
    return ndarray(handle=_new_alloc_handle(shape, ctx, False, dtype))


@set_module('mxnet.numpy')
def array(object, dtype=None, ctx=None):
    """
    Create an array.

    Parameters
    ----------
    object : array_like or `numpy.ndarray` or `mxnet.numpy.ndarray`
        An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array. Default is `float32`.
    ctx : device context, optional
        Device context on which the memory is allocated. Default is
        `mxnet.context.current_context()`.

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.
    """
    if ctx is None:
        ctx = current_context()
    if isinstance(object, ndarray):
        dtype = object.dtype if dtype is None else dtype
    else:
        dtype = mx_real_t if dtype is None else dtype
        if not isinstance(object, (ndarray, _np.ndarray)):
            try:
                object = _np.array(object, dtype=dtype)
            except Exception as e:
                raise TypeError('{}'.format(str(e)))
    ret = empty(object.shape, dtype=dtype, ctx=ctx)
    if len(object.shape) == 0:
        ret[()] = object
    else:
        ret[:] = object
    return ret


@set_module('mxnet.numpy')
def zeros(shape, dtype=_np.float32, **kwargs):
    """Return a new array of given shape and type, filled with zeros.
    This function currently only supports storing multi-dimensional data
    in row-major (C-style).

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array.
    dtype : str or numpy.dtype, optional
        An optional value type (default is `numpy.float32`). Note that this
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
    return _mx_nd_np.zeros(shape, dtype, **kwargs)


@set_module('mxnet.numpy')
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
    return _mx_nd_np.ones(shape, dtype, **kwargs)


@set_module('mxnet.numpy')
def maximum(x1, x2, out=None):
    """Returns element-wise maximum of the input arrays with broadcasting.

    Parameters
    ----------
    x1, x2 : scalar or mxnet.numpy.ndarray
        The arrays holding the elements to be compared. They must have the same shape,
        or shapes that can be broadcast to a single shape.

    Returns
    -------
    out : mxnet.numpy.ndarray or scalar
        The maximum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars."""
    return _mx_nd_np.maximum(x1, x2, out=out)


@set_module('mxnet.numpy')
def minimum(x1, x2, out=None):
    """Returns element-wise minimum of the input arrays with broadcasting.

    Parameters
    ----------
    x1, x2 : scalar or mxnet.numpy.ndarray
        The arrays holding the elements to be compared. They must have the same shape,
        or shapes that can be broadcast to a single shape.

    Returns
    -------
    out : mxnet.numpy.ndarray or scalar
        The minimum of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars."""
    return _mx_nd_np.minimum(x1, x2, out=out)


@set_module('mxnet.numpy')
def stack(arrays, axis=0, out=None):
    """Join a sequence of arrays along a new axis.

        The axis parameter specifies the index of the new axis in the dimensions of the result.
        For example, if `axis=0` it will be the first dimension and if `axis=-1` it will be the last dimension.

    Parameters
    ----------
    arrays : sequence of array_like
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
    return _mx_nd_np.stack(arrays, axis=axis, out=out)


@set_module('mxnet.numpy')
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
    return _mx_nd_np.arange(start, stop, step, dtype, ctx)


@set_module('mxnet.numpy')
def argmax(a, axis=None, out=None):
    """Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : ndarray
        Input array. Only support ndarrays of dtype `float16`, `float32`, and `float64`.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of indices whose dtype is same as the input ndarray.
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.
    """
    return _mx_nd_np.argmax(a, axis, out)


@set_module('mxnet.numpy')
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
    return _mx_nd_np.concatenate(seq, axis=axis, out=out)


@set_module('mxnet.numpy')
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
    return _mx_nd_np.add(x1, x2, out)


@set_module('mxnet.numpy')
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
    return _mx_nd_np.subtract(x1, x2, out)


@set_module('mxnet.numpy')
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
        The difference of x1 and x2, element-wise. This is a scalar if both x1 and x2 are scalars.
    """
    return _mx_nd_np.multiply(x1, x2, out)


@set_module('mxnet.numpy')
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
    return _mx_nd_np.divide(x1, x2, out=out)


@set_module('mxnet.numpy')
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
    return _mx_nd_np.mod(x1, x2, out=out)


@set_module('mxnet.numpy')
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
    return _mx_nd_np.power(x1, x2, out=out)


@set_module('mxnet.numpy')
def clip(a, a_min, a_max, out=None):
    """Clip (limit) the values in an array.

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
        to hold the output.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.
    """
    return _mx_nd_np.clip(a, a_min, a_max, out=out)


@set_module('mxnet.numpy')
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


@set_module('mxnet.numpy')
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


@set_module('mxnet.numpy')
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
        a split does not result in equal division."""
    return _mx_nd_np.split(ary, indices_or_sections, axis=axis)


@set_module('mxnet.numpy')
def tile(A, reps):
    """
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

    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use numpy's broadcasting operations and functions.

    Parameters
    ----------
    A : ndarray
        The input array.
    reps : tuple of integers
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.
    """
    return _npi.tile(A, reps)
