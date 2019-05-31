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
from array import array as native_array
import ctypes
import numpy as _np
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _GRAD_REQ_MAP
from ..ndarray._internal import _set_np_ndarray_class
from . import _op as _mx_np_op
from ..base import check_call, _LIB, NDArrayHandle
from ..base import mx_real_t, c_array_buf, mx_uint, numeric_types
from ..util import _sanity_check_params, set_module, use_np_shape
from ..context import current_context
from ..ndarray import numpy as _mx_nd_np
from ..ndarray.numpy import _internal as _npi

__all__ = ['ndarray', 'empty', 'array', 'zeros', 'ones', 'maximum', 'minimum', 'stack']


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


@set_module('mxnet.numpy')  # pylint: disable=invalid-name
@use_np_shape
class ndarray(NDArray):
    """An array object represents a multidimensional, homogeneous array of fixed-size items.
    An associated data-type object describes the format of each element in the array
    (its byte-order, how many bytes it occupies in memory, whether it is an integer, a
    floating point number, or something else, etc.). Arrays should be constructed using
    `array`, `zeros` or `empty`. Currently, only c-contiguous arrays are supported."""

    def __getitem__(self, item):
        # TODO(junwu): make output shape of integer indexing correct
        raise NotImplementedError

    def __setitem__(self, key, value):
        if self.size == 0:
            return
        if self.ndim == 0:
            if key != ():
                raise IndexError('scalar tensor can only accept `()` as index')
            # TODO(junwu): Better handling of this situation
            hdl = NDArrayHandle()
            check_call(_LIB.MXShallowCopyNDArray(self.handle, ctypes.byref(hdl)))
            classic_ndarray = NDArray(handle=hdl, writable=self.writable)
            classic_ndarray.__setitem__(slice(None), value)
            return
        self._as_classic_ndarray().__setitem__(key, value)

    def __add__(self, other):
        """x.__add__(y) <=> x + y"""
        if isinstance(other, ndarray):
            return _npi.add(self, other)
        elif isinstance(other, numeric_types):
            return _npi.add_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __iadd__(self, other):
        """x.__iadd__(y) <=> x += y"""
        if not self.writable:
            raise ValueError('trying to add to a readonly ndarray')
        if isinstance(other, ndarray):
            return _npi.add(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _npi.add_scalar(self, float(other), out=self)
        else:
            raise TypeError('type {} is not supported'.format(str(type(other))))

    def __sub__(self, other):
        """x.__sub__(y) <=> x - y"""
        if isinstance(other, ndarray):
            return _npi.subtract(self, other)
        elif isinstance(other, numeric_types):
            return _npi.subtract_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __isub__(self, other):
        """x.__isub__(y) <=> x -= y"""
        if not self.writable:
            raise ValueError('trying to subtract from a readonly ndarray')
        if isinstance(other, ndarray):
            return _npi.subtract(self, other, out=self)
        elif isinstance(other, numeric_types):
            return _npi.subtract_scalar(self, float(other), out=self)
        else:
            raise TypeError('type {} is not supported'.format(str(type(other))))

    def __rsub__(self, other):
        """x.__rsub__(y) <=> y - x"""
        if isinstance(other, ndarray):
            return _npi.subtract(other, self)
        elif isinstance(other, numeric_types):
            return _npi.rsubtract_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __mul__(self, other):
        """x.__mul__(y) <=> x * y"""
        if isinstance(other, ndarray):
            return _npi.multiply(self, other)
        elif isinstance(other, numeric_types):
            return _npi.multiply_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __neg__(self):
        return self.__mul__(-1.0)

    def __imul__(self, other):
        raise NotImplementedError

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
        raise NotImplementedError

    def __truediv__(self, other):
        """x.__truediv__(y) <=> x / y"""
        if isinstance(other, ndarray):
            return _npi.true_divide(self, other)
        elif isinstance(other, numeric_types):
            return _npi.true_divide_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as divisor".format(str(type(other))))

    def __rtruediv__(self, other):
        """x.__rtruediv__(y) <=> y / x"""
        if isinstance(other, ndarray):
            return _npi.true_divide(other, self)
        elif isinstance(other, numeric_types):
            return _npi.rtrue_divide_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as dividend".format(str(type(other))))

    def __itruediv__(self, other):
        raise NotImplementedError

    def __mod__(self, other):
        """x.__mod__(y) <=> x % y"""
        if isinstance(other, ndarray):
            return _npi.mod(self, other)
        elif isinstance(other, numeric_types):
            return _npi.mod_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __rmod__(self, other):
        """x.__rmod__(y) <=> y % x"""
        if isinstance(other, ndarray):
            return _npi.mod(other, self)
        elif isinstance(other, numeric_types):
            return _npi.rmod_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __imod__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        """x.__pow__(y) <=> x ** y"""
        if isinstance(other, ndarray):
            return _npi.power(self, other)
        elif isinstance(other, numeric_types):
            return _npi.power_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __rpow__(self, other):
        """x.__rpow__(y) <=> y ** x"""
        if isinstance(other, ndarray):
            return _npi.power(other, self)
        elif isinstance(other, numeric_types):
            return _npi.rpower_scalar(self, float(other))
        else:
            raise TypeError("ndarray does not support type {} as operand".format(str(type(other))))

    def __eq__(self, other):
        """x.__eq__(y) <=> x == y"""
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def __ne__(self, other):
        """x.__ne__(y) <=> x != y"""
        raise NotImplementedError

    def __gt__(self, other):
        """x.__gt__(y) <=> x > y"""
        raise NotImplementedError

    def __ge__(self, other):
        """x.__ge__(y) <=> x >= y"""
        raise NotImplementedError

    def __lt__(self, other):
        """x.__lt__(y) <=> x < y"""
        raise NotImplementedError

    def __le__(self, other):
        """x.__le__(y) <=> x <= y"""
        raise NotImplementedError

    def __bool__(self):
        raise NotImplementedError

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

    def _slice(self, start, stop):
        raise NotImplementedError

    def _at(self, idx):
        raise NotImplementedError

    def all(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def any(self, axis=None, out=None, keepdims=False):
        raise NotImplementedError

    def _as_classic_ndarray(self):
        """This is not a user-facing API."""
        hdl = NDArrayHandle()
        check_call(_LIB.MXShallowCopyNDArray(self.handle, ctypes.byref(hdl)))
        return NDArray(handle=hdl, writable=self.writable)

    def as_classic_ndarray(self):
        """Convert mxnet.numpy.ndarray to mxnet.ndarray.NDArray to use its fluent methods."""
        if self.ndim == 0:  # TODO(junwu): this costs ~10ns, can be moved to backend
            raise ValueError('cannot convert a scalar np.ndarray to mx.nd.NDArray')
        if self.size == 0:  # TODO(junwu): this costs ~10ns, can be moved to backend
            raise ValueError('cannot convert a zero-size np.ndarray to mx.nd.NDArray')
        return self._as_classic_ndarray()

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
            other = other._as_classic_ndarray()
        return self._as_classic_ndarray().copyto(other).as_np_ndarray()

    def asscalar(self):
        raise AttributeError('mxnet.numpy.ndarray object has no attribute as_scalar')

    def as_in_context(self, context):
        return super(ndarray, self).as_in_context(context).as_np_ndarray()

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

    def repeat(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`repeat`.

        The arguments are the same as for :py:func:`repeat`, with
        this array as data.
        """
        raise NotImplementedError

    def pad(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pad`.

        The arguments are the same as for :py:func:`pad`, with
        this array as data.
        """
        raise AttributeError('mxnet.numpy.ndarray object has no attribute pad')

    def swapaxes(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`swapaxes`.

        The arguments are the same as for :py:func:`swapaxes`, with
        this array as data.
        """
        raise NotImplementedError

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

    def argmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax`.

        The arguments are the same as for :py:func:`argmax`, with
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

    def clip(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`clip`.

        The arguments are the same as for :py:func:`clip`, with
        this array as data.
        """
        raise NotImplementedError

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

    def mean(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`mean`.

        The arguments are the same as for :py:func:`mean`, with
        this array as data.
        """
        raise NotImplementedError

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

    def squeeze(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`squeeze`.

        The arguments are the same as for :py:func:`squeeze`, with
        this array as data.
        """
        raise NotImplementedError

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
@use_np_shape
def array(object, dtype=None, **kwargs):
    """
    Create an array.

    Parameters
    ----------
    object : array_like or `mxnet.ndarray.NDArray` or `mxnet.numpy.ndarray`
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
    _sanity_check_params('array', ['copy', 'order', 'subok', 'ndim'], kwargs)
    ctx = kwargs.get('ctx', current_context())
    if ctx is None:
        ctx = current_context()
    if dtype is None:
        dtype = _np.float32
    if not isinstance(object, (ndarray, NDArray, _np.ndarray)):
        try:
            object = _np.array(object, dtype=dtype)
        except:
            raise TypeError('source array must be an array like object')
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
