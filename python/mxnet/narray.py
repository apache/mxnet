# coding: utf-8
"""NArray interface of mxnet"""
from __future__ import absolute_import

import ctypes
import warnings
import sys
import numpy as np
from .base import _LIB, string_types, numeric_types
from .base import c_array, py_str, c_str
from .base import mx_uint, mx_float, NArrayHandle, FunctionHandle
from .base import ctypes2buffer
from .base import check_call
from .context import Context

def _new_empty_handle():
    """Return a new empty handle.

    Empty handle can be used to hold result

    Returns
    -------
    a new empty narray handle
    """
    hdl = NArrayHandle()
    check_call(_LIB.MXNArrayCreateNone(ctypes.byref(hdl)))
    return hdl

def _new_alloc_handle(shape, ctx, delay_alloc):
    """Return a new handle with specified shape and context.

    Empty handle is only used to hold results

    Returns
    -------
    a new empty narray handle
    """
    hdl = NArrayHandle()
    check_call(_LIB.MXNArrayCreate(
        c_array(mx_uint, shape),
        len(shape),
        ctx.device_mask,
        ctx.device_id,
        int(delay_alloc),
        ctypes.byref(hdl)))
    return hdl

class NArray(object):
    """NArray object in mxnet.

    NArray is basic ndarray/Tensor like data structure in mxnet.
    """
    # pylint: disable= no-member
    def __init__(self, handle):
        """initialize a new NArray

        Parameters
        ----------
        handle : NArrayHandle
            NArray handle of C API
        """
        assert isinstance(handle, NArrayHandle)
        self.handle = handle

    def __del__(self):
        check_call(_LIB.MXNArrayFree(self.handle))

    def __add__(self, other):
        if isinstance(other, NArray):
            return NArray._plus(self, other)
        elif isinstance(other, numeric_types):
            return NArray._plus_scalar(self, float(other))
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __iadd__(self, other):
        if isinstance(other, NArray):
            return NArray._plus(self, other, out=self)
        elif isinstance(other, numeric_types):
            return NArray._plus_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, NArray):
            return NArray._minus(self, other)
        elif isinstance(other, numeric_types):
            return NArray._minus_scalar(self, float(other))
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __isub__(self, other):
        if isinstance(other, NArray):
            return NArray._minus(self, other, out=self)
        elif isinstance(other, numeric_types):
            return NArray._minus_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rsub__(self, other):
        if isinstance(other, numeric_types):
            return NArray._rminus_scalar(self, float(other))
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __mul__(self, other):
        if isinstance(other, NArray):
            return NArray._mul(self, other)
        elif isinstance(other, numeric_types):
            return NArray._mul_scalar(self, float(other))
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __neg__(self):
        return NArray._mul_scalar(self, -1.0, out=self)

    def __imul__(self, other):
        if isinstance(other, NArray):
            return NArray._mul(self, other, out=self)
        elif isinstance(other, numeric_types):
            return NArray._mul_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, NArray):
            return NArray._div(self, other)
        elif isinstance(other, numeric_types):
            return NArray._div_scalar(self, float(other))
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rdiv__(self, other):
        if isinstance(other, numeric_types):
            return NArray._rdiv_scalar(self, float(other))
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __idiv__(self, other):
        if isinstance(other, NArray):
            return NArray._div(self, other, out=self)
        elif isinstance(other, numeric_types):
            return NArray._div_scalar(self, float(other), out=self)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __truediv__(self, other):
        return self.__div__(other)

    def __getstate__(self):
        this = self.__dict__.copy()
        handle = this['handle']
        if handle is not None:
            length = ctypes.c_ulong()
            cptr = ctypes.POINTER(ctypes.c_char)()
            check_call(_LIB.MXNArraySaveRawBytes(self.handle,
                                                 ctypes.byref(length),
                                                 ctypes.byref(cptr)))
            this['handle'] = ctypes2buffer(cptr, length.value)
        return this

    def __setstate__(self, state):
        handle = state['handle']
        if handle is not None:
            buf = handle
            handle = NArrayHandle()
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            length = ctypes.c_ulong(len(buf))
            check_call(_LIB.MXNArrayLoadFromRawBytes(ptr, length, ctypes.byref(handle)))
            state['handle'] = handle
        self.__dict__.update(state)

    def __setitem__(self, in_slice, value):
        """Set narray value"""
        if in_slice.step != None:
            raise Exception("Set NArray should use empty index array[:] = target_array")
        if isinstance(value, NArray):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, numeric_types):
            NArray._set_value(float(value), out=self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self._sync_copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def __getitem__(self, in_slice):
        """Get narray"""
        if in_slice.step != None:
            raise Exception("Set NArray should use empty index array[:] += value")
        return self

    def _sync_copyfrom(self, source_array):
        """Peform an synchronize copy from the array.

        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=np.float32)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported' % str(type(array)))
        source_array = np.ascontiguousarray(source_array, dtype=np.float32)

        if source_array.shape != self.shape:
            raise ValueError('array shape do not match the shape of NArray')

        check_call(_LIB.MXNArraySyncCopyFromCPU(
            self.handle,
            source_array.ctypes.data_as(ctypes.POINTER(mx_float)),
            source_array.size))

    def wait_to_read(self):
        """Block until all pending writes operations on current NArray are finished.

        This function will return when all the pending writes to the current
        NArray finishes. There can still be pending read going on when the
        function returns.
        """
        check_call(_LIB.MXNArrayWaitToRead(self.handle))

    def wait_to_write(self):
        """Block until all pending read/write operations on current NArray are finished.

        This function will return when all the pending writes to the current
        NArray finishes. There can still be pending read going on when the
        function returns.
        """
        check_call(_LIB.MXNArrayWaitToWrite(self.handle))

    @property
    def shape(self):
        """Get shape of current NArray.

        Returns
        -------
        a tuple representing shape of current narray
        """
        ndim = mx_uint()
        pdata = ctypes.POINTER(mx_uint)()
        check_call(_LIB.MXNArrayGetShape(
            self.handle, ctypes.byref(ndim), ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])

    @property
    def context(self):
        """Get context of current NArray.

        Returns
        -------
        context : mxnet.Context
            The context of current NArray.
        """
        dev_mask = ctypes.c_int()
        dev_id = ctypes.c_int()
        check_call(_LIB.MXNArrayGetContext(
            self.handle, ctypes.byref(dev_mask), ctypes.byref(dev_id)))
        return Context(Context.devmask2type[dev_mask.value], dev_id.value)

    def asnumpy(self):
        """Return a copied numpy array of current array.

        Returns
        -------
        array : numpy.ndarray
            A copy of array content.
        """
        data = np.empty(self.shape, dtype=np.float32)
        check_call(_LIB.MXNArraySyncCopyToCPU(
            self.handle,
            data.ctypes.data,
            data.size))
        return data

    def copyto(self, other):
        """Copy the content of current array to other.

        When other is NArray, the content is copied over.
        When other is a Context, a new NArray in the context
        will be created as target

        Parameters
        ----------
        other : NArray or Context
            Target Narray or context we want to copy data to.

        Returns
        -------
        dst : NArray
            The copy target NArray
        """
        if isinstance(other, NArray):
            if other.handle is self.handle:
                warnings.warn('copy an array to itself, is it intended?',
                              RuntimeWarning)
                return
            return NArray._copyto(self, out=other)
        elif isinstance(other, Context):
            hret = NArray(_new_alloc_handle(self.shape, other, True))
            return NArray._copyto(self, out=hret)
        else:
            raise TypeError('copyto do not support type ' + type(other))
    # pylint: enable= no-member


def empty(shape, ctx=None):
    """Create an empty uninitialized new NArray, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NArray.

    ctx : Context, optional
        The context of the NArray, default to current default context.

    Returns
    -------
    out: Array
        The created NArray.
    """
    if ctx is None:
        ctx = Context.default_ctx
    return NArray(handle=_new_alloc_handle(shape, ctx, False))

def zeros(shape, ctx=None):
    """Create a new NArray filled with 0, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NArray.

    ctx : Context, optional
        The context of the NArray, default to current default context.

    Returns
    -------
    out: Array
        The created NArray.
    """
    arr = empty(shape, ctx)
    arr[:] = 0.0
    return arr

def ones(shape, ctx=None):
    """Create a new NArray filled with 1, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NArray.

    ctx : Context, optional
        The context of the NArray, default to current default context.

    Returns
    -------
    out: Array
        The created NArray.
    """
    arr = empty(shape, ctx)
    arr[:] = 1.0
    return arr


def array(source_array, ctx=None):
    """Create a new NArray that copies content from source_array.

    Parameters
    ----------
    source_array : array_like
        Source data to create NArray from.

    ctx : Context, optional
        The context of the NArray, default to current default context.

    Returns
    -------
    out: Array
        The created NArray.
    """

    if not isinstance(source_array, np.ndarray):
        try:
            source_array = np.array(source_array, dtype=np.float32)
        except:
            raise TypeError('source_array must be array like object')
    arr = empty(source_array.shape, ctx)
    arr[:] = source_array
    return arr


def load(fname):
    """Load narray from binary file.

    You can also use pickle to do the job if you only work on python.
    The advantage of load/save is the file is language agnostic.
    This means the file saved using save can be loaded by other language binding of mxnet.

    Parameters
    ----------
    fname : str
        The name of the file

    Returns
    -------
    out : list of NArray or dict of str to NArray
        List of NArray or dict of str->NArray, depending on what was saved.
    """
    if not isinstance(fname, string_types):
        raise TypeError('fname need to be string')
    out_size = mx_uint()
    out_name_size = mx_uint()
    handles = ctypes.POINTER(NArrayHandle)()
    names = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXNArrayListLoad(c_str(fname),
                                     ctypes.byref(out_size),
                                     ctypes.byref(handles),
                                     ctypes.byref(out_name_size),
                                     ctypes.byref(names)))
    if out_name_size.value == 0:
        return [NArray(NArrayHandle(handles[i])) for i in range(out_size.value)]
    else:
        assert out_name_size.value == out_size.value
        return dict(
            (py_str(names[i]), NArray(NArrayHandle(handles[i]))) for i in range(out_size.value))


def save(fname, data):
    """Save list of NArray or dict of str->NArray to binary file.

    You can also use pickle to do the job if you only work on python.
    The advantage of load/save is the file is language agnostic.
    This means the file saved using save can be loaded by other language binding of mxnet.

    Parameters
    ----------
    fname : str
        The name of the file

    data : list of NArray or dict of str to NArray
        The data to be saved.
    """
    handles = []
    if isinstance(data, dict):
        keys = []
        for key, val in data.items():
            if not isinstance(key, string_types):
                raise TypeError('save only accept dict str->NArray or list of NArray')
            if not isinstance(val, NArray):
                raise TypeError('save only accept dict str->NArray or list of NArray')
            keys.append(c_str(key))
            handles.append(val.handle)
        keys = c_array(ctypes.c_char_p, keys)
    else:
        for val in data:
            if not isinstance(val, NArray):
                raise TypeError('save only accept dict str->NArray or list of NArray')
            handles.append(val.handle)
        keys = None
    check_call(_LIB.MXNArrayListSave(c_str(fname),
                                     len(handles),
                                     c_array(NArrayHandle, handles),
                                     keys))


# pylint: disable=too-many-locals, invalid-name
def _make_narray_function(handle):
    """Create a NArray function from the FunctionHandle."""
    NARRAY_ARG_BEFORE_SCALAR = 1
    ACCEPT_EMPTY_MUTATE_TARGET = 1 << 2
    # Get the property of NArray
    n_mutate_vars = 0
    n_used_vars = mx_uint()
    n_scalars = mx_uint()
    n_mutate_vars = mx_uint()
    type_mask = ctypes.c_int()
    check_call(_LIB.MXFuncDescribe(
        handle,
        ctypes.byref(n_used_vars),
        ctypes.byref(n_scalars),
        ctypes.byref(n_mutate_vars),
        ctypes.byref(type_mask)))
    n_mutate_vars = n_mutate_vars.value
    n_used_vars = n_used_vars.value
    n_scalars = n_scalars.value
    type_mask = type_mask.value
    accept_empty_mutate = (type_mask & ACCEPT_EMPTY_MUTATE_TARGET) != 0
    # infer type of the function
    if (type_mask & NARRAY_ARG_BEFORE_SCALAR) != 0:
        use_vars_range = range(0, n_used_vars)
        scalar_range = range(n_used_vars, n_used_vars + n_scalars)
    else:
        scalar_range = range(0, n_scalars)
        use_vars_range = range(n_scalars, n_used_vars + n_scalars)

    # Get the information from the function
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXFuncGetInfo(
        handle, ctypes.byref(name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs)))
    func_name = py_str(name.value)

    param_str = []
    for i in range(num_args.value):
        ret = '%s : %s' % (py_str(arg_names[i]), py_str(arg_types[i]))
        if len(arg_descs[i]) != 0:
            ret += '\n    ' + py_str(arg_descs[i])
        param_str.append(ret)

    doc_str = ('%s\n\n' +
               'Parameters\n' +
               '----------\n' +
               '%s\n' +
               'out : NArray, optional\n' +
               '    The output NArray to hold the result.\n\n'+
               'Returns\n' +
               '-------\n' +
               'out : NArray\n'+
               '    The output of binary function.')
    doc_str = doc_str % (py_str(desc.value), '\n'.join(param_str))

    # Definition of internal functions.
    def binary_narray_function(lhs, rhs, out=None):
        """Internal binary function
        """
        if out:
            if isinstance(out, NArray) == False:
                raise TypeError('out must be NArray')
        else:
            if not accept_empty_mutate:
                raise TypeError('argument out is required to call %s' % func_name)
            out = NArray(_new_empty_handle())
        check_call(_LIB.MXFuncInvoke(handle,
                                     c_array(NArrayHandle, (lhs.handle, rhs.handle)),
                                     c_array(mx_float, ()),
                                     c_array(NArrayHandle, (out.handle,))))
        return out

    def unary_narray_function(src, out=None):
        """internal NArray function"""
        if out:
            if isinstance(out, NArray) == False:
                raise TypeError('out must be NArray')
        else:
            if not accept_empty_mutate:
                raise TypeError('argument out is required to call %s' % func_name)
            out = NArray(_new_empty_handle())
        check_call(_LIB.MXFuncInvoke( \
                handle, \
                c_array(NArrayHandle, (src.handle)), \
                c_array(mx_float, ()), \
                c_array(NArrayHandle, (out.handle,))))
        return out

    def generic_narray_function(*args, **kwargs):
        """Invoke this function by passing in parameters

        Parameters
        ----------
        *args
            Positional arguments of input scalars and NArray
        out : NArray or tuple of NArray, optional
            Output NArray, used to hold the output result.

        Returns
        -------
        out : NArray
            The result NArray(tuple) of result of computation.
        """
        if 'out' in kwargs:
            mutate_vars = kwargs['out']
            if isinstance(mutate_vars, NArray):
                mutate_vars = (mutate_vars,)
            if len(mutate_vars) != n_mutate_vars:
                raise TypeError('expect %d out in %s', n_mutate_vars, func_name)
        else:
            if accept_empty_mutate:
                mutate_vars = tuple(
                    NArray(_new_empty_handle()) for i in range(n_mutate_vars))
            else:
                raise TypeError('argument out is required to call %s' % func_name)
        check_call(_LIB.MXFuncInvoke( \
                handle, \
                c_array(NArrayHandle, [args[i].handle for i in use_vars_range]), \
                c_array(mx_float, [args[i] for i in scalar_range]), \
                c_array(NArrayHandle, [v.handle for v in mutate_vars])))
        if n_mutate_vars == 1:
            return mutate_vars[0]
        else:
            return mutate_vars
    # End of function declaration
    if n_mutate_vars == 1 and n_used_vars == 2 and n_scalars == 0:
        ret_function = binary_narray_function
    elif n_mutate_vars == 1 and n_used_vars == 2 and n_scalars == 0:
        ret_function = unary_narray_function
    else:
        ret_function = generic_narray_function
    ret_function.__name__ = func_name
    ret_function.__doc__ = doc_str
    return ret_function
# pylint: enable=too-many-locals, invalid-name

def _init_narray_module():
    """List and add all the narray functions to current module."""
    plist = ctypes.POINTER(FunctionHandle)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListFunctions(ctypes.byref(size),
                                    ctypes.byref(plist)))

    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = FunctionHandle(plist[i])
        function = _make_narray_function(hdl)
        # if function name starts with underscore, register as static method of NArray
        if function.__name__.startswith('_'):
            setattr(NArray, function.__name__, staticmethod(function))
        else:
            setattr(module_obj, function.__name__, function)

# Initialize the NArray module
_init_narray_module()
