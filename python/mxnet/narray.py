# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-locals, fixme, no-member
"""NArray interface of mxnet"""
from __future__ import absolute_import

import ctypes
import sys
from .base import _LIB
from .base import c_array
from .base import mx_uint, mx_float, NArrayHandle, FunctionHandle
from .base import ctypes2numpy_shared
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
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, NArray):
            return NArray._minus(self, other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __mul__(self, other):
        if isinstance(other, NArray):
            return NArray._mul(self, other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, NArray):
            return NArray._div(self, other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def wait(self):
        """Wait until the data on current NArray is available."""
        check_call(_LIB.MXNArrayWait(self.handle))

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
        the context of current NArray
        """
        dev_mask = ctypes.c_int()
        dev_id = ctypes.c_int()
        check_call(_LIB.MXNArrayGetContext(
            self.handle, ctypes.byref(dev_mask), ctypes.byref(dev_id)))
        return Context(Context.devmask2type[dev_mask.value], dev_id.value)

    @property
    def numpy(self):
        """Return a numpy representation of current array.

        This array have to sit on CPU

        Returns
        -------
        a numpy array view
        """
        self.wait()
        pdata = ctypes.POINTER(mx_float)()
        check_call(_LIB.MXNArrayGetData(self.handle, ctypes.byref(pdata)))
        return ctypes2numpy_shared(pdata, self.shape)

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
            return NArray._copyto(self, out=other)
        elif isinstance(other, Context):
            hret = NArray(_new_alloc_handle(self.shape, other, True))
            return NArray._copyto(self, out=hret)
        else:
            raise TypeError('copyto do not support type ' + type(other))


def create(shape, ctx=Context.default_ctx):
    """Create a new NArray, with specified shape.

    Parameters
    ----------
    shape : tuple
        shape of the NArray

    Returns
    -------
    a new NArray
    """
    return NArray(handle=_new_alloc_handle(shape, ctx, False))


def _make_narray_function(handle):
    """Create a NArray function from the FunctionHandle."""
    # Constants for type masks.
    NARRAY_ARG_BEFORE_SCALAR = 1
    ACCEPT_EMPTY_MUTATE_TARGET = 1 << 2
    # Get the property of NArray
    n_mutate_vars = 0
    n_used_vars = mx_uint()
    n_scalars = mx_uint()
    n_mutate_vars = mx_uint()
    type_mask = ctypes.c_int()
    check_call(_LIB.MXFuncDescribe( \
            handle, \
            ctypes.byref(n_used_vars), \
            ctypes.byref(n_scalars), \
            ctypes.byref(n_mutate_vars), \
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

    check_call(_LIB.MXFuncGetInfo( \
            handle, ctypes.byref(name), ctypes.byref(desc), \
            ctypes.byref(num_args), \
            ctypes.byref(arg_names), \
            ctypes.byref(arg_types), \
            ctypes.byref(arg_descs)))
    func_name = name.value

    param_str = []
    for i in range(num_args.value):
        ret = '%s : %s' % (arg_names[i], arg_types[i])
        if len(arg_descs[i]) != 0:
            ret += '\n    ' + arg_descs[i]
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
    doc_str = doc_str % (desc.value, '\n'.join(param_str))

    # Definition of internal functions.
    def binary_narray_function(lhs, rhs, out=None):
        """Internal binary function
        """
        if out:
            if isinstance(out, NArray):
                raise TypeError('out must be NArray')
        else:
            if not accept_empty_mutate:
                raise TypeError('argument out is required to call %s' % func_name)
            out = NArray(_new_empty_handle())
        check_call(_LIB.MXFuncInvoke( \
                handle, \
                c_array(NArrayHandle, (lhs.handle, rhs.handle)), \
                c_array(mx_float, ()), \
                c_array(NArrayHandle, (out.handle,))))
        return out

    def unary_narray_function(src, out=None):
        """internal NArray function"""
        if out:
            if isinstance(out, NArray):
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
