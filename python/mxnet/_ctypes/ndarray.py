# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments
# pylint: disable=global-statement, unused-import
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import sys as _sys
import numpy as np

from ..base import _LIB
from ..base import c_array, py_str, c_str, mx_uint, _Null
from ..base import NDArrayHandle, OpHandle
from ..base import check_call
from ..ndarray_doc import _build_doc

_ndarray_cls = None

class NDArrayBase(object):
    """Base data structure for ndarray"""
    __slots__ = ["handle", "writable"]
    # pylint: disable= no-member
    def __init__(self, handle, writable=True):
        """initialize a new NDArray

        Parameters
        ----------
        handle : NDArrayHandle
            NDArray handle of C API
        """
        if handle is not None:
            assert isinstance(handle, NDArrayHandle)
        self.handle = handle
        self.writable = writable

    def __del__(self):
        check_call(_LIB.MXNDArrayFree(self.handle))

    def __reduce__(self):
        return (_ndarray_cls, (None,), self.__getstate__())


# pylint: disable=too-many-locals, invalid-name
def _make_ndarray_function(handle, name):
    """Create a NDArray function from the FunctionHandle."""
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    key_var_num_args = ctypes.c_char_p()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
        handle, ctypes.byref(real_name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(key_var_num_args),
        ctypes.byref(ret_type)))
    narg = int(num_args.value)
    arg_names = [py_str(arg_names[i]) for i in range(narg)]
    arg_types = [py_str(arg_types[i]) for i in range(narg)]
    func_name = name
    key_var_num_args = py_str(key_var_num_args.value)
    ret_type = py_str(ret_type.value) if ret_type.value is not None else ''
    doc_str = _build_doc(func_name,
                         py_str(desc.value),
                         arg_names,
                         arg_types,
                         [py_str(arg_descs[i]) for i in range(narg)],
                         key_var_num_args,
                         ret_type)

    dtype_name = None
    arr_name = None
    ndsignature = []
    signature = []
    ndarg_names = []
    kwarg_names = []
    for i in range(narg):
        name, atype = arg_names[i], arg_types[i]
        if name == 'dtype':
            dtype_name = name
            signature.append('%s=_Null'%name)
        elif atype.startswith('NDArray') or atype.startswith('Symbol'):
            assert not arr_name, \
                "Op can only have one argument with variable " \
                "size and it must be the last argument."
            if atype.endswith('[]'):
                ndsignature.append('*%s'%name)
                arr_name = name
            else:
                ndsignature.append('%s=None'%name)
                ndarg_names.append(name)
        else:
            signature.append('%s=_Null'%name)
            kwarg_names.append(name)
    #signature.append('is_train=False')
    signature.append('out=None')
    signature.append('**kwargs')
    signature = ndsignature + signature

    code = []
    if arr_name:
        code.append("""
def %s(*%s, **kwargs):"""%(func_name, arr_name))
        code.append("""
    ndargs = []
    for i in {}:
        assert isinstance(i, NDArrayBase), \\
            "Positional arguments must have NDArray type, " \\
            "but got %s"%str(type(i))
        ndargs.append(i.handle)""".format(arr_name))
        if dtype_name is not None:
            code.append("""
    if '%s' in kwargs:
        kwargs['%s'] = np.dtype(kwargs['%s']).name"""%(
            dtype_name, dtype_name, dtype_name))
        code.append("""
    out = kwargs.pop('out', None)
    keys = list(kwargs.keys())
    vals = [str(i) for i in kwargs.values()]""")
    else:
        code.append("""
def %s(%s):
    ndargs = []
    keys = list(kwargs.keys())
    vals = [str(i) for i in kwargs.values()]"""%(func_name, ', '.join(signature)))
        # NDArray args
        for name in ndarg_names:
            code.append("""
    if {name} is not None:
        assert isinstance({name}, NDArrayBase), \\
            "Argument {name} must have NDArray type, but got %s"%str(type({name}))
        ndargs.append({name}.handle)""".format(name=name))
        # kwargs
        for name in kwarg_names:
            code.append("""
    if %s is not _Null:
        keys.append('%s')
        vals.append(str(%s))"""%(name, name, name))
        # dtype
        if dtype_name is not None:
            code.append("""
    if %s is not _Null:
        keys.append('%s')
        vals.append(np.dtype(%s).name)"""%(dtype_name, dtype_name, dtype_name))

    # output
    code.append("""
    global handle
    if out is not None:
        original_output = out
        if isinstance(out, NDArrayBase):
            out = (out,)
        num_output = ctypes.c_int(len(out))
        output_vars = c_array(NDArrayHandle, [i.handle for i in out])
        output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
    else:
        original_output = None
        output_vars = ctypes.POINTER(NDArrayHandle)()
        num_output = ctypes.c_int(0)

    check_call(_LIB.MXImperativeInvoke(
        ctypes.c_void_p(%d),
        ctypes.c_int(len(ndargs)),
        c_array(NDArrayHandle, ndargs),
        ctypes.byref(num_output),
        ctypes.byref(output_vars),
        ctypes.c_int(len(keys)),
        c_array(ctypes.c_char_p, [c_str(key) for key in keys]),
        c_array(ctypes.c_char_p, [c_str(val) for val in vals])))
    if original_output is not None:
        return original_output
    if num_output.value == 1:
        return _ndarray_cls(ctypes.cast(output_vars[0], NDArrayHandle))
    else:
        return [_ndarray_cls(ctypes.cast(output_vars[i], NDArrayHandle))
                for i in range(num_output.value)]
"""%handle.value)

    local = {}
    exec(''.join(code), None, local)  # pylint: disable=exec-used
    ndarray_function = local[func_name]
    ndarray_function.__name__ = func_name
    ndarray_function.__doc__ = doc_str
    ndarray_function.__module__ = 'mxnet.ndarray'
    return ndarray_function


def _set_ndarray_class(cls):
    """Set the symbolic class to be cls"""
    global _ndarray_cls
    _ndarray_cls = cls


# pylint: enable=too-many-locals, invalid-name
def _init_ndarray_module(ndarray_class, root_namespace):
    """List and add all the ndarray functions to current module."""
    _set_ndarray_class(ndarray_class)
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_names.append(py_str(plist[i]))

    module_obj = _sys.modules["%s.ndarray" % root_namespace]
    module_internal = _sys.modules["%s._ndarray_internal" % root_namespace]
    module_contrib = _sys.modules["%s.contrib.ndarray" % root_namespace]
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        function = _make_ndarray_function(hdl, name)
        if function.__name__.startswith('_contrib_'):
            function.__name__ = function.__name__[9:]
            function.__module__ = 'mxnet.contrib.ndarray'
            setattr(module_contrib, function.__name__, function)
        elif function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
        else:
            setattr(module_obj, function.__name__, function)
