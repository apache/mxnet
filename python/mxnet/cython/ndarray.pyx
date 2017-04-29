from __future__ import absolute_import as _abs

import sys as _sys
import ctypes as _ctypes
import numpy as np
from ..ndarray_doc import _build_doc

include "./base.pyi"

cdef class NDArrayBase:
    """Symbol is symbolic graph."""
    # handle for symbolic operator.
    cdef NDArrayHandle chandle
    cdef int cwritable

    cdef _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = handle.value
            self.chandle = <SymbolHandle>(ptr)

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return _ctypes.cast(<unsigned long long>self.chandle, _ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)

    property writable:
        def __get__(self):
            return bool(self.cwritable)

    def __init__(self, handle, writable=True):
        self._set_handle(handle)
        self.cwritable = writable

    def __dealloc__(self):
        CALL(MXNDArrayFree(self.chandle))

    def __reduce__(self):
        return (_ndarray_cls, (None,), self.__getstate__())


_ndarray_cls = NDArrayBase

cdef _set_ndarray_class(cls):
    global _ndarray_cls
    _ndarray_cls = cls


cdef NewArray(NDArrayHandle handle):
    """Create a new array given handle"""
    nd = _ndarray_cls(None)
    (<NDArrayBase>nd).chandle = handle
    (<NDArrayBase>nd).cwritable = True
    return nd

cdef _make_ndarray_function(OpHandle handle, string name):
    """Create a NDArray function from the FunctionHandle."""
    cdef const char *real_name
    cdef const char *desc
    cdef nn_uint num_args
    cdef const char** arg_names
    cdef const char** arg_types
    cdef const char** arg_descs
    cdef const char* return_type
    cdef const char* key_var_num_args

    CALL(MXSymbolGetAtomicSymbolInfo(
        handle, &real_name, &desc,
        &num_args, &arg_names,
        &arg_types, &arg_descs,
        &key_var_num_args, &return_type))
    func_name = py_str(name.c_str())

    key_vargs = py_str(key_var_num_args)
    num_args = int(num_args)
    doc_str = _build_doc(func_name,
                         py_str(desc),
                         [py_str(arg_names[i]) for i in range(num_args)],
                         [py_str(arg_types[i]) for i in range(num_args)],
                         [py_str(arg_descs[i]) for i in range(num_args)],
                         key_vargs,
                         py_str(return_type) if return_type != NULL else '')
    func_hint = func_name.lower()

    arguments = []
    for i in range(num_args):
        dtype = py_str(arg_types[i])
        if not (dtype.startswith('NDArray') or dtype.startswith('Symbol')):
            arguments.append(py_str(arg_names[i]))

    num_param_args = len(arguments)

    # Definition of internal functions.
    def generic_ndarray_function(*args, **kwargs):
        """Invoke this function by passing in parameters

        Parameters
        ----------
        *args
            Positional arguments of input scalars and NDArray
        out : NDArray or tuple of NDArray, optional
            Output NDArray, used to hold the output result.

        Returns
        -------
        out : NDArray
            The result NDArray(tuple) of result of computation.
        """
        cdef vector[string] sparam_keys
        cdef vector[string] sparam_vals
        cdef vector[NDArrayHandle] nd_args
        cdef vector[NDArrayHandle] output_vars
        cdef NDArrayHandle* p_output_vars
        cdef NDArrayHandle ret_handle
        cdef int pos_param_arg
        cdef int num_output

        pos_param_arg = 0

        for v in args:
            if isinstance(v, NDArrayBase):
                nd_args.push_back((<NDArrayBase>v).chandle)
            else:
                if pos_param_arg >= num_param_args:
                    raise ValueError("Too many positional arguments")
                if arguments[pos_param_arg] == 'dtype':
                    sparam_vals.push_back(c_str(np.dtype(v).name))
                else:
                    sparam_vals.push_back(c_str(str(v)))
                sparam_keys.push_back(c_str(arguments[pos_param_arg]))
                pos_param_arg = pos_param_arg + 1

        original_output = None
        for k, v in kwargs.items():
            if k == "out":
                original_output = v
                if isinstance(v, NDArrayBase):
                    output_vars.push_back((<NDArrayBase>v).chandle)
                else:
                    for item in v:
                        if not isinstance(item, NDArrayBase):
                            raise ValueError("out need to be of type NDArray")
                        output_vars.push_back((<NDArrayBase>v).chandle)
            elif k == 'dtype':
                sparam_vals.push_back(c_str(np.dtype(v).name))
                sparam_keys.push_back(c_str(k))
            else:
                sparam_vals.push_back(c_str(str(v)))
                sparam_keys.push_back(c_str(k))

        num_output = output_vars.size()
        if output_vars.size() == 0:
            output_vars.resize(1)
            p_output_vars = NULL
        else:
            p_output_vars = &output_vars[0]

        cdef vector[const char*] param_keys = SVec2Ptr(sparam_keys)
        cdef vector[const char*] param_vals = SVec2Ptr(sparam_vals)

        CALL(MXImperativeInvoke(
            handle,
            <int>nd_args.size(),
            &nd_args[0] if nd_args.size() != 0 else NULL,
            &num_output,
            &p_output_vars,
            <int>param_keys.size(),
            CBeginPtr(param_keys),
            CBeginPtr(param_vals)))

        if original_output is not None:
            return original_output

        if num_output == 1:
            return NewArray(p_output_vars[0])
        else:
            return tuple(NewArray(p_output_vars[i]) for i in range(num_output))

    # End of function declaration
    generic_ndarray_function.__name__ = func_name
    generic_ndarray_function.__doc__ = doc_str
    generic_ndarray_function.__module__ = 'mxnet.ndarray'
    return generic_ndarray_function


def _init_ndarray_module(nd_class, root_namespace):
    """List and add all the atomic symbol functions to current module."""
    cdef const char** op_name_ptrs
    cdef nn_uint size
    cdef vector[string] op_names
    cdef OpHandle handle

    _set_ndarray_class(nd_class)
    CALL(MXListAllOpNames(&size, &op_name_ptrs))
    for i in range(size):
        op_names.push_back(string(op_name_ptrs[i]))

    module_obj = _sys.modules["%s.ndarray" % root_namespace]
    module_internal = _sys.modules["%s._ndarray_internal" % root_namespace]
    for i in range(op_names.size()):
        CALL(NNGetOpHandle(op_names[i].c_str(), &handle))
        function = _make_ndarray_function(handle, op_names[i])
        if function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
        else:
            setattr(module_obj, function.__name__, function)
