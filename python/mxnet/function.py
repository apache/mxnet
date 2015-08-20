# coding: utf-8
"""NArray functions support of mxnet"""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import c_array
from .base import mx_uint, mx_float, NArrayHandle
from .base import check_call, MXNetError
from .narray import NArray, _new_empty_handle

class _Function(object):
    """Function Object."""
    # constants for type masks
    NARRAY_ARG_BEFORE_SCALAR = 1
    SCALAR_ARG_BEFORE_NARRAY = 1 << 1
    ACCEPT_EMPTY_MUTATE_TARGET = 1 << 2

    def __init__(self, handle, name):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the function handle of the function

        name : string
            the name of the function
        """
        self.handle = handle
        self.name = name
        n_used_vars = mx_uint()
        n_scalars = mx_uint()
        n_mutate_vars = mx_uint()
        type_mask = ctypes.c_int()
        check_call(_LIB.MXFuncDescribe(
            self.handle,
            ctypes.byref(n_used_vars),
            ctypes.byref(n_scalars),
            ctypes.byref(n_mutate_vars),
            ctypes.byref(type_mask)))
        self.n_used_vars = n_used_vars.value
        self.n_scalars = n_scalars.value
        self.n_mutate_vars = n_mutate_vars.value
        self.type_mask = type_mask.value
        # infer type of the function
        if (self.type_mask & _Function.NARRAY_ARG_BEFORE_SCALAR) != 0:
            self.use_vars_range = range(0, self.n_used_vars)
            self.scalar_range = range(self.n_used_vars,
                                      self.n_used_vars + self.n_scalars)
        else:
            self.scalar_range = range(0, self.n_scalars)
            self.use_vars_range = range(self.n_scalars,
                                        self.n_scalars + self.n_used_vars)
        self.accept_empty_mutate = (self.type_mask &
                                    _Function.ACCEPT_EMPTY_MUTATE_TARGET) != 0

    def __call__(self, *args, **kwargs):
        """Invoke this function by passing in parameters

        Parameters
        ----------
        *args: positional arguments
            positional arguments of input scalars and NArray
        mutate_vars: kwarg(optional)
            provide the NArray to store the result of the operation
        Returns
        -------
        the result NArrays of mutated result
        """
        if 'mutate_vars' in kwargs:
            mutate_vars = kwargs['mutate_vars']
            if isinstance(mutate_vars, NArray):
                mutate_vars = (mutate_vars,)
            if len(mutate_vars) != self.n_mutate_vars:
                raise MXNetError('expect %d mutate_vars in op.%s', self.n_mutate_vars, self.name)
        else:
            if self.accept_empty_mutate:
                mutate_vars = tuple(
                    NArray(_new_empty_handle()) for i in range(self.n_mutate_vars))
            else:
                raise MXNetError('mutate_vars argument is required to call op.%s' % self.name)

        self.invoke_with_handle_([args[i].handle for i in self.use_vars_range],
                                 [args[i] for i in self.scalar_range],
                                 [v.handle for v in mutate_vars])
        if self.n_mutate_vars == 1:
            return mutate_vars[0]
        else:
            return mutate_vars

    def invoke_with_handle_(self, use_vars, scalars, mutate_vars):
        """Invoke this function by passing in arguments as tuples

        This is a very primitive call to the function handle that
        involves passing in a C handle

        Parameters
        ----------
        fhandle : FunctionHandle
            function handle of C API

        use_vars : tuple
            tuple of NArray handles

        scalars : tuple
            tuple of real number arguments

        mutate_vars : tuple
            tuple of NArray handles to mutate
        """
        check_call(_LIB.MXFuncInvoke(
            self.handle,
            c_array(NArrayHandle, use_vars),
            c_array(mx_float, scalars),
            c_array(NArrayHandle, mutate_vars)))

class _FunctionRegistry(object):
    """Function Registry"""
    def __init__(self):
        plist = ctypes.POINTER(ctypes.c_void_p)()
        size = ctypes.c_uint()
        check_call(_LIB.MXListFunctions(ctypes.byref(size),
                                        ctypes.byref(plist)))
        hmap = {}
        for i in range(size.value):
            hdl = ctypes.c_void_p(plist[i])
            name = ctypes.c_char_p()
            check_call(_LIB.MXFuncGetName(hdl, ctypes.byref(name)))
            hmap[name.value] = _Function(hdl, name.value)
        self.__dict__.update(hmap)
