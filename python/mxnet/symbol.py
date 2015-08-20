# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-locals, fixme
"""Symbol support of mxnet"""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import c_array, c_str, mx_uint, NArrayHandle, ExecutorHandle, SymbolHandle
from .base import check_call
from .context import Context
from .executor import Executor

class Symbol(object):
    """Symbol is symbolic graph of the mxnet."""
    _registry = None

    @staticmethod
    def _init_symbol_creator_registry(symbol_creator_registry):
        """Initialize symbol creator registry

        Parameters
        ----------
        symbol_creator_registry:
            pass in symbol_creator_registry
        Returns
        -------
        the passed in registry
        """
        _registry = symbol_creator_registry
        return _registry

    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        self.handle = handle

    def __del__(self):
        check_call(_LIB.MXSymbolFree(self.handle))

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self):
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolCopy(self.handle,
                                     ctypes.byref(handle)))
        return Symbol(handle)

    def __call__(self, *args, **kwargs):
        """Invoke symbol as function on inputs.

        Parameters
        ----------
        args:
            provide positional arguments

        kwargs:
            provide keyword arguments
        Returns
        -------
        the resulting symbol
        """
        s = self.__deepcopy__()
        s._compose(*args, **kwargs)
        return s

    def _compose(self, *args, **kwargs):
        """Compose symbol on inputs.

        This call mutates the current symbol.

        Parameters
        ----------
        args:
            provide positional arguments

        kwargs:
            provide keyword arguments
        Returns
        -------
        the resulting symbol
        """
        name = kwargs.pop('name', None)
        if name:
            name = c_str(name)
        if len(args) != 0 and len(kwargs) != 0:
            raise TypeError('compose only accept input Symbols \
                either as positional or keyword arguments, not both')

        for arg in args:
            if not isinstance(arg, Symbol):
                raise TypeError('Compose expect `Symbol` as arguments')
        for _, val in kwargs.items():
            if not isinstance(val, Symbol):
                raise TypeError('Compose expect `Symbol` as arguments')

        num_args = len(args) + len(kwargs)
        if len(kwargs) != 0:
            keys = c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()])
            args = c_array(SymbolHandle, [s.handle for s in kwargs.values()])
        else:
            keys = None
            args = c_array(SymbolHandle, [s.handle for s in args])
        check_call(_LIB.MXSymbolCompose( \
                self.handle, name, num_args, keys, args))

    def list_arguments(self):
        """List all the arguments in the symbol.

        Returns
        -------
        args : list of string
            List of all the arguments.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListArguments( \
                self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [sarr[i] for i in range(size.value)]

    def list_returns(self):
        """List all returns in the symbol.

        Returns
        -------
        args: list of string
            List of all the returns.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListReturns( \
                self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [sarr[i] for i in range(size.value)]

    def infer_shape(self, *args, **kwargs):
        """Infer the shape of outputs and arguments of given known shapes of arguments.

        User can either pass in the known shapes in positional way or keyword argument way.
        Pair of Nones is returned if there is not enough information passed in.
        An error will be raised if there is inconsistency found in the known shapes passed in.

        Parameters
        ----------
        *args :
            Provide shape of arguments in a positional way.
            Unknown shape can be marked as None

        **kwargs :
            Provide keyword arguments of known shapes.

        Returns
        -------
        arg_shapes : list of tuple or None
            List of shapes of arguments.
            The order is in the same order as list_arguments()
        out_shapes : list of tuple or None
            List of shapes of outputs.
            The order is in the same order as list_returns()
        """
        if len(args) != 0 and len(kwargs) != 0:
            raise ValueError('Can only specify known argument \
                    shapes either by positional or kwargs way.')
        sdata = []
        indptr = [0]
        if len(args) != 0:
            keys = None
            for s in args:
                if s is not None:
                    if not isinstance(s, tuple):
                        raise TypeError('Argument need to be shapes(tuple)')
                    sdata.extend(s)
                indptr.append(len(sdata))
        else:
            keys = []
            for k, v in kwargs.items():
                keys.append(c_str(k))
                if not isinstance(v, tuple):
                    raise TypeError('Argument need to be shapes(tuple)')
                sdata.extend(v)
                indptr.append(len(sdata))
        arg_shape_size = mx_uint()
        arg_shape_ndim = ctypes.POINTER(mx_uint)()
        arg_shape_data = ctypes.POINTER(ctypes.POINTER(mx_uint))()
        out_shape_size = mx_uint()
        out_shape_ndim = ctypes.POINTER(mx_uint)()
        out_shape_data = ctypes.POINTER(ctypes.POINTER(mx_uint))()
        complete = ctypes.c_int()
        check_call(_LIB.MXSymbolInferShape( \
                self.handle, len(indptr) - 1, \
                c_array(ctypes.c_char_p, keys), \
                c_array(mx_uint, indptr), \
                c_array(mx_uint, sdata), \
                ctypes.byref(arg_shape_size), \
                ctypes.byref(arg_shape_ndim), \
                ctypes.byref(arg_shape_data), \
                ctypes.byref(out_shape_size), \
                ctypes.byref(out_shape_ndim), \
                ctypes.byref(out_shape_data), \
                ctypes.byref(complete)))
        if complete.value != 0:
            arg_shapes = [tuple(arg_shape_data[i][:arg_shape_ndim[i]]) \
                    for i in range(arg_shape_size.value)]
            out_shapes = [tuple(out_shape_data[i][:out_shape_ndim[i]]) \
                    for i in range(out_shape_size.value)]
            return (arg_shapes, out_shapes)
        else:
            return (None, None)

    def debug_str(self):
        """Get a debug string.

        Returns
        -------
        debug_str : string
            Debug string of the symbol.
        """
        debug_str = ctypes.c_char_p()
        check_call(_LIB.MXSymbolPrint( \
                self.handle, ctypes.byref(debug_str)))
        return debug_str.value

    def bind(self, ctx, args, args_grad, reqs):
        """bind current symbol to get an executor.

        Parameters
        ----------
        ctx: Context
            context executor to run on
        args: Array of NArray
            input args to the symbol
        args_grad: Array of NArray
            input args' gradient
        reqs: Array of enum
            graident requirements
        """
        # TODO(bing): consider a more friendly interface
        # For example, pass in args_grad by dict

        enum = {"null" : 0, "write_to" : 1, "in_place":2, "add_to" : 3}
        if not isinstance(ctx, Context):
            raise TypeError("Context type error")
        args_handle = c_array(NArrayHandle, [item.handle for item in args])
        args_grad_handle = c_array(NArrayHandle, [item.handle for item in args_grad])
        reqs_array = c_array(mx_uint, [mx_uint(enum[item]) for item in reqs])
        handle = ExecutorHandle()
        check_call(_LIB.MXExecutorBind(self.handle,
                                       mx_uint(ctx.device_mask),
                                       mx_uint(ctx.device_id),
                                       len(args),
                                       args_handle,
                                       args_grad_handle,
                                       reqs_array,
                                       ctypes.byref(handle)))
        return Executor(handle)
