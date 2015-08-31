# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme
"""Symbol support of mxnet"""
from __future__ import absolute_import

import ctypes
import sys
from .base import _LIB
from .base import c_array, c_str, mx_uint, py_str, string_types
from .base import NArrayHandle, ExecutorHandle, SymbolHandle
from .base import check_call
from .context import Context
from .executor import Executor

class Symbol(object):
    """Symbol is symbolic graph of the mxnet."""

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
        check_call(_LIB.MXSymbolCompose(
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
        check_call(_LIB.MXSymbolListArguments(
            self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [py_str(sarr[i]) for i in range(size.value)]

    def list_returns(self):
        """List all returns in the symbol.

        Returns
        -------
        args: list of string
            List of all the returns.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListReturns(
            self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [py_str(sarr[i]) for i in range(size.value)]

    def list_auxiliary_states(self):
        """List all auxiliary states in the symbool.

        Returns
        -------
        args: list of string
            List of all the auxiliary
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListAuxiliaryStates(
            self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [py_str(sarr[i]) for i in range(size.value)]

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
        aux_shapes : list of tuple or None
            List of shapes of outputs.
            The order is in the same order as list_auxiliary()
        """
        # pylint: disable=too-many-locals
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
        aux_shape_size = mx_uint()
        aux_shape_ndim = ctypes.POINTER(mx_uint)()
        aux_shape_data = ctypes.POINTER(ctypes.POINTER(mx_uint))()
        complete = ctypes.c_int()
        check_call(_LIB.MXSymbolInferShape(
            self.handle, len(indptr) - 1,
            c_array(ctypes.c_char_p, keys),
            c_array(mx_uint, indptr),
            c_array(mx_uint, sdata),
            ctypes.byref(arg_shape_size),
            ctypes.byref(arg_shape_ndim),
            ctypes.byref(arg_shape_data),
            ctypes.byref(out_shape_size),
            ctypes.byref(out_shape_ndim),
            ctypes.byref(out_shape_data),
            ctypes.byref(aux_shape_size),
            ctypes.byref(aux_shape_ndim),
            ctypes.byref(aux_shape_data),
            ctypes.byref(complete)))
        if complete.value != 0:
            arg_shapes = [
                tuple(arg_shape_data[i][:arg_shape_ndim[i]]) for i in range(arg_shape_size.value)]
            out_shapes = [
                tuple(out_shape_data[i][:out_shape_ndim[i]]) for i in range(out_shape_size.value)]
            aux_shapes = [
                tuple(aux_shape_data[i][:aux_shape_ndim[i]]) for i in range(aux_shape_size.value)]
            return (arg_shapes, out_shapes, aux_shapes)
        else:
            return (None, None)
        # pylint: enable=too-many-locals

    def debug_str(self):
        """Get a debug string.

        Returns
        -------
        debug_str : string
            Debug string of the symbol.
        """
        debug_str = ctypes.c_char_p()
        check_call(_LIB.MXSymbolPrint(
            self.handle, ctypes.byref(debug_str)))
        return py_str(debug_str.value)

    def bind(self, ctx, args, args_grad, reqs, aux_states=None):
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
        aux_states: Array of NArray
            input auxiliary states to the symbol
        """
        # TODO(bing): consider a more friendly interface
        # For example, pass in args_grad by dict
        enum = {"null" : 0, "write_to" : 1, "in_place":2, "add_to" : 3}
        if not isinstance(ctx, Context):
            raise TypeError("Context type error")
        if aux_states == None:
            aux_states = []
        args_handle = c_array(NArrayHandle, [item.handle for item in args])
        args_grad_handle = c_array(NArrayHandle, [item.handle for item in args_grad])
        reqs_array = c_array(mx_uint, [mx_uint(enum[item]) for item in reqs])
        aux_args_handle = c_array(NArrayHandle, [item.handle for item in aux_args])
        handle = ExecutorHandle()
        check_call(_LIB.MXExecutorBind(self.handle,
                                       mx_uint(ctx.device_mask),
                                       mx_uint(ctx.device_id),
                                       len(args),
                                       args_handle,
                                       args_grad_handle,
                                       reqs_array,
                                       len(aux_args),
                                       aux_args_handle,
                                       ctypes.byref(handle)))
        return Executor(handle)

    def grad(self, wrt):
        """get the autodiff of current symbol.

        Parameters
        ----------
        wrt: Array of String
            keyword arguments of the symbol that the gradients are taken.
        """
        handle = SymbolHandle()
        c_wrt = c_array(ctypes.c_char_p, [c_str(key) for key in wrt])
        check_call(_LIB.MXSymbolGrad(self.handle,
                                     mx_uint(len(wrt)),
                                     c_wrt,
                                     ctypes.byref(handle)))
        return Symbol(handle)

def Variable(name):
    """Create a symbolic variable with specified name.

    Parameters
    ----------
    name : str
       Name of the variable.

    Returns
    -------
    variable : Symbol
        The created variable symbol.
    """
    if not isinstance(name, string_types):
        raise TypeError('Expect a string for variable `name`')
    handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateVariable(c_str(name), ctypes.byref(handle)))
    return Symbol(handle)


def Group(symbols):
    """Create a symbolic variable that groups several symbols together.

    Parameters
    ----------
    symbols : list
        List of symbols to be grouped.

    Returns
    -------
    sym : Symbol
        The created group symbol.
     """
    ihandles = []
    for sym in symbols:
        if not isinstance(sym, Symbol):
            raise TypeError('Expect Symbols in the list input')
        ihandles.append(sym.handle)
    handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateGroup(
        len(ihandles), c_array(SymbolHandle, ihandles), ctypes.byref(handle)))
    return Symbol(handle)


def _make_atomic_symbol_function(handle):
    """Create an atomic symbol function by handle and funciton name."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
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
               'name : string, required.\n' +
               '    Name of the resulting symbol.\n\n' +
               'Returns\n' +
               '-------\n' +
               'symbol: Symbol\n'+
               '    The result symbol.')
    doc_str = doc_str % (py_str(desc.value), '\n'.join(param_str))

    def creator(*args, **kwargs):
        """Activation Operator of Neural Net.
        The parameters listed below can be passed in as keyword arguments.

        Parameters
        ----------
        name : string, required.
            Name of the resulting symbol.

        Returns
        -------
        symbol: Symbol
            the resulting symbol
        """
        param_keys = []
        param_vals = []
        symbol_kwargs = {}
        name = kwargs.pop('name', None)

        for k, v in kwargs.items():
            if isinstance(v, Symbol):
                symbol_kwargs[k] = v
            else:
                param_keys.append(c_str(k))
                param_vals.append(c_str(str(v)))
        # create atomic symbol
        param_keys = c_array(ctypes.c_char_p, param_keys)
        param_vals = c_array(ctypes.c_char_p, param_vals)
        sym_handle = SymbolHandle()
        check_call(_LIB.MXSymbolCreateAtomicSymbol(
            handle, len(param_keys),
            param_keys, param_vals,
            ctypes.byref(sym_handle)))

        if len(args) != 0 and len(symbol_kwargs) != 0:
            raise TypeError(
                '%s can only accept input'
                'Symbols either as positional or keyword arguments, not both' % func_name)

        s = Symbol(sym_handle)
        s._compose(*args, name=name, **symbol_kwargs)
        return s

    creator.__name__ = func_name
    creator.__doc__ = doc_str
    return creator


def _init_symbol_module():
    """List and add all the atomic symbol functions to current module."""
    plist = ctypes.POINTER(ctypes.c_void_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXSymbolListAtomicSymbolCreators(ctypes.byref(size),
                                                     ctypes.byref(plist)))
    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = SymbolHandle(plist[i])
        function = _make_atomic_symbol_function(hdl)
        setattr(module_obj, function.__name__, function)

# Initialize the atomic symbo in startups
_init_symbol_module()
