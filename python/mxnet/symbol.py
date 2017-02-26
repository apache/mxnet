# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
# pylint: disable=import-error, no-name-in-module
"""Symbolic configuration API of mxnet."""
from __future__ import absolute_import as _abs

import ctypes
import warnings
from numbers import Number

import os as _os
import sys as _sys
import numpy as _numpy

from .base import _LIB, numeric_types
from .base import c_array, c_str, mx_uint, py_str, string_types, mx_real_t
from .base import NDArrayHandle, ExecutorHandle, SymbolHandle
from .base import check_call, MXNetError
from .context import Context
from .ndarray import NDArray, zeros as _nd_zeros, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .executor import Executor
from . import _symbol_internal as _internal
from .attribute import AttrScope

# Use different verison of SymbolBase
# When possible, use cython to speedup part of computation.
try:
    if int(_os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from ._ctypes.symbol import SymbolBase, _init_symbol_module
    elif _sys.version_info >= (3, 0):
        from ._cy3.symbol import SymbolBase, _init_symbol_module
    else:
        from ._cy2.symbol import SymbolBase, _init_symbol_module
except ImportError:
    if int(_os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from ._ctypes.symbol import SymbolBase, _init_symbol_module


class Symbol(SymbolBase):
    """Symbol is symbolic graph of the mxnet."""
    # disable dictionary storage, also do not have parent type.
    # pylint: disable=no-member
    __slots__ = []

    def __repr__(self):
        """Get a string representation of the symbol."""
        name = self.name
        return '<%s %s>' % (self.__class__.__name__,
                            'Grouped' if name is None else name)

    def __iter__(self):
        """Return all outputs in a list"""
        return (self[i] for i in self.list_outputs())

    def __add__(self, other):
        if isinstance(other, Symbol):
            return _internal._Plus(self, other)
        if isinstance(other, Number):
            return _internal._PlusScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Symbol):
            return _internal._Minus(self, other)
        if isinstance(other, Number):
            return _internal._MinusScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rsub__(self, other):
        if isinstance(other, Number):
            return _internal._RMinusScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __mul__(self, other):
        if isinstance(other, Symbol):
            return _internal._Mul(self, other)
        if isinstance(other, Number):
            return _internal._MulScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Symbol):
            return _internal._Div(self, other)
        if isinstance(other, Number):
            return _internal._DivScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rdiv__(self, other):
        if isinstance(other, Number):
            return _internal._RDivScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __pow__(self, other):
        if isinstance(other, Symbol):
            return _internal._Power(self, other)
        if isinstance(other, Number):
            return _internal._PowerScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __neg__(self):
        return self.__mul__(-1.0)

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, _):
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolCopy(self.handle,
                                     ctypes.byref(handle)))
        return Symbol(handle)

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return _internal._equal(self, other)
        if isinstance(other, numeric_types):
            return _internal._equal_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __ne__(self, other):
        if isinstance(other, Symbol):
            return _internal._not_equal(self, other)
        if isinstance(other, numeric_types):
            return _internal._not_equal_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __gt__(self, other):
        if isinstance(other, Symbol):
            return _internal._greater(self, other)
        if isinstance(other, numeric_types):
            return _internal._greater_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __ge__(self, other):
        if isinstance(other, Symbol):
            return _internal._greater_equal(self, other)
        if isinstance(other, numeric_types):
            return _internal._greater_equal_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return _internal._lesser(self, other)
        if isinstance(other, numeric_types):
            return _internal._lesser_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __le__(self, other):
        if isinstance(other, Symbol):
            return _internal._lesser_equal(self, other)
        if isinstance(other, numeric_types):
            return _internal._lesser_equal_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __getstate__(self):
        handle = self.handle
        if handle is not None:
            return {'handle': self.tojson()}
        else:
            return {'handle': None}

    def __setstate__(self, state):
        # pylint: disable=assigning-non-slot
        handle = state['handle']
        if handle is not None:
            json_str = handle
            handle = SymbolHandle()
            check_call(_LIB.MXSymbolCreateFromJSON(c_str(json_str), ctypes.byref(handle)))
            self.handle = handle
        else:
            self.handle = None

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
        s = self.__copy__()
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
        for val in kwargs.values():
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

    def __getitem__(self, index):
        if isinstance(index, string_types):
            idx = None
            for i, name in enumerate(self.list_outputs()):
                if name == index:
                    if idx is not None:
                        raise ValueError('There are multiple outputs with name \"%s\"' % index)
                    idx = i
            if idx is None:
                raise ValueError('Cannot find output that matches name \"%s\"' % index)
            index = idx
        if not isinstance(index, int):
            raise TypeError('Symbol only support integer index to fetch i-th output')
        if index >= (len(self.list_outputs())):
            # Important, python determines the end by this exception
            raise IndexError
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolGetOutput(
            self.handle, mx_uint(index), ctypes.byref(handle)))
        return Symbol(handle=handle)

    @property
    def name(self):
        """Get name string from the symbol, this function only works for non-grouped symbol.

        Returns
        -------
        value : str
            The name of this symbol, returns None for grouped symbol.
        """
        ret = ctypes.c_char_p()
        success = ctypes.c_int()
        check_call(_LIB.MXSymbolGetName(
            self.handle, ctypes.byref(ret), ctypes.byref(success)))
        if success.value != 0:
            return py_str(ret.value)
        else:
            return None

    def attr(self, key):
        """Get attribute string from the symbol, this function only works for non-grouped symbol.

        Parameters
        ----------
        key : str
            The key to get attribute from.

        Returns
        -------
        value : str
            The attribute value of the key, returns None if attribute do not exist.
        """
        ret = ctypes.c_char_p()
        success = ctypes.c_int()
        check_call(_LIB.MXSymbolGetAttr(
            self.handle, c_str(key), ctypes.byref(ret), ctypes.byref(success)))
        if success.value != 0:
            return py_str(ret.value)
        else:
            return None

    def list_attr(self, recursive=False):
        """Get all attributes from the symbol.

        Returns
        -------
        ret : dict of str to str
            a dicitonary mapping attribute keys to values
        """
        if recursive:
            raise DeprecationWarning("Symbol.list_attr with recursive=True has been deprecated. "
                                     "Please use attr_dict instead.")
        size = mx_uint()
        pairs = ctypes.POINTER(ctypes.c_char_p)()
        f_handle = _LIB.MXSymbolListAttrShallow
        check_call(f_handle(self.handle, ctypes.byref(size), ctypes.byref(pairs)))
        return {py_str(pairs[i*2]): py_str(pairs[i*2+1]) for i in range(size.value)}

    def attr_dict(self):
        """Recursively get all attributes from the symbol and its childrens

        Returns
        -------
        ret : dict of str to dict
            Returns a dict whose keys are names of the symbol and its children.
            Values of the returned dict are dictionaries that map attribute keys to values
        """
        size = mx_uint()
        pairs = ctypes.POINTER(ctypes.c_char_p)()
        f_handle = _LIB.MXSymbolListAttr
        check_call(f_handle(self.handle, ctypes.byref(size), ctypes.byref(pairs)))
        ret = {}
        for i in range(size.value):
            name, key = py_str(pairs[i*2]).split('$')
            val = py_str(pairs[i*2+1])
            if name not in ret:
                ret[name] = {}
            ret[name][key] = val
        return ret

    def _set_attr(self, **kwargs):
        """Set the attribute of the symbol.

        Parameters
        ----------
        **kwargs
            The attributes to set
        """
        for key, value in kwargs.items():
            if not isinstance(value, string_types):
                raise ValueError("Set Attr only accepts string values")
            check_call(_LIB.MXSymbolSetAttr(
                self.handle, c_str(key), c_str(str(value))))

    def get_internals(self):
        """Get a new grouped symbol whose output contains
        internal outputs of this symbol.

        Returns
        -------
        sgroup : Symbol
            The internal of the symbol.
        """
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolGetInternals(
            self.handle, ctypes.byref(handle)))
        return Symbol(handle=handle)

    def get_children(self):
        """Get a new grouped symbol whose output contains
        inputs to output nodes of the original symbol

        Returns
        -------
        sgroup : Symbol or None
            The children of the head node. If the symbol has no
            inputs None will be returned.
        """
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolGetChildren(
            self.handle, ctypes.byref(handle)))
        ret = Symbol(handle=handle)
        if len(ret.list_outputs()) == 0:
            return None
        return ret

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

    def list_outputs(self):
        """List all outputs in the symbol.

        Returns
        -------
        returns : list of string
            List of all the outputs.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListOutputs(
            self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [py_str(sarr[i]) for i in range(size.value)]

    def list_auxiliary_states(self):
        """List all auxiliary states in the symbol.

        Returns
        -------
        aux_states : list of string
            List the names of the auxiliary states.

        Notes
        -----
        Auxiliary states are special states of symbols that do not corresponds to an argument,
        and do not have gradient. But still be useful for the specific operations.
        A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
        Most operators do not have Auxiliary states.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListAuxiliaryStates(
            self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [py_str(sarr[i]) for i in range(size.value)]

    def infer_type(self, *args, **kwargs):
        """Infer the type of outputs and arguments of given known types of arguments.

        User can either pass in the known types in positional way or keyword argument way.
        Tuple of Nones is returned if there is not enough information passed in.
        An error will be raised if there is inconsistency found in the known types passed in.

        Parameters
        ----------
        *args :
            Provide type of arguments in a positional way.
            Unknown type can be marked as None

        **kwargs :
            Provide keyword arguments of known types.

        Returns
        -------
        arg_types : list of numpy.dtype or None
            List of types of arguments.
            The order is in the same order as list_arguments()
        out_types : list of numpy.dtype or None
            List of types of outputs.
            The order is in the same order as list_outputs()
        aux_types : list of numpy.dtype or None
            List of types of outputs.
            The order is in the same order as list_auxiliary()
        """
        # pylint: disable=too-many-locals
        if len(args) != 0 and len(kwargs) != 0:
            raise ValueError('Can only specify known argument \
                    types either by positional or kwargs way.')
        sdata = []
        if len(args) != 0:
            keys = None
            for s in args:
                if s is not None:
                    s = _numpy.dtype(s).type
                    if s not in _DTYPE_NP_TO_MX:
                        raise TypeError('Argument need to be one of '+str(_DTYPE_NP_TO_MX))
                    sdata.append(_DTYPE_NP_TO_MX[s])
                else:
                    sdata.append(-1)
        else:
            keys = []
            for k, v in kwargs.items():
                v = _numpy.dtype(v).type
                if v in _DTYPE_NP_TO_MX:
                    keys.append(c_str(k))
                    sdata.append(_DTYPE_NP_TO_MX[v])
        arg_type_size = mx_uint()
        arg_type_data = ctypes.POINTER(ctypes.c_int)()
        out_type_size = mx_uint()
        out_type_data = ctypes.POINTER(ctypes.c_int)()
        aux_type_size = mx_uint()
        aux_type_data = ctypes.POINTER(ctypes.c_int)()
        complete = ctypes.c_int()
        check_call(_LIB.MXSymbolInferType(
            self.handle,
            mx_uint(len(sdata)),
            c_array(ctypes.c_char_p, keys),
            c_array(ctypes.c_int, sdata),
            ctypes.byref(arg_type_size),
            ctypes.byref(arg_type_data),
            ctypes.byref(out_type_size),
            ctypes.byref(out_type_data),
            ctypes.byref(aux_type_size),
            ctypes.byref(aux_type_data),
            ctypes.byref(complete)))
        if complete.value != 0:
            arg_types = [
                _DTYPE_MX_TO_NP[arg_type_data[i]] for i in range(arg_type_size.value)]
            out_types = [
                _DTYPE_MX_TO_NP[out_type_data[i]] for i in range(out_type_size.value)]
            aux_types = [
                _DTYPE_MX_TO_NP[aux_type_data[i]] for i in range(aux_type_size.value)]
            return (arg_types, out_types, aux_types)
        else:
            return (None, None, None)
        # pylint: enable=too-many-locals

    def infer_shape(self, *args, **kwargs):
        """Infer the shape of outputs and arguments of given known shapes of arguments.

        User can either pass in the known shapes in positional way or keyword argument way.
        Tuple of Nones is returned if there is not enough information passed in.
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
            The order is in the same order as list_outputs()
        aux_shapes : list of tuple or None
            List of shapes of outputs.
            The order is in the same order as list_auxiliary()
        """
        try:
            res = self._infer_shape_impl(False, *args, **kwargs)
            if res[1] is None:
                arg_shapes, _, _ = self._infer_shape_impl(True, *args, **kwargs)
                arg_names = self.list_arguments()
                unknowns = []
                for name, shape in zip(arg_names, arg_shapes):
                    if not shape or not _numpy.prod(shape):
                        if len(unknowns) >= 10:
                            unknowns.append('...')
                            break
                        unknowns.append('%s: %s'%(name, str(shape)))
                warnings.warn(
                    "Cannot decide shape for the following arguments " +
                    "(0s in shape means unknown dimensions). " +
                    "Consider providing them as input:\n\t" +
                    "\n\t".join(unknowns), stacklevel=2)
            return res
        except MXNetError:
            print("infer_shape error. Arguments:")
            for i, arg in enumerate(args):
                print("  #%d: %s" % (i, arg))
            for k, v in kwargs.items():
                print("  %s: %s" % (k, v))
            raise

    def infer_shape_partial(self, *args, **kwargs):
        """Partially infer the shape. The same as infer_shape, except that the partial
        results can be returned.
        """
        return self._infer_shape_impl(True, *args, **kwargs)

    def _infer_shape_impl(self, partial, *args, **kwargs):
        """The actual implementation for calling shape inference API."""
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
                if isinstance(v, tuple):
                    keys.append(c_str(k))
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
        if partial:
            infer_func = _LIB.MXSymbolInferShapePartial
        else:
            infer_func = _LIB.MXSymbolInferShape
        check_call(infer_func(
            self.handle,
            mx_uint(len(indptr) - 1),
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
            return (None, None, None)
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

    def save(self, fname):
        """Save symbol into file.

        You can also use pickle to do the job if you only work on python.
        The advantage of load/save is the file is language agnostic.
        This means the file saved using save can be loaded by other language binding of mxnet.
        You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)

        Parameters
        ----------
        fname : str
            The name of the file
            - s3://my-bucket/path/my-s3-symbol
            - hdfs://my-bucket/path/my-hdfs-symbol
            - /path-to/my-local-symbol

        See Also
        --------
        symbol.load : Used to load symbol from file.
        """
        if not isinstance(fname, string_types):
            raise TypeError('fname need to be string')
        check_call(_LIB.MXSymbolSaveToFile(self.handle, c_str(fname)))

    def tojson(self):
        """Save symbol into a JSON string.

        See Also
        --------
        symbol.load_json : Used to load symbol from JSON string.
        """
        json_str = ctypes.c_char_p()
        check_call(_LIB.MXSymbolSaveToJSON(self.handle, ctypes.byref(json_str)))
        return py_str(json_str.value)

    @staticmethod
    def _get_ndarray_inputs(arg_key, args, arg_names, allow_missing):
        """Helper function to get ndarray lists handles from various inputs.

        Parameters
        ----------
        arg_key : str
            The name of argument, used for error message.

        args : list of NDArray or dict of str to NDArray
            Input arguments to the symbols.
            If type is list of NDArray, the position is in the same order of arg_names.
            If type is dict of str to NDArray, then it maps the name of arguments
            to the corresponding NDArray,

        args_names : list of string
            List of argument names.

        allow_missing : boolean
            Whether missing argument is allowed.
            When allowed, the missing handle will be set to None(null)

        Returns
        -------
        handles : list of NDArrayHandle
            The positional list of NDArrayHandles generated from input.
        """
        # setup args
        arg_handles = []
        arg_arrays = []
        if isinstance(args, list):
            if len(args) != len(arg_names):
                raise ValueError('Length of %s do not match number of arguments' % arg_key)
            for narr in args:
                if not isinstance(narr, NDArray):
                    raise TypeError('Only Accept list of NDArrays or dict of str to NDArray')
                arg_handles.append(narr.handle)
            arg_arrays = args
        elif isinstance(args, dict):
            for name in arg_names:
                if name in args:
                    narr = args[name]
                    if not isinstance(narr, NDArray):
                        raise TypeError('Only Accept list of NDArrays or dict of str to NDArray')
                    arg_handles.append(narr.handle)
                    arg_arrays.append(narr)
                else:
                    if allow_missing:
                        arg_handles.append(None)
                        arg_arrays.append(None)
                    else:
                        raise ValueError('key `%s` is missing in `%s`' % (name, arg_key))
        else:
            raise TypeError('Only Accept list of NDArrays or dict of str to NDArray')
        return c_array(NDArrayHandle, arg_handles), arg_arrays

    def simple_bind(self, ctx,
                    grad_req='write',
                    type_dict=None,
                    group2ctx=None,
                    **kwargs):
        """Bind current symbol to get an executor, allocate all the ndarrays needed.
        Allows specifying data types.

        This function will ask user to pass in ndarray of position
        they like to bind to, and it will automatically allocate the ndarray
        for arguments and auxiliary states that user did not specify explicitly.

        Parameters
        ----------
        ctx : Context
            The device context the generated executor to run on.

        grad_req: string
            {'write', 'add', 'null'}, or list of str or dict of str to str, optional
            Specifies how we should update the gradient to the args_grad.
            - 'write' means everytime gradient is write to specified args_grad NDArray.
            - 'add' means everytime gradient is add to the specified NDArray.
            - 'null' means no action is taken, the gradient may not be calculated.

        type_dict  : dict of str->numpy.dtype
            Input type dictionary, name->dtype

        group2ctx : dict of string to mx.Context
            The dict mapping the ``ctx_group`` attribute to the context assignment.

        kwargs : dict of str->shape
            Input shape dictionary, name->shape

        Returns
        -------
        executor : mxnet.Executor
            The generated Executor
        """
        # pylint: disable=too-many-locals
        if type_dict is None:
            attrs = self.attr_dict()
            type_dict = {k: mx_real_t for k in self.list_arguments()
                         if k not in attrs or '__dtype__' not in attrs[k]}
        arg_shapes, _, aux_shapes = self.infer_shape(**kwargs)
        arg_types, _, aux_types = self.infer_type(**type_dict)

        if arg_shapes is None or arg_types is None:
            raise ValueError("Input node is not complete")

        if group2ctx is not None:
            attr_dict = self.attr_dict()
            arg_ctx = [group2ctx.get(attr_dict[name]['__ctx_group__'], ctx) \
                         if name in attr_dict and '__ctx_group__' in attr_dict[name] \
                         else ctx for name in self.list_arguments()]
            aux_ctx = [group2ctx.get(attr_dict[name]['__ctx_group__'], ctx) \
                         if name in attr_dict and '__ctx_group__' in attr_dict[name] \
                         else ctx for name in self.list_auxiliary_states()]
        else:
            arg_ctx = [ctx] * len(arg_shapes)
            aux_ctx = [ctx] * len(aux_shapes)

        # alloc space
        arg_ndarrays = [
            _nd_zeros(shape, dev, dtype=dtype)
            for dtype, dev, shape in zip(arg_types, arg_ctx, arg_shapes)]
        if grad_req != 'null':
            grad_ndarrays = {}
            for name, shape, dev, dtype in zip(
                    self.list_arguments(), arg_shapes, arg_ctx, arg_types):
                if not isinstance(grad_req, dict) or grad_req[name] != 'null':
                    grad_ndarrays[name] = _nd_zeros(shape, dev, dtype=dtype)
        else:
            grad_ndarrays = None

        aux_ndarrays = [_nd_zeros(shape, dev, dtype=dtype)
                        for shape, dev, dtype in zip(aux_shapes, aux_ctx, aux_types)]
        executor = self.bind(ctx, arg_ndarrays,
                             grad_ndarrays, grad_req, aux_ndarrays,
                             group2ctx=group2ctx)
        return executor

    def bind(self, ctx, args, args_grad=None, grad_req='write',
             aux_states=None, group2ctx=None, shared_exec=None):
        """Bind current symbol to get an executor.

        Parameters
        ----------
        ctx : Context
            The device context the generated executor to run on.

        args : list of NDArray or dict of str to NDArray
            Input arguments to the symbol.

            - If type is list of NDArray, the position is in the same order of list_arguments.
            - If type is dict of str to NDArray, then it maps the name of arguments
              to the corresponding NDArray.
            - In either case, all the arguments must be provided.

        args_grad : list of NDArray or dict of str to NDArray, optional
            When specified, args_grad provide NDArrays to hold
            the result of gradient value in backward.

            - If type is list of NDArray, the position is in the same order of list_arguments.
            - If type is dict of str to NDArray, then it maps the name of arguments
              to the corresponding NDArray.
            - When the type is dict of str to NDArray, users only need to provide the dict
              for needed argument gradient.
              Only the specified argument gradient will be calculated.

        grad_req : {'write', 'add', 'null'}, or list of str or dict of str to str, optional
            Specifies how we should update the gradient to the args_grad.

            - 'write' means everytime gradient is write to specified args_grad NDArray.
            - 'add' means everytime gradient is add to the specified NDArray.
            - 'null' means no action is taken, the gradient may not be calculated.

        aux_states : list of NDArray, or dict of str to NDArray, optional
            Input auxiliary states to the symbol, only need to specify when
            list_auxiliary_states is not empty.

            - If type is list of NDArray, the position is in the same order of list_auxiliary_states
            - If type is dict of str to NDArray, then it maps the name of auxiliary_states
              to the corresponding NDArray,
            - In either case, all the auxiliary_states need to be provided.

        group2ctx : dict of string to mx.Context
            The dict mapping the ``ctx_group`` attribute to the context assignment.

        shared_exec : mx.executor.Executor
            Executor to share memory with. This is intended for runtime reshaping, variable length
            sequences, etc. The returned executor shares state with shared_exec, and should not be
            used in parallel with it.

        Returns
        -------
        executor : Executor
            The generated Executor

        Notes
        -----
        Auxiliary states are special states of symbols that do not corresponds to an argument,
        and do not have gradient. But still be useful for the specific operations.
        A common example of auxiliary state is the moving_mean and moving_variance in BatchNorm.
        Most operators do not have auxiliary states and this parameter can be safely ignored.

        User can give up gradient by using a dict in args_grad and only specify
        gradient they interested in.
        """
        # pylint: disable=too-many-locals, too-many-branches
        if not isinstance(ctx, Context):
            raise TypeError("Context type error")

        listed_arguments = self.list_arguments()
        args_handle, args = self._get_ndarray_inputs('args', args, listed_arguments, False)
        # setup args gradient
        if args_grad is None:
            args_grad_handle = c_array(NDArrayHandle, [None] * len(args))
        else:
            args_grad_handle, args_grad = self._get_ndarray_inputs(
                'args_grad', args_grad, listed_arguments, True)

        if aux_states is None:
            aux_states = []
        aux_args_handle, aux_states = self._get_ndarray_inputs(
            'aux_states', aux_states, self.list_auxiliary_states(), False)

        # setup requirements
        req_map = {'null': 0, 'write': 1, 'add': 3}
        if isinstance(grad_req, string_types):
            if grad_req not in req_map:
                raise ValueError('grad_req must be in %s' % str(req_map))
            reqs_array = c_array(mx_uint, [mx_uint(req_map[grad_req])] * len(listed_arguments))
        elif isinstance(grad_req, list):
            reqs_array = c_array(mx_uint, [mx_uint(req_map[item]) for item in grad_req])
        elif isinstance(grad_req, dict):
            req_array = []
            for name in listed_arguments:
                if name in grad_req:
                    req_array.append(mx_uint(req_map[grad_req[name]]))
                else:
                    req_array.append(mx_uint(0))
            reqs_array = c_array(mx_uint, req_array)

        ctx_map_keys = []
        ctx_map_dev_types = []
        ctx_map_dev_ids = []

        if group2ctx:
            for key, val in group2ctx.items():
                ctx_map_keys.append(c_str(key))
                ctx_map_dev_types.append(ctypes.c_int(val.device_typeid))
                ctx_map_dev_ids.append(ctypes.c_int(val.device_id))

        handle = ExecutorHandle()
        shared_handle = shared_exec.handle if shared_exec is not None else ExecutorHandle()
        check_call(_LIB.MXExecutorBindEX(self.handle,
                                         ctypes.c_int(ctx.device_typeid),
                                         ctypes.c_int(ctx.device_id),
                                         mx_uint(len(ctx_map_keys)),
                                         c_array(ctypes.c_char_p, ctx_map_keys),
                                         c_array(ctypes.c_int, ctx_map_dev_types),
                                         c_array(ctypes.c_int, ctx_map_dev_ids),
                                         mx_uint(len(args)),
                                         args_handle,
                                         args_grad_handle,
                                         reqs_array,
                                         mx_uint(len(aux_states)),
                                         aux_args_handle,
                                         shared_handle,
                                         ctypes.byref(handle)))
        executor = Executor(handle, self, ctx, grad_req, group2ctx)
        executor.arg_arrays = args
        executor.grad_arrays = args_grad
        executor.aux_arrays = aux_states
        return executor

    def grad(self, wrt):
        """Get the autodiff of current symbol.

        This function can only be used if current symbol is a loss function.

        Parameters
        ----------
        wrt : Array of String
            keyword arguments of the symbol that the gradients are taken.

        Returns
        -------
        grad : Symbol
            A gradient Symbol with returns to be the corresponding gradients.
        """
        handle = SymbolHandle()
        c_wrt = c_array(ctypes.c_char_p, [c_str(key) for key in wrt])
        check_call(_LIB.MXSymbolGrad(self.handle,
                                     mx_uint(len(wrt)),
                                     c_wrt,
                                     ctypes.byref(handle)))
        return Symbol(handle)
    # pylint: enable= no-member


def Variable(name, attr=None, shape=None, lr_mult=None, wd_mult=None, dtype=None, init=None):
    """Create a symbolic variable with specified name.

    Parameters
    ----------
    name : str
        Name of the variable.
    attr : dict of string -> string
        Additional attributes to set on the variable.
    shape : tuple
        Optionally, one can specify the shape of a variable. This will be used during
        shape inference. If user specified a different shape for this variable using
        keyword argument when calling shape inference, this shape information will be ignored.
    lr_mult : float
        Specify learning rate muliplier for this variable.
    wd_mult : float
        Specify weight decay muliplier for this variable.
    dtype : str or numpy.dtype
        Similar to shape, we can specify dtype for this variable.
    init : initializer (mxnet.init.*)
        Specify initializer for this variable to override the default initializer

    Returns
    -------
    variable : Symbol
        The created variable symbol.
    """
    if not isinstance(name, string_types):
        raise TypeError('Expect a string for variable `name`')
    handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateVariable(c_str(name), ctypes.byref(handle)))
    ret = Symbol(handle)
    attr = AttrScope.current.get(attr)
    attr = {} if attr is None else attr
    if shape is not None:
        attr['__shape__'] = str(shape)
    if lr_mult is not None:
        attr['__lr_mult__'] = str(lr_mult)
    if wd_mult is not None:
        attr['__wd_mult__'] = str(wd_mult)
    if dtype is not None:
        attr['__dtype__'] = str(_DTYPE_NP_TO_MX[_numpy.dtype(dtype).type])
    if init is not None:
        attr['__init__'] = init.dumps()
    ret._set_attr(**attr)
    return ret


def Group(symbols):
    """Create a symbol that groups symbols together.

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
        mx_uint(len(ihandles)),
        c_array(SymbolHandle, ihandles), ctypes.byref(handle)))
    return Symbol(handle)


def load(fname):
    """Load symbol from a JSON file.

    You can also use pickle to do the job if you only work on python.
    The advantage of load/save is the file is language agnostic.
    This means the file saved using save can be loaded by other language binding of mxnet.
    You also get the benefit being able to directly load/save from cloud storage(S3, HDFS)

    Parameters
    ----------
    fname : str
        The name of the file, examples:

        - `s3://my-bucket/path/my-s3-symbol`
        - `hdfs://my-bucket/path/my-hdfs-symbol`
        - `/path-to/my-local-symbol`

    Returns
    -------
    sym : Symbol
        The loaded symbol.

    See Also
    --------
    Symbol.save : Used to save symbol into file.
    """
    if not isinstance(fname, string_types):
        raise TypeError('fname need to be string')
    handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateFromFile(c_str(fname), ctypes.byref(handle)))
    return Symbol(handle)


def load_json(json_str):
    """Load symbol from json string.

    Parameters
    ----------
    json_str : str
        A json string.

    Returns
    -------
    sym : Symbol
        The loaded symbol.

    See Also
    --------
    Symbol.tojson : Used to save symbol into json string.
    """
    if not isinstance(json_str, string_types):
        raise TypeError('fname need to be string')
    handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateFromJSON(c_str(json_str), ctypes.byref(handle)))
    return Symbol(handle)


# Initialize the atomic symbol in startups
_init_symbol_module(Symbol, "mxnet")

# pylint: disable=no-member
# pylint: disable=redefined-builtin
def pow(base, exp):
    """ Raise base to an exp.

    Parameters
    ---------
    base: Symbol or Number
    exp: Symbol or Number

    Returns
    -------
    result: Symbol or Number
    """
    if isinstance(base, Symbol) and isinstance(exp, Symbol):
        return _internal._Power(base, exp)
    if isinstance(base, Symbol) and isinstance(exp, Number):
        return _internal._PowerScalar(base, scalar=exp)
    if isinstance(base, Number) and isinstance(exp, Symbol):
        return _internal._RPowerScalar(exp, scalar=base)
    if isinstance(base, Number) and isinstance(exp, Number):
        return base**exp
    else:
        raise TypeError('types (%s, %s) not supported' % (str(type(base)), str(type(exp))))


# pylint: disable=no-member
# pylint: disable=redefined-builtin
def maximum(left, right):
    """ maximum left and right

    Parameters
    ---------
    left: Symbol or Number
    right: Symbol or Number

    Returns
    -------
    result: Symbol or Number
    """
    if isinstance(left, Symbol) and isinstance(right, Symbol):
        return _internal._Maximum(left, right)
    if isinstance(left, Symbol) and isinstance(right, Number):
        return _internal._MaximumScalar(left, scalar=right)
    if isinstance(left, Number) and isinstance(right, Symbol):
        return _internal._MaximumScalar(right, scalar=left)
    if isinstance(left, Number) and isinstance(right, Number):
        return left if left > right else right
    else:
        raise TypeError('types (%s, %s) not supported' % (str(type(left)), str(type(right))))


# pylint: disable=no-member
# pylint: disable=redefined-builtin
def minimum(left, right):
    """ minimum left and right

    Parameters
    ---------
    left: Symbol or Number
    right: Symbol or Number

    Returns
    -------
    result: Symbol or Number
    """
    if isinstance(left, Symbol) and isinstance(right, Symbol):
        return _internal._Minimum(left, right)
    if isinstance(left, Symbol) and isinstance(right, Number):
        return _internal._MinimumScalar(left, scalar=right)
    if isinstance(left, Number) and isinstance(right, Symbol):
        return _internal._MinimumScalar(right, scalar=left)
    if isinstance(left, Number) and isinstance(right, Number):
        return left if left > right else right
    else:
        raise TypeError('types (%s, %s) not supported' % (str(type(left)), str(type(right))))


# pylint: disable=no-member
# pylint: disable=redefined-builtin
def hypot(left, right):
    """ minimum left and right

    Parameters
    ---------
    left: Symbol or Number
    right: Symbol or Number

    Returns
    -------
    result: Symbol or Number
    """
    if isinstance(left, Symbol) and isinstance(right, Symbol):
        return _internal._Hypot(left, right)
    if isinstance(left, Symbol) and isinstance(right, Number):
        return _internal._HypotScalar(left, scalar=right)
    if isinstance(left, Number) and isinstance(right, Symbol):
        return _internal._HypotScalar(right, scalar=left)
    if isinstance(left, Number) and isinstance(right, Number):
        return _numpy.hypot(left, right)
    else:
        raise TypeError('types (%s, %s) not supported' % (str(type(left)), str(type(right))))


def zeros(shape, dtype=None, **kwargs):
    """Create a Tensor filled with zeros, similar to numpy.zeros
        See Also https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html.

    Parameters
    ----------
    shape :  int or sequence of ints
        Shape of the new array.
    dtype : str or numpy.dtype, optional
        The value type of the inner value, default to np.float32

    Returns
    -------
    out : Symbol
        The created Symbol
    """
    if dtype is None:
        dtype = _numpy.float32
    return _internal._zeros(shape=shape, dtype=dtype, **kwargs)


def ones(shape, dtype=None, **kwargs):
    """Create a Tensor filled with ones, similar to numpy.ones
        See Also https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html.

    Parameters
    ----------
    shape :  int or sequence of ints
        Shape of the new array.
    dtype : str or numpy.dtype, optional
        The value type of the inner value, default to np.float32

    Returns
    -------
    out : Symbol
        The created Symbol
    """
    if dtype is None:
        dtype = _numpy.float32
    return _internal._ones(shape=shape, dtype=dtype, **kwargs)


def arange(start, stop=None, step=1.0, repeat=1, name=None, dtype=None):
    """Simlar function in the MXNet ndarray as numpy.arange
        See Also https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html.

    Parameters
    ----------
    start : number
        Start of interval. The interval includes this value. The default start value is 0.
    stop : number, optional
        End of interval. The interval does not include this value.
    step : number, optional
        Spacing between values
    repeat : int, optional
        "The repeating time of all elements.
        E.g repeat=3, the element a will be repeated three times --> a, a, a.
    dtype : str or numpy.dtype, optional
        The value type of the inner value, default to np.float32

    Returns
    -------
    out : Symbol
        The created Symbol
    """
    if dtype is None:
        dtype = _numpy.float32
    return _internal._arange(start=start, stop=stop, step=step, repeat=repeat,
                             name=name, dtype=dtype)
