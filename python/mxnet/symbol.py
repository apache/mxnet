# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
"""Symbolic configuration API of mxnet."""
from __future__ import absolute_import

import copy
import ctypes
from numbers import Number
import re
import sys
import numpy
from .base import _LIB
from .base import c_array, c_str, mx_uint, py_str, string_types, mx_real_t
from .base import NDArrayHandle, ExecutorHandle, SymbolHandle
from .base import check_call, ctypes2docstring
from .name import NameManager
from .attribute import AttrScope
from .context import Context
from .ndarray import NDArray, zeros, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .executor import Executor
from .symbol_doc import SymbolDoc
from . import _symbol_internal as _internal

class Symbol(object):
    """Symbol is symbolic graph of the mxnet."""

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        self.handle = handle

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

    def __del__(self):
        check_call(_LIB.MXSymbolFree(self.handle))

    def __copy__(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, _):
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolCopy(self.handle,
                                     ctypes.byref(handle)))
        return Symbol(handle)

    def __getstate__(self):
        this = self.__dict__.copy()
        handle = this['handle']
        if handle is not None:
            this['handle'] = self.tojson()
        return this

    def __setstate__(self, state):
        handle = state['handle']
        if handle is not None:
            json_str = handle
            handle = SymbolHandle()
            check_call(_LIB.MXSymbolCreateFromJSON(c_str(json_str), ctypes.byref(handle)))
            state['handle'] = handle
        self.__dict__.update(state)

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
        s = copy.deepcopy(self)
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

        Parameters
        ----------
        recursive : bool
            Default `False`. When `recursive` is `True`, list recursively all the
            attributes in the descendents. The attribute names are pre-pended with
            the symbol names to avoid conflicts. If `False`, then only attributes
            that belongs to this symbol is returned, and the attribute names will
            **not** be pre-pended with the symbol name.
        """
        size = mx_uint()
        pairs = ctypes.POINTER(ctypes.c_char_p)()
        f_handle = _LIB.MXSymbolListAttr if recursive else _LIB.MXSymbolListAttrShallow
        check_call(f_handle(self.handle, ctypes.byref(size), ctypes.byref(pairs)))
        return {py_str(pairs[i*2]): py_str(pairs[i*2+1]) for i in range(size.value)}

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
        """Get a new grouped symbol whose output contains all the internal outputs of this symbol.

        Returns
        -------
        sgroup : Symbol
            The internal of the symbol.
        """
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolGetInternals(
            self.handle, ctypes.byref(handle)))
        return Symbol(handle=handle)

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
                    s = numpy.dtype(s).type
                    if s not in _DTYPE_NP_TO_MX:
                        raise TypeError('Argument need to be one of '+str(_DTYPE_NP_TO_MX))
                    sdata.append(_DTYPE_NP_TO_MX[s])
                else:
                    sdata.append(-1)
        else:
            keys = []
            for k, v in kwargs.items():
                v = numpy.dtype(v).type
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
        return self._infer_shape_impl(False, *args, **kwargs)

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
                        raise ValueError('Must specify all the arguments in %s' % arg_key)
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
            type_dict = {k: mx_real_t for k in self.list_arguments()}
        arg_shapes, _, aux_shapes = self.infer_shape(**kwargs)
        arg_types, _, aux_types = self.infer_type(**type_dict)

        if arg_shapes is None or arg_types is None:
            raise ValueError("Input node is not complete")

        if group2ctx is not None:
            attr_dict = {
                k : group2ctx.get(v, ctx)
                for k, v in self.list_attr(recursive=True).items()
                if k.endswith('ctx_group')
            } if group2ctx is not None else {}
            arg_ctx = [attr_dict.get(name + '_ctx_group', ctx)
                       for name in self.list_arguments()]
            aux_ctx = [attr_dict.get(name + '_ctx_group', ctx)
                       for name in self.list_auxiliary_states()]
        else:
            arg_ctx = [ctx] * len(arg_shapes)
            aux_ctx = [ctx] * len(aux_shapes)

        # alloc space
        arg_ndarrays = [
            zeros(shape, dev, dtype=dtype)
            for dtype, dev, shape in zip(arg_types, arg_ctx, arg_shapes)]
        if grad_req != 'null':
            grad_ndarrays = {}
            for name, shape, dev, dtype in zip(
                    self.list_arguments(), arg_shapes, arg_ctx, arg_types):
                if not isinstance(grad_req, dict) or grad_req[name] != 'null':
                    grad_ndarrays[name] = zeros(shape, dev, dtype=dtype)
        else:
            grad_ndarrays = None

        aux_ndarrays = [zeros(shape, dev, dtype=dtype)
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
        executor : mxnet.Executor
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


def Variable(name, attr=None, shape=None):
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
    if shape is not None:
        attr = {} if attr is None else attr
        attr['__shape__'] = str(shape)
    if attr:
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


def _make_atomic_symbol_function(handle):
    """Create an atomic symbol function by handle and funciton name."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    key_var_num_args = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
        handle, ctypes.byref(name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(key_var_num_args),
        ctypes.byref(ret_type)))
    param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)
    key_var_num_args = py_str(key_var_num_args.value)
    func_name = py_str(name.value)
    desc = py_str(desc.value)
    if key_var_num_args:
        desc += '\nThis function support variable length of positional input.'
    doc_str = ('%s\n\n' +
               '%s\n' +
               'name : string, optional.\n' +
               '    Name of the resulting symbol.\n\n' +
               'Returns\n' +
               '-------\n' +
               'symbol: Symbol\n' +
               '    The result symbol.')
    doc_str = doc_str % (desc, param_str)
    extra_doc = "\n" + '\n'.join([x.__doc__ for x in type.__subclasses__(SymbolDoc)
                                  if x.__name__ == '%sDoc' % func_name])
    doc_str += re.sub(re.compile("    "), "", extra_doc)

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
        attr = kwargs.pop('attr', None)

        if key_var_num_args and key_var_num_args not in kwargs:
            param_keys.append(c_str(key_var_num_args))
            param_vals.append(c_str(str(len(args))))

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
            handle,
            mx_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(sym_handle)))

        if len(args) != 0 and len(symbol_kwargs) != 0:
            raise TypeError(
                '%s can only accept input'
                'Symbols either as positional or keyword arguments, not both' % func_name)
        if key_var_num_args and len(symbol_kwargs) != 0:
            raise ValueError('This function supports variable length of Symbol arguments.\n' +
                             'Please pass all the input Symbols via positional arguments' +
                             ' instead of keyword arguments.')
        s = Symbol(sym_handle)
        attr = AttrScope.current.get(attr)
        if attr:
            s._set_attr(**attr)
        hint = func_name.lower()
        name = NameManager.current.get(name, hint)
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
    module_internal = sys.modules["mxnet._symbol_internal"]
    for i in range(size.value):
        hdl = SymbolHandle(plist[i])
        function = _make_atomic_symbol_function(hdl)
        if function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
        else:
            setattr(module_obj, function.__name__, function)

# Initialize the atomic symbo in startups
_init_symbol_module()


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
