# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
# pylint: disable=import-error, no-name-in-module
"""Symbolic configuration API of MXNet."""
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
from .context import Context, cpu
from .ndarray import NDArray, zeros as _nd_zeros, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .executor import Executor
from . import _symbol_internal as _internal
from .attribute import AttrScope

# Use different version of SymbolBase
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

_GRAD_REQ_MAP = {'null': 0, 'write': 1, 'add': 3}

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
        """Returns a generator object of symbol.

        One can loop through the returned object list to get outputs.

        Example usage:
        ----------
        >>> a = mx.sym.Variable('a')
        >>> b = mx.sym.Variable('b')
        >>> c = a+b
        >>> d = mx.sym.Variable('d')
        >>> e = d+c
        >>> out = e.get_children()
        >>> out
        <Symbol Grouped>
        >>> for i in out:
        ...     i
        ...
        <Symbol d>
        <Symbol _plus0>
        """
        return (self[i] for i in self.list_outputs())

    def __add__(self, other):
        """x.__add__(y) <=> x+y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_add` instead. """
        if isinstance(other, Symbol):
            return _internal._Plus(self, other)
        if isinstance(other, Number):
            return _internal._PlusScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """x.__sub__(y) <=> x-y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_sub` instead. """
        if isinstance(other, Symbol):
            return _internal._Minus(self, other)
        if isinstance(other, Number):
            return _internal._MinusScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rsub__(self, other):
        """x.__rsub__(y) <=> y-x

        Only `NDArray` is supported for now.

        Example usage:
        ----------
        >>> x = mx.nd.ones((2,3))*3
        >>> y = mx.nd.ones((2,3))
        >>> x.__rsub__(y).asnumpy()
        array([[-2., -2., -2.],
               [-2., -2., -2.]], dtype=float32)
        """
        if isinstance(other, Number):
            return _internal._RMinusScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __mul__(self, other):
        """x.__mul__(y) <=> x*y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_mul` instead. """
        if isinstance(other, Symbol):
            return _internal._Mul(self, other)
        if isinstance(other, Number):
            return _internal._MulScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """x.__div__(y) <=> x/y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_div` instead. """
        if isinstance(other, Symbol):
            return _internal._Div(self, other)
        if isinstance(other, Number):
            return _internal._DivScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rdiv__(self, other):
        """x.__rdiv__(y) <=> y/x

        Only `NDArray` is supported for now.

        Example usage:
        ----------
        >>> x = mx.nd.ones((2,3))*3
        >>> y = mx.nd.ones((2,3))
        >>> x.__rdiv__(y).asnumpy()
        array([[ 0.33333334,  0.33333334,  0.33333334],
               [ 0.33333334,  0.33333334,  0.33333334]], dtype=float32)
        """
        if isinstance(other, Number):
            return _internal._RDivScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __pow__(self, other):
        """x.__pow__(y) <=> x**y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_pow` instead. """
        if isinstance(other, Symbol):
            return _internal._Power(self, other)
        if isinstance(other, Number):
            return _internal._PowerScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __neg__(self):
        """x.__neg__() <=> -x

        Numerical negative, element-wise.

        Example usage:
        ----------
        >>> a = mx.sym.Variable('a')
        >>> a
        <Symbol a>
        >>> -a
        <Symbol _mulscalar0>
        >>> a_neg = a.__neg__()
        >>> c = a_neg*b
        >>> ex = c.eval(ctx=mx.cpu(), a=mx.nd.ones([2,3]), b=mx.nd.ones([2,3]))
        >>> ex[0].asnumpy()
        array([[-1., -1., -1.],
               [-1., -1., -1.]], dtype=float32)
        """
        return self.__mul__(-1.0)

    def __copy__(self):
        return self.__deepcopy__(None)

    def __deepcopy__(self, _):
        """Returns a deep copy of the input object.

        This function returns a deep copy of the input object including the current state
        of all its parameters such as weights, biases, etc.

        Any changes made to the deep copy do not reflect in the original object.

        Example usage:
        ----------
        >>> import copy
        >>> data = mx.sym.Variable('data')
        >>> data_1 = copy.deepcopy(data)
        >>> data_1 = 2*data
        >>> data_1.tojson()
        >>> data_1 is data    # Data got modified
        False
        """
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolCopy(self.handle,
                                     ctypes.byref(handle)))
        return Symbol(handle)

    def __eq__(self, other):
        """x.__eq__(y) <=> x==y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_equal` instead. """
        if isinstance(other, Symbol):
            return _internal._equal(self, other)
        if isinstance(other, numeric_types):
            return _internal._equal_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __ne__(self, other):
        """x.__ne__(y) <=> x!=y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_not_equal` instead. """
        if isinstance(other, Symbol):
            return _internal._not_equal(self, other)
        if isinstance(other, numeric_types):
            return _internal._not_equal_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __gt__(self, other):
        """x.__gt__(y) <=> x>y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_greater` instead. """
        if isinstance(other, Symbol):
            return _internal._greater(self, other)
        if isinstance(other, numeric_types):
            return _internal._greater_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __ge__(self, other):
        """x.__ge__(y) <=> x>=y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_greater_equal` instead. """
        if isinstance(other, Symbol):
            return _internal._greater_equal(self, other)
        if isinstance(other, numeric_types):
            return _internal._greater_equal_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __lt__(self, other):
        """x.__lt__(y) <=> x<y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_lesser` instead. """
        if isinstance(other, Symbol):
            return _internal._lesser(self, other)
        if isinstance(other, numeric_types):
            return _internal._lesser_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __le__(self, other):
        """x.__le__(y) <=> x<=y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_lesser_equal` instead. """
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
        """Composes symbol using inputs.

        x.__call__(y, z) <=> x(y,z)

        This function internally calls `_compose` to compose the symbol and
        returns the composed symbol.

        Example usage:
        ----------
        >>> data = mx.symbol.Variable('data')
        >>> net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
        >>> net2 = mx.symbol.FullyConnected(name='fc3', num_hidden=10)
        >>> composed = net2(fc3_data=net1, name='composed')
        >>> composed
        <Symbol composed>
        >>> called = net2.__call__(fc3_data=net1, name='composed')
        >>> called
        <Symbol composed>

        Parameters
        ----------
        args:
            Positional arguments.

        kwargs:
            Keyword arguments.

        Returns
        -------
            The resulting symbol.
        """
        s = self.__copy__()
        s._compose(*args, **kwargs)
        return s

    def _compose(self, *args, **kwargs):
        """Composes symbol using inputs.

        x._compose(y, z) <=> x(y,z)

        This function mutates the current symbol.

        Example usage:
        ----------
        >>> data = mx.symbol.Variable('data')
        >>> net1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=10)
        >>> net2 = mx.symbol.FullyConnected(name='fc3', num_hidden=10)
        >>> net2
        <Symbol fc3>
        >>> net2._compose(fc3_data=net1, name='composed')
        >>> net2
        <Symbol composed>

        Parameters
        ----------
        args:
            Positional arguments.

        kwargs:
            Keyword arguments.

        Returns
        -------
            The resulting symbol.
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
        """x.__getitem__(i) <=> x[i]

        Returns a sliced view of the input symbol.

        Example usage:
        ----------
        >>> a = mx.sym.var('a')
        >>> a.__getitem__(0)
        <Symbol a>
        >>> a[0]
        <Symbol a>

        Parameters
        ----------
        index : int or str
            Indexing key

        """
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
            The name of this symbol, returns ``None`` for grouped symbol.
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
        """Returns the attribute string for corresponding input key from the symbol.

        This function only works for non-grouped symbols.

        Example usage:
        ----------
        >>> data = mx.sym.Variable('data', attr={'mood': 'angry'})
        >>> data.attr('mood')
        'angry'

        Parameters
        ----------
        key : str
            The key corresponding to the desired attribute.

        Returns
        -------
        value : str
            The desired attribute value, returns ``None`` if the attribute does not exist.
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
        """Gets all attributes from the symbol.

        Example usage:
        ----------
        >>> data = mx.sym.Variable('data', attr={'mood': 'angry'})
        >>> data.list_attr()
        {'mood': 'angry'}

        Returns
        -------
        ret : Dict of str to str
            A dictionary mapping attribute keys to values.
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
        """Recursively gets all attributes from the symbol and its children.

        Example usage:
        ----------
        >>> a = mx.sym.Variable('a', attr={'a1':'a2'})
        >>> b = mx.sym.Variable('b', attr={'b1':'b2'})
        >>> c = a+b
        >>> c.attr_dict()
        {'a': {'a1': 'a2'}, 'b': {'b1': 'b2'}}

        Returns
        -------
        ret : Dict of str to dict
            There is a key in the returned dict for every child with non-empty attribute set.
            For each symbol, the name of the symbol is its key in the dict
            and the correspond value is that symbol's attribute list (itself a dictionary).
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
        """Sets an attribute of the symbol.

        For example. A._set_attr(foo="bar") adds the mapping ``"{foo: bar}"``
        to the symbol's attribute dictionary.

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
        """Gets a new grouped symbol `sgroup`. The output of `sgroup` is a list of
        outputs of all of the internal nodes.

        Consider the following code:

        Example usage:
        ----------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> d = c.get_internals()
        >>> d
        <Symbol Grouped>
        >>> d.list_outputs()
        ['a', 'b', '_plus4_output']

        Returns
        -------
        sgroup : Symbol
            A symbol group containing all internal and leaf nodes of the computation graph
            used to compute the symbol.
        """
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolGetInternals(
            self.handle, ctypes.byref(handle)))
        return Symbol(handle=handle)

    def get_children(self):
        """Gets a new grouped symbol whose output contains
        inputs to output nodes of the original symbol.

        Example usage:
        ----------
        >>> x = mx.sym.Variable('x')
        >>> y = mx.sym.Variable('y')
        >>> z = mx.sym.Variable('z')
        >>> a = y+z
        >>> b = x+a
        >>> b.get_children()
        <Symbol Grouped>
        >>> b.get_children().list_outputs()
        ['x', '_plus10_output']
        >>> b.get_children().get_children().list_outputs()
        ['y', 'z']

        Returns
        -------
        sgroup : Symbol or None
            The children of the head node. If the symbol has no
            inputs then ``None`` will be returned.
        """
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolGetChildren(
            self.handle, ctypes.byref(handle)))
        ret = Symbol(handle=handle)
        if len(ret.list_outputs()) == 0:
            return None
        return ret

    def list_arguments(self):
        """Lists all the arguments in the symbol.

        Example usage:
        ----------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> c.list_arguments
        ['a', 'b']

        Returns
        -------
        args : list of string
            List containing the names of all the arguments required to compute the symbol.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListArguments(
            self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [py_str(sarr[i]) for i in range(size.value)]

    def list_outputs(self):
        """Lists all the outputs in the symbol.

        Example usage:
        ----------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> c.list_outputs()
        ['_plus12_output']

        Returns
        -------
        list of str
            List of all the outputs.
            For most symbols, this list contains only the name of this symbol.
            For symbol groups, this is a list with the names of all symbols
            in the group.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListOutputs(
            self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [py_str(sarr[i]) for i in range(size.value)]

    def list_auxiliary_states(self):
        """Lists all the auxiliary states in the symbol.

        Example usage:
        ----------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> c.list_auxiliary_states()
        []

        Example of auxiliary states in `BatchNorm`.

        >>> data = mx.symbol.Variable('data')
        >>> weight = mx.sym.Variable(name='fc1_weight')
        >>> fc1  = mx.symbol.FullyConnected(data = data, weight=weight, name='fc1', num_hidden=128)
        >>> fc2 = mx.symbol.BatchNorm(fc1, name='batchnorm0')
        >>> fc2.list_auxiliary_states()
        ['batchnorm0_moving_mean', 'batchnorm0_moving_var']

        Returns
        -------
        aux_states : list of string
            List of the auxiliary states in input symbol.

        Notes
        -----
        Auxiliary states are special states of symbols that do not correspond to an argument,
        and are not updated by gradient descent. Common examples of auxiliary states
        include the `moving_mean` and `moving_variance` in `BatchNorm`.
        Most operators do not have auxiliary states.
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.MXSymbolListAuxiliaryStates(
            self.handle, ctypes.byref(size), ctypes.byref(sarr)))
        return [py_str(sarr[i]) for i in range(size.value)]

    def infer_type(self, *args, **kwargs):
        """Infers the type of all arguments and all outputs, given the known types
        for some arguments.

        This function takes the known types of some arguments in either positional way
        or keyword argument way as input. It returns a tuple of `None` values
        if there is not enough information to deduce the missing types.

        Inconsistencies in the known types will cause an error to be raised.

        Example usage:
        ----------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> arg_types, out_types, aux_types = c.infer_type(a='float32')
        >>> arg_types
        [<type 'numpy.float32'>, <type 'numpy.float32'>]
        >>> out_types
        [<type 'numpy.float32'>]
        >>> aux_types
        []

        Parameters
        ----------
        *args :
            Type of known arguments in a positional way.
            Unknown type can be marked as None.

        **kwargs :
            Keyword arguments of known types.

        Returns
        -------
        arg_types : list of numpy.dtype or None
            List of argument types.
            The order is same as the order of list_arguments().
        out_types : list of numpy.dtype or None
            List of output types.
            The order is same as the order of list_outputs().
        aux_types : list of numpy.dtype or None
            List of auxiliary state types.
            The order is same as the order of list_auxiliary_states().
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
        """Infers the shapes of all arguments and all outputs given the known shapes of
        some arguments.

        This function takes the known shapes of some arguments in either positional way
        or keyword argument way as input. It returns a tuple of `None` values
        if there is not enough information to deduce the missing shapes.

        Example usage:
        ----------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> arg_shapes, out_shapes, aux_shapes = c.infer_shape(a=(3,3))
        >>> arg_shapes
        [(3L, 3L), (3L, 3L)]
        >>> out_shapes
        [(3L, 3L)]
        >>> aux_shapes
        []
        >>> c.infer_shape(a=(0,3)) # 0s in shape means unknown dimensions. So, returns None.
        (None, None, None)

        Inconsistencies in the known shapes will cause an error to be raised.
        See the following example:

        >>> data = mx.sym.Variable('data')
        >>> out = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=1000)
        >>> out = mx.sym.Activation(data=out, act_type='relu')
        >>> out = mx.sym.FullyConnected(data=out, name='fc2', num_hidden=10)
        >>> weight_shape= (1, 100)
        >>> data_shape = (100, 100)
        >>> out.infer_shape(data=data_shape, fc1_weight=weight_shape)
        Error in operator fc1: Shape inconsistent, Provided=(1,100), inferred shape=(1000,100)

        Parameters
        ----------
        *args :
            Shape of arguments in a positional way.
            Unknown shape can be marked as None.

        **kwargs :
            Keyword arguments of the known shapes.

        Returns
        -------
        arg_shapes : list of tuple or None
            List of argument shapes.
            The order is same as the order of list_arguments().
        out_shapes : list of tuple or None
            List of output shapes.
            The order is same as the order of list_outputs().
        aux_shapes : list of tuple or None
            List of auxiliary state shapes.
            The order is same as the order of list_auxiliary_states().
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
        """Infers the shape partially.

        This functions works the same way as `infer_shape`,
        except that this function can return partial results.

        In the following example, information about fc2 is not available. So, `infer_shape`
        will return a tuple of `None` values but `infer_shape_partial` will return partial values.

        Example usage:
        ----------
        >>> data = mx.sym.Variable('data')
        >>> prev = mx.sym.Variable('prev')
        >>> fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
        >>> fc2  = mx.sym.FullyConnected(data=prev, name='fc2', num_hidden=128)
        >>> out  = mx.sym.Activation(data=mx.sym.elemwise_add(fc1, fc2), act_type='relu')
        >>> out.list_arguments()
        ['data', 'fc1_weight', 'fc1_bias', 'prev', 'fc2_weight', 'fc2_bias']
        >>> out.infer_shape(data=(10,64))
        (None, None, None)
        >>> out.infer_shape_partial(data=(10,64))
        ([(10L, 64L), (128L, 64L), (128L,), (), (), ()], [(10L, 128L)], [])
        >>> # infers shape if you give information about fc2
        >>> out.infer_shape(data=(10,64), prev=(10,128))
        ([(10L, 64L), (128L, 64L), (128L,), (10L, 128L), (128L, 128L), (128L,)], [(10L, 128L)], [])

        Parameters
        ----------
        *args :
            Shape of arguments in a positional way.
            Unknown shape can be marked as None

        **kwargs :
            Keyword arguments of known shapes.

        Returns
        -------
        arg_shapes : list of tuple or None
            List of argument shapes.
            The order is same as the order of list_arguments().
        out_shapes : list of tuple or None
            List of output shapes.
            The order is same as the order of list_outputs().
        aux_shapes : list of tuple or None
            List of auxiliary state shapes.
            The order is same as the order of list_auxiliary_states().
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
                        raise TypeError('Arguments must be shapes (tuple)')
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
        """Gets a debug string.

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
        """Saves symbol to a file.

        You can also use pickle to do the job if you only work on python.
        The advantage of `load`/`save` functions is that the file contents are language agnostic.
        This means the model saved by one language binding can be loaded by a different
        language binding of `MXNet`.
        You also get the benefit of being able to directly load/save from cloud storage(S3, HDFS).

        Parameters
        ----------
        fname : str
            The name of the file.

            - "s3://my-bucket/path/my-s3-symbol"
            - "hdfs://my-bucket/path/my-hdfs-symbol"
            - "/path-to/my-local-symbol"

        See Also
        --------
        symbol.load : Used to load symbol from file.
        """
        if not isinstance(fname, string_types):
            raise TypeError('fname need to be string')
        check_call(_LIB.MXSymbolSaveToFile(self.handle, c_str(fname)))

    def tojson(self):
        """Saves symbol to a JSON string.

        See Also
        --------
        symbol.load_json : Used to load symbol from JSON string.
        """
        json_str = ctypes.c_char_p()
        check_call(_LIB.MXSymbolSaveToJSON(self.handle, ctypes.byref(json_str)))
        return py_str(json_str.value)

    @staticmethod
    def _get_ndarray_inputs(arg_key, args, arg_names, allow_missing):
        """Helper function to get NDArray lists handles from various inputs.

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
                raise ValueError('Length of %s does not match the number of arguments' % arg_key)
            for narr in args:
                if not isinstance(narr, NDArray):
                    raise TypeError('Only accept list of NDArrays or dict of str to NDArray')
                arg_handles.append(narr.handle)
            arg_arrays = args
        elif isinstance(args, dict):
            for name in arg_names:
                if name in args:
                    narr = args[name]
                    if not isinstance(narr, NDArray):
                        raise TypeError('Only accept list of NDArrays or dict of str to NDArray')
                    arg_handles.append(narr.handle)
                    arg_arrays.append(narr)
                else:
                    if allow_missing:
                        arg_handles.append(None)
                        arg_arrays.append(None)
                    else:
                        raise ValueError('key `%s` is missing in `%s`' % (name, arg_key))
        else:
            raise TypeError('Only accept list of NDArrays or dict of str to NDArray')
        return c_array(NDArrayHandle, arg_handles), arg_arrays

    def simple_bind(self, ctx,
                    grad_req='write',
                    type_dict=None,
                    group2ctx=None,
                    **kwargs):
        """Binds current symbol to get an executor, allocate all the arguments needed.

        This function simplifies the binding procedure. You need to specify only input data shapes.
        Before binding the executor, the function allocates arguments and auxiliary states
        that were not explicitly specified. Allows specifying data types.

        Example usage:
        ----------
        >>> x = mx.sym.Variable('x')
        >>> y = mx.sym.FullyConnected(x, num_hidden=4)
        >>> exe = y.simple_bind(mx.cpu(), x=(5,4), grad_req=[])
        >>> exe.forward()
        [<NDArray 5x4 @cpu(0)>]
        >>> exe.outputs[0].asnumpy()
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)
        >>> exe.arg_arrays
        [<NDArray 5x4 @cpu(0)>, <NDArray 4x4 @cpu(0)>, <NDArray 4 @cpu(0)>]
        >>> exe.grad_arrays
        [<NDArray 5x4 @cpu(0)>, <NDArray 4x4 @cpu(0)>, <NDArray 4 @cpu(0)>]

        Parameters
        ----------
        ctx : Context
            The device context the generated executor to run on.

        grad_req: string
            {'write', 'add', 'null'}, or list of str or dict of str to str, optional
            To specify how we should update the gradient to the `args_grad`.

            - 'write' means every time gradient is written to specified `args_grad` NDArray.
            - 'add' means every time gradient is added to the specified NDArray.
            - 'null' means no action is taken, the gradient may not be calculated.

        type_dict  : Dict of str->numpy.dtype
            Input type dictionary, name->dtype

        group2ctx : Dict of string to mx.Context
            The dict mapping the `ctx_group` attribute to the context assignment.

        kwargs : Dict of str->shape
            Input shape dictionary, name->shape

        Returns
        -------
        executor : mxnet.Executor
            The generated executor
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
        """Binds the current symbol to an executor and returns it.

        We first declare the computation and then bind to the data to run.
        This function returns an executor which provides method `forward()` method for evaluation
        and a `outputs()` method to get all the results.

        Example usage:
        ----------
        >>> a = mx.sym.Variable('a')
        >>> b = mx.sym.Variable('b')
        >>> c = a + b
        <Symbol _plus1>
        >>> ex = c.bind(ctx=mx.cpu(), args={'a' : mx.nd.ones([2,3]), 'b' : mx.nd.ones([2,3])})
        >>> ex.forward()
        [<NDArray 2x3 @cpu(0)>]
        >>> ex.outputs[0].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        Parameters
        ----------
        ctx : Context
            The device context the generated executor to run on.

        args : list of NDArray or dict of str to NDArray
            Input arguments to the symbol.

            - If the input type is a list of `NDArray`, the order should be same as the order
              of `list_arguments()`.
            - If the input type is a dict of str to `NDArray`, then it maps the name of arguments
              to the corresponding `NDArray`.
            - In either case, all the arguments must be provided.

        args_grad : list of NDArray or dict of str to `NDArray`, optional
            When specified, `args_grad` provides NDArrays to hold
            the result of gradient value in backward.

            - If the input type is a list of `NDArray`, the order should be same as the order
              of `list_arguments()`.
            - If the input type is a dict of str to `NDArray`, then it maps the name of arguments
              to the corresponding NDArray.
            - When the type is a dict of str to `NDArray`, one only need to provide the dict
              for required argument gradient.
              Only the specified argument gradient will be calculated.

        grad_req : {'write', 'add', 'null'}, or list of str or dict of str to str, optional
            To specify how we should update the gradient to the `args_grad`.

            - 'write' means everytime gradient is write to specified `args_grad` `NDArray`.
            - 'add' means everytime gradient is add to the specified NDArray.
            - 'null' means no action is taken, the gradient may not be calculated.

        aux_states : list of `NDArray`, or dict of str to `NDArray`, optional
            Input auxiliary states to the symbol, only needed when the output of
            `list_auxiliary_states()` is not empty.

            - If the input type is a list of `NDArray`, the order should be same as the order
              of `list_auxiliary_states()`.
            - If the input type is a dict of str to `NDArray`, then it maps the name of
              `auxiliary_states` to the corresponding `NDArray`,
            - In either case, all the auxiliary states need to be provided.

        group2ctx : Dict of string to mx.Context
            The dict mapping the `ctx_group` attribute to the context assignment.

        shared_exec : mx.executor.Executor
            Executor to share memory with. This is intended for runtime reshaping, variable length
            sequences, etc. The returned executor shares state with `shared_exec`, and should not be
            used in parallel with it.

        Returns
        -------
        executor : Executor
            The generated executor

        Notes
        -----
        Auxiliary states are the special states of symbols that do not correspond
        to an argument, and do not have gradient but are still useful
        for the specific operations. Common examples of auxiliary states include
        the `moving_mean` and `moving_variance` states in `BatchNorm`.
        Most operators do not have auxiliary states and in those cases,
        this parameter can be safely ignored.

        One can give up gradient by using a dict in `args_grad` and only specify
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
        if isinstance(grad_req, string_types):
            if grad_req not in _GRAD_REQ_MAP:
                raise ValueError('grad_req must be in %s' % str(_GRAD_REQ_MAP))
            reqs_array = c_array(
                mx_uint,
                [mx_uint(_GRAD_REQ_MAP[grad_req])] * len(listed_arguments))
        elif isinstance(grad_req, list):
            reqs_array = c_array(mx_uint, [mx_uint(_GRAD_REQ_MAP[item]) for item in grad_req])
        elif isinstance(grad_req, dict):
            req_array = []
            for name in listed_arguments:
                if name in grad_req:
                    req_array.append(mx_uint(_GRAD_REQ_MAP[grad_req[name]]))
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

        .. note:: This function is currently not implemented.

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

    def eval(self, ctx=cpu(), **kwargs):
        """Evaluates a symbol given arguments.

        The `eval` method combines a call to `bind` (which returns an executor)
        with a call to `forward` (executor method).
        For the common use case, where you might repeatedly evaluate with same arguments,
        eval is slow.
        In that case, you should call `bind` once and then repeatedly call forward.
        This function allows simpler syntax for less cumbersome introspection.

        Example usage:
        ----------
        >>> a = mx.sym.Variable('a')
        >>> b = mx.sym.Variable('b')
        >>> c = a + b
        >>> ex = c.eval(ctx = mx.cpu(), a = mx.nd.ones([2,3]), b = mx.nd.ones([2,3]))
        >>> ex
        [<NDArray 2x3 @cpu(0)>]
        >>> ex[0].asnumpy()
        array([[ 2.,  2.,  2.],
               [ 2.,  2.,  2.]], dtype=float32)

        Parameters
        ----------
        ctx : Context
            The device context the generated executor to run on.

        kwargs : Keyword arguments of type `NDArray`
            Input arguments to the symbol. All the arguments must be provided.

        Returns
        ----------
        result :  a list of NDArrays corresponding to the values taken by each symbol when
        evaluated on given args. When called on a single symbol (not a group),
        the result will be a list with one element.
        """
        return self.bind(ctx, kwargs).forward()



def var(name, attr=None, shape=None, lr_mult=None, wd_mult=None, dtype=None, init=None, **kwargs):
    """Creates a symbolic variable with specified name.

    Example usage:
    ----------
    >>> data = mx.sym.Variable('data', attr={'a': 'b'})
    >>> data
    <Symbol data>

    Parameters
    ----------
    name : str
        Variable name.
    attr : Dict of strings
        Additional attributes to set on the variable. Format {string : string}.
    shape : tuple
        The shape of a variable. If specified, this will be used during the shape inference.
        If one has specified a different shape for this variable using
        a keyword argument when calling shape inference, this shape information will be ignored.
    lr_mult : float
        The learning rate multiplier for input variable.
    wd_mult : float
        Weight decay multiplier for input variable.
    dtype : str or numpy.dtype
        The dtype for input variable. If not specified, this value will be inferred.
    init : initializer (mxnet.init.*)
        Initializer for this variable to (optionally) override the default initializer.
    kwargs : Additional attribute variables
        Additional attributes must start and end with double underscores.

    Returns
    -------
    variable : Symbol
        A symbol corresponding to an input to the computation graph.
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
    for k, v in kwargs.items():
        if k.startswith('__') and k.endswith('__'):
            attr[k] = str(v)
        else:
            raise ValueError('Attribute name=%s is not supported.'
                             ' Additional attributes must start and end with double underscores,'
                             ' e.g, __yourattr__' % k)
    ret._set_attr(**attr)
    return ret

# for back compatibility
Variable = var

def Group(symbols):
    """Creates a symbol that contains a collection of other symbols, grouped together.

    Parameters
    ----------
    symbols : list
        List of symbols to be grouped.

    Returns
    -------
    sym : Symbol
        A group symbol.
     """
    ihandles = []
    for sym in symbols:
        if not isinstance(sym, Symbol):
            raise TypeError('Expected a list of symbols as input')
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
    You also get the benefit being able to directly load/save from cloud storage(S3, HDFS).

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
        A JSON string.

    Returns
    -------
    sym : Symbol
        The loaded symbol.

    See Also
    --------
    Symbol.tojson : Used to save symbol into json string.
    """
    if not isinstance(json_str, string_types):
        raise TypeError('fname required to be string')
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
    """Return a new symbol of given shape and type, filled with zeros.

    Parameters
    ----------
    shape :  int or sequence of ints
        Shape of the new array.
    dtype : str or numpy.dtype, optional
        The value type of the inner value, default to ``np.float32``.

    Returns
    -------
    out : Symbol
        The created Symbol.
    """
    if dtype is None:
        dtype = _numpy.float32
    return _internal._zeros(shape=shape, dtype=dtype, **kwargs)


def ones(shape, dtype=None, **kwargs):
    """Return a new symbol of given shape and type, filled with ones.

    Parameters
    ----------
    shape :  int or sequence of ints
        Shape of the new array.
    dtype : str or numpy.dtype, optional
        The value type of the inner value, default to ``np.float32``.

    Returns
    -------
    out : Symbol
        The created Symbol
    """
    if dtype is None:
        dtype = _numpy.float32
    return _internal._ones(shape=shape, dtype=dtype, **kwargs)


def arange(start, stop=None, step=1.0, repeat=1, name=None, dtype=None):
    """Return evenly spaced values within a given interval.

    Parameters
    ----------
    start : number
        Start of interval. The interval includes this value. The default start value is 0.
    stop : number, optional
        End of interval. The interval does not include this value.
    step : number, optional
        Spacing between values.
    repeat : int, optional
        "The repeating time of all elements.
        E.g repeat=3, the element a will be repeated three times --> a, a, a.
    dtype : str or numpy.dtype, optional
        The value type of the inner value, default to ``np.float32``.

    Returns
    -------
    out : Symbol
        The created Symbol
    """
    if dtype is None:
        dtype = _numpy.float32
    return _internal._arange(start=start, stop=stop, step=step, repeat=repeat,
                             name=name, dtype=dtype)
