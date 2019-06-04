# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
# pylint: disable=import-error, no-name-in-module
"""Symbolic configuration API of MXNet."""
from __future__ import absolute_import as _abs
try:
    from __builtin__ import slice as py_slice
except ImportError:
    from builtins import slice as py_slice

from array import array
import ctypes
import warnings
from numbers import Number

import numpy as _numpy  # pylint: disable=relative-import

from ..attribute import AttrScope
from ..base import _LIB, numeric_types, c_array, c_array_buf, c_str, c_str_array, c_handle_array
from ..base import mx_uint, py_str, string_types, integer_types, mx_int
from ..base import NDArrayHandle, ExecutorHandle, SymbolHandle
from ..base import check_call, MXNetError, NotImplementedForSymbol
from ..context import Context, current_context
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, _GRAD_REQ_MAP
from ..ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from ..ndarray import _ndarray_cls
from ..executor import Executor
from . import _internal
from . import op
from ._internal import SymbolBase, _set_symbol_class
from ..util import is_np_shape

__all__ = ["Symbol", "var", "Variable", "Group", "load", "load_json",
           "pow", "power", "maximum", "minimum", "hypot", "eye", "zeros",
           "ones", "full", "arange", "linspace", "histogram", "split_v2"]


class Symbol(SymbolBase):
    """Symbol is symbolic graph of the mxnet."""
    # disable dictionary storage, also do not have parent type.
    # pylint: disable=no-member
    __slots__ = []

    # Make numpy functions return Symbol instead of numpy object array
    __array_priority__ = 1000.0

    def as_np_ndarray(self):
        """Convert mx.sym.Symbol to mx.sym.np._Symbol."""
        from .numpy import _Symbol
        hdl = SymbolHandle()
        check_call(_LIB.MXShallowCopySymbol(self.handle, ctypes.byref(hdl)))
        return _Symbol(hdl)

    def as_classic_ndarray(self):
        """Returns self. For the convenience of conversion between legacy and np symbols."""
        return self

    def __repr__(self):
        """Gets a string representation of the symbol."""
        name = self.name
        if name is None:
            name = ', '.join([i.name for i in self])
            return '<%s group [%s]>' % (self.__class__.__name__, name)
        else:
            return '<%s %s>' % (self.__class__.__name__, name)

    def __iter__(self):
        """Returns a generator object of symbol.

        One can loop through the returned object list to get outputs.

        Example
        -------
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
        return (self[i] for i in range(len(self)))

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

    def __bool__(self):
        raise NotImplementedForSymbol(self.__bool__, 'bool')

    __nonzero__ = __bool__

    def __iadd__(self, other):
        raise NotImplementedForSymbol(self.__iadd__, '+=', other, 1)

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

    def __isub__(self, other):
        raise NotImplementedForSymbol(self.__isub__, '-=', other)

    def __rsub__(self, other):
        """x.__rsub__(y) <=> y-x

        Only `NDArray` is supported for now.

        Example
        -------
        >>> x = mx.nd.ones((2,3))*3
        >>> y = mx.nd.ones((2,3))
        >>> x.__rsub__(y).asnumpy()
        array([[-2., -2., -2.],
               [-2., -2., -2.]], dtype=float32)
        """
        if isinstance(other, Symbol):
            return other.__sub__(self)
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

    def __imul__(self, other):
        raise NotImplementedForSymbol(self.__imul__, '*=', other)

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

        Example
        -------
        >>> x = mx.nd.ones((2,3))*3
        >>> y = mx.nd.ones((2,3))
        >>> x.__rdiv__(y).asnumpy()
        array([[ 0.33333334,  0.33333334,  0.33333334],
               [ 0.33333334,  0.33333334,  0.33333334]], dtype=float32)
        """
        if isinstance(other, Symbol):
            return other.__truediv__(self)
        if isinstance(other, Number):
            return _internal._RDivScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __mod__(self, other):
        """x.__mod__(y) <=> x%y

        Scalar input is supported.
        Broadcasting is not supported. Use `broadcast_mod` instead. """
        if isinstance(other, Symbol):
            return _internal._Mod(self, other)
        if isinstance(other, Number):
            return _internal._ModScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __rmod__(self, other):
        """x.__rmod__(y) <=> y%x

        Only `NDArray` is supported for now.

        Example
        -------
        >>> x = mx.nd.ones((2,3))*3
        >>> y = mx.nd.ones((2,3))
        >>> x.__rmod__(y).asnumpy()
        array([[ 1.,  1.,  1.,
               [ 1.,  1.,  1., dtype=float32)
        """
        if isinstance(other, Symbol):
            return other.__mod__(self)
        if isinstance(other, Number):
            return _internal._RModScalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __idiv__(self, other):
        raise NotImplementedForSymbol(self.__idiv__, '/=', other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __itruediv__(self, other):
        raise NotImplementedForSymbol(self.__itruediv__, '/=', other)

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

    def __rpow__(self, other):
        """x.__rpow__(y) <=> y ** x"""
        if isinstance(other, Symbol):
            return other.__pow__(self)
        elif isinstance(other, Number):
            return _internal._rpower_scalar(self, scalar=other)
        else:
            raise TypeError('type %s not supported' % str(type(other)))

    def __neg__(self):
        """x.__neg__() <=> -x

        Numerical negative, element-wise.

        Example
        -------
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

        Example
        -------
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

        Example
        -------
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

        Example
        -------
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
            keys = c_str_array(kwargs.keys())
            args = c_handle_array(kwargs.values())
        else:
            keys = None
            args = c_handle_array(args)
        check_call(_LIB.MXSymbolCompose(
            self.handle, name, num_args, keys, args))

    def __getitem__(self, index):
        """x.__getitem__(i) <=> x[i]

        Returns a sliced view of the input symbol.

        Example
        -------
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
        output_count = len(self)
        if isinstance(index, py_slice):
            start = 0 if index.start is None else index.start
            stop = output_count if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            return Group([self[i] for i in range(start, stop, step)])

        if isinstance(index, string_types):
            # Returning this list of names is expensive. Some symbols may have hundreds of outputs
            output_names = self.list_outputs()
            idx = None
            for i, name in enumerate(output_names):
                if name == index:
                    if idx is not None:
                        raise ValueError('There are multiple outputs with name \"%s\"' % index)
                    idx = i
            if idx is None:
                raise ValueError('Cannot find output that matches name \"%s\"' % index)
            index = idx

        if not isinstance(index, int):
            raise TypeError('Symbol only support integer index to fetch i-th output')
        if index >= output_count:
            # Important, python determines the end by this exception
            raise IndexError
        handle = SymbolHandle()
        check_call(_LIB.MXSymbolGetOutput(
            self.handle, mx_uint(index), ctypes.byref(handle)))
        return Symbol(handle=handle)

    @property
    def name(self):
        """Gets name string from the symbol, this function only works for non-grouped symbol.

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

        Example
        -------
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

        Example
        -------
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
        return {py_str(pairs[i * 2]): py_str(pairs[i * 2 + 1]) for i in range(size.value)}

    def attr_dict(self):
        """Recursively gets all attributes from the symbol and its children.

        Example
        -------
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
            name, key = py_str(pairs[i * 2]).split('$')
            val = py_str(pairs[i * 2 + 1])
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

        Example
        -------
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

        Example
        -------
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

        Example
        -------
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

        Example
        -------
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

    # pylint: disable=invalid-length-returned
    def __len__(self):
        """Get number of outputs for the symbol.

        Example
        -------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> len(c)

        Returns
        -------
        len(self): Number of outputs
            Number of outputs
        """
        output_count = mx_uint()
        check_call(_LIB.MXSymbolGetNumOutputs(self.handle, ctypes.byref(output_count)))
        return output_count.value

    def list_auxiliary_states(self):
        """Lists all the auxiliary states in the symbol.

        Example
        -------
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
        aux_states : list of str
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

    def list_inputs(self):
        """Lists all arguments and auxiliary states of this Symbol.

        Returns
        -------
        inputs : list of str
            List of all inputs.

        Examples
        --------
        >>> bn = mx.sym.BatchNorm(name='bn')
        >>> bn.list_arguments()
        ['bn_data', 'bn_gamma', 'bn_beta']
        >>> bn.list_auxiliary_states()
        ['bn_moving_mean', 'bn_moving_var']
        >>> bn.list_inputs()
        ['bn_data', 'bn_gamma', 'bn_beta', 'bn_moving_mean', 'bn_moving_var']
        """
        size = ctypes.c_uint()
        sarr = ctypes.POINTER(ctypes.c_char_p)()
        check_call(_LIB.NNSymbolListInputNames(
            self.handle, 0, ctypes.byref(size), ctypes.byref(sarr)))
        return [py_str(sarr[i]) for i in range(size.value)]

    def infer_type(self, *args, **kwargs):
        """Infers the type of all arguments and all outputs, given the known types
        for some arguments.

        This function takes the known types of some arguments in either positional way
        or keyword argument way as input. It returns a tuple of `None` values
        if there is not enough information to deduce the missing types.

        Inconsistencies in the known types will cause an error to be raised.

        Example
        -------
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
        try:
            res = self._infer_type_impl(False, *args, **kwargs)
            if res[1] is None:
                arg_shapes, _, _ = self._infer_type_impl(True, *args, **kwargs)
                arg_names = self.list_arguments()
                unknowns = []
                for name, dtype in zip(arg_names, arg_shapes):
                    if not dtype:
                        if len(unknowns) >= 10:
                            unknowns.append('...')
                            break
                        unknowns.append('%s: %s' % (name, str(dtype)))
                warnings.warn(
                    "Cannot decide type for the following arguments. " +
                    "Consider providing them as input:\n\t" +
                    "\n\t".join(unknowns), stacklevel=2)
            return res
        except MXNetError:
            print("infer_type error. Arguments:")
            for i, arg in enumerate(args):
                print("  #%d: %s" % (i, arg))
            for k, v in kwargs.items():
                print("  %s: %s" % (k, v))
            raise

    def infer_type_partial(self, *args, **kwargs):
        """Infers the type partially.

        This functions works the same way as `infer_type`,
        except that this function can return partial results.

        In the following example, information about fc2 is not available. So, `infer_shape`
        will return a tuple of `None` values but `infer_shape_partial` will return partial values.

        Example
        -------
        >>> data = mx.sym.Variable('data')
        >>> prev = mx.sym.Variable('prev')
        >>> casted_prev  = mx.sym.cast(prev, dtype='float32')
        >>> out  = mx.sym.Activation(data=mx.sym.elemwise_add(data, casted_prev), act_type='relu')
        >>> out.list_arguments()
        ['data', 'prev']
        >>> out.infer_type(data='float32')
        (None, None, None)
        >>> out.infer_type_partial(data='float32')
        ([numpy.float32, None], [numpy.float32], [])
        >>> # infers type if you give information about prev
        >>> out.infer_type(data='float32', prev='float16')
        ([numpy.float32, numpy.float16], [numpy.float32], [])

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
        return self._infer_type_impl(True, *args, **kwargs)

    def _infer_type_impl(self, partial, *args, **kwargs):
        """The actual implementation for calling type inference API."""
        # pylint: disable=too-many-locals
        if len(args) != 0 and len(kwargs) != 0:
            raise ValueError('Can only specify known argument \
                    types either by positional or kwargs way.')
        sdata = []
        if len(args) != 0:
            keys = c_array(ctypes.c_char_p, [])
            for s in args:
                if s is not None:
                    s = _numpy.dtype(s).type
                    if s not in _DTYPE_NP_TO_MX:
                        raise TypeError('Argument need to be one of ' + str(_DTYPE_NP_TO_MX))
                    sdata.append(_DTYPE_NP_TO_MX[s])
                else:
                    sdata.append(-1)
        else:
            str_keys = []
            for k, v in kwargs.items():
                v = _numpy.dtype(v).type
                if v in _DTYPE_NP_TO_MX:
                    str_keys.append(k)
                    sdata.append(_DTYPE_NP_TO_MX[v])
            keys = c_str_array(str_keys)
        arg_type_size = mx_uint()
        arg_type_data = ctypes.POINTER(ctypes.c_int)()
        out_type_size = mx_uint()
        out_type_data = ctypes.POINTER(ctypes.c_int)()
        aux_type_size = mx_uint()
        aux_type_data = ctypes.POINTER(ctypes.c_int)()
        complete = ctypes.c_int()
        if partial:
            infer_func = _LIB.MXSymbolInferTypePartial
        else:
            infer_func = _LIB.MXSymbolInferType
        check_call(infer_func(
            self.handle,
            mx_uint(len(sdata)),
            keys,
            c_array_buf(ctypes.c_int, array('i', sdata)),
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

    def infer_shape(self, *args, **kwargs):
        """Infers the shapes of all arguments and all outputs given the known shapes of
        some arguments.

        This function takes the known shapes of some arguments in either positional way
        or keyword argument way as input. It returns a tuple of `None` values
        if there is not enough information to deduce the missing shapes.

        Example
        -------
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
        # pylint: disable=too-many-locals
        try:
            res = self._infer_shape_impl(False, *args, **kwargs)
            if res[1] is None:
                arg_shapes, _, _ = self._infer_shape_impl(True, *args, **kwargs)
                arg_names = self.list_arguments()
                unknowns = []
                for name, shape in zip(arg_names, arg_shapes):
                    if is_np_shape():
                        shape_is_none = not shape or -1 in shape
                    else:
                        shape_is_none = not shape or 0 in shape
                    if shape_is_none:
                        if len(unknowns) >= 10:
                            unknowns.append('...')
                            break
                        unknowns.append('%s: %s' % (name, str(shape)))
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

        Example
        -------
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
            keys = c_array(ctypes.c_char_p, [])
            for i, s in enumerate(args):
                if s is not None:
                    if not isinstance(s, tuple):
                        raise TypeError("Arguments need to be shapes (tuple), "
                                        "but argument %d is %s." % (i, type(s)))
                    sdata.extend(s)
                indptr.append(len(sdata))
        else:
            str_keys = []
            for k, v in kwargs.items():
                if not isinstance(v, tuple):
                    raise TypeError("Arguments need to be shapes (tuple), "
                                    "but '%s' is %s." % (k, type(v)))
                str_keys.append(k)
                sdata.extend(v)
                indptr.append(len(sdata))
            keys = c_str_array(str_keys)
        arg_shape_size = mx_uint()
        arg_shape_ndim = ctypes.POINTER(mx_int)()
        arg_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int))()
        out_shape_size = mx_uint()
        out_shape_ndim = ctypes.POINTER(mx_int)()
        out_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int))()
        aux_shape_size = mx_uint()
        aux_shape_ndim = ctypes.POINTER(mx_int)()
        aux_shape_data = ctypes.POINTER(ctypes.POINTER(mx_int))()
        complete = ctypes.c_int()
        if partial:
            infer_func = _LIB.MXSymbolInferShapePartialEx
        else:
            infer_func = _LIB.MXSymbolInferShapeEx
        check_call(infer_func(
            self.handle,
            mx_uint(len(indptr) - 1),
            keys,
            c_array_buf(mx_uint, array('I', indptr)),
            c_array_buf(mx_int, array('i', sdata)),
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
            arg_shapes = [tuple(arg_shape_data[i][:arg_shape_ndim[i]])
                          if arg_shape_ndim[i] >= 0 else None
                          for i in range(arg_shape_size.value)]
            out_shapes = [tuple(out_shape_data[i][:out_shape_ndim[i]])
                          if out_shape_ndim[i] >= 0 else None
                          for i in range(out_shape_size.value)]
            aux_shapes = [tuple(aux_shape_data[i][:aux_shape_ndim[i]])
                          if aux_shape_ndim[i] >= 0 else None
                          for i in range(aux_shape_size.value)]
            return (arg_shapes, out_shapes, aux_shapes)
        else:
            return (None, None, None)
        # pylint: enable=too-many-locals

    def debug_str(self):
        """Gets a debug string of symbol.

        It contains Symbol output, variables and operators in the computation graph
        with their inputs, variables and attributes.

        Returns
        -------
        string
            Debug string of the symbol.

        Examples
        --------
        >>> a = mx.sym.Variable('a')
        >>> b = mx.sym.sin(a)
        >>> c = 2 * a + b
        >>> d = mx.sym.FullyConnected(data=c, num_hidden=10)
        >>> d.debug_str()
        >>> print d.debug_str()
        Symbol Outputs:
	        output[0]=fullyconnected0(0)
        Variable:a
        --------------------
        Op:_mul_scalar, Name=_mulscalar0
        Inputs:
        	arg[0]=a(0) version=0
        Attrs:
        	scalar=2
        --------------------
        Op:sin, Name=sin0
        Inputs:
        	arg[0]=a(0) version=0
        --------------------
        Op:elemwise_add, Name=_plus0
        Inputs:
        	arg[0]=_mulscalar0(0)
        	arg[1]=sin0(0)
        Variable:fullyconnected0_weight
        Variable:fullyconnected0_bias
        --------------------
        Op:FullyConnected, Name=fullyconnected0
        Inputs:
        	arg[0]=_plus0(0)
        	arg[1]=fullyconnected0_weight(0) version=0
        	arg[2]=fullyconnected0_bias(0) version=0
        Attrs:
        	num_hidden=10
        """
        debug_str = ctypes.c_char_p()
        check_call(_LIB.MXSymbolPrint(
            self.handle, ctypes.byref(debug_str)))
        return py_str(debug_str.value)

    def save(self, fname, remove_amp_cast=True):
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
        remove_amp_cast : bool, optional
            Whether to remove the amp_cast and amp_multicast operators, before saving the model.

        See Also
        --------
        symbol.load : Used to load symbol from file.
        """
        if not isinstance(fname, string_types):
            raise TypeError('fname need to be string')
        if remove_amp_cast:
            handle = SymbolHandle()
            check_call(_LIB.MXSymbolRemoveAmpCast(self.handle, ctypes.byref(handle)))
            check_call(_LIB.MXSymbolSaveToFile(handle, c_str(fname)))
        else:
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
                if narr is None and allow_missing:
                    arg_handles.append(None)
                elif not isinstance(narr, NDArray):
                    raise TypeError('Only accept list of NDArrays or dict of str to NDArray')
                else:
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

    def _gen_atomic_symbol(self):
        handle = SymbolHandle()
        check_call(_LIB.MXGenAtomicSymbolFromSymbol(self.handle, ctypes.byref(handle)))
        return Symbol(handle)


    # pylint: disable=too-many-locals
    def simple_bind(self, ctx, grad_req='write', type_dict=None, stype_dict=None,
                    group2ctx=None, shared_arg_names=None, shared_exec=None,
                    shared_buffer=None, **kwargs):
        """Bind current symbol to get an executor, allocate all the arguments needed.
        Allows specifying data types.

        This function simplifies the binding procedure. You need to specify only input data shapes.
        Before binding the executor, the function allocates arguments and auxiliary states
        that were not explicitly specified. Allows specifying data types.

        Example
        -------
        >>> x = mx.sym.Variable('x')
        >>> y = mx.sym.FullyConnected(x, num_hidden=4)
        >>> exe = y.simple_bind(mx.cpu(), x=(5,4), grad_req='null')
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

        stype_dict  : Dict of str->str
            Input storage type dictionary, name->storage_type

        group2ctx : Dict of string to mx.Context
            The dict mapping the `ctx_group` attribute to the context assignment.

        shared_arg_names : List of string
            The argument names whose `NDArray` of shared_exec can be reused for initializing
            the current executor.

        shared_exec : Executor
            The executor whose arg_arrays, arg_arrays, grad_arrays, and aux_arrays can be
            reused for initializing the current executor.

        shared_buffer : Dict of string to `NDArray`
            The dict mapping argument names to the `NDArray` that can be reused for initializing
            the current executor. This buffer will be checked for reuse if one argument name
            of the current executor is not found in `shared_arg_names`. The `NDArray` s are
            expected have default storage type.

        kwargs : Dict of str->shape
            Input shape dictionary, name->shape

        Returns
        -------
        executor : mxnet.Executor
            The generated executor
        """
        # data types
        num_provided_arg_types = 0
        provided_arg_type_names = ctypes.POINTER(ctypes.c_char_p)()  # provided type argument names
        provided_arg_type_data = ctypes.POINTER(mx_uint)()  # provided types
        if type_dict is not None:
            provided_arg_type_names = []
            provided_arg_type_data = []
            for k, v in type_dict.items():
                v = _numpy.dtype(v).type
                if v in _DTYPE_NP_TO_MX:
                    provided_arg_type_names.append(k)
                    provided_arg_type_data.append(_DTYPE_NP_TO_MX[v])
            num_provided_arg_types = mx_uint(len(provided_arg_type_names))
            provided_arg_type_names = c_str_array(provided_arg_type_names)
            provided_arg_type_data = c_array_buf(ctypes.c_int, array('i', provided_arg_type_data))

        # storage types
        num_provided_arg_stypes = 0
        # provided storage type argument names
        provided_arg_stype_names = ctypes.POINTER(ctypes.c_char_p)()
        provided_arg_stype_data = ctypes.POINTER(mx_uint)()  # provided storage types
        if stype_dict is not None:
            provided_arg_stype_names = []
            provided_arg_stype_data = []
            for k, v in stype_dict.items():
                if v in _STORAGE_TYPE_STR_TO_ID:
                    provided_arg_stype_names.append(k)
                    provided_arg_stype_data.append(_STORAGE_TYPE_STR_TO_ID[v])
            num_provided_arg_stypes = mx_uint(len(provided_arg_stype_names))
            provided_arg_stype_names = c_str_array(provided_arg_stype_names)
            provided_arg_stype_data = c_array_buf(ctypes.c_int, array('i', provided_arg_stype_data))

        provided_arg_shape_data = []  # shape data
        # argument shape index in sdata,
        # e.g. [sdata[indptr[0]], sdata[indptr[1]]) is the shape of the first arg
        provided_arg_shape_idx = [0]
        provided_arg_shape_names = []  # provided argument names
        for k, v in kwargs.items():
            # if k not in listed_arguments and k not in listed_aux_states:
            #   raise ValueError('arg name %s is not valid', k)
            if isinstance(v, tuple):
                provided_arg_shape_names.append(k)
                provided_arg_shape_data.extend(v)
                provided_arg_shape_idx.append(len(provided_arg_shape_data))

        provided_req_type_list_len = 0
        provided_grad_req_types = ctypes.POINTER(ctypes.c_char_p)()
        provided_grad_req_names = ctypes.POINTER(ctypes.c_char_p)()
        if grad_req is not None:
            if isinstance(grad_req, string_types):
                # use provided_req_type_list_len = 0 to indicate this situation
                provided_req_type_list_len = 0
                provided_grad_req_types = [grad_req]
            elif isinstance(grad_req, list):
                if len(grad_req) == 0:
                    raise RuntimeError('grad_req in simple_bind cannot be an empty list')
                provided_grad_req_types = grad_req
                provided_req_type_list_len = len(provided_grad_req_types)
            elif isinstance(grad_req, dict):
                if len(grad_req) == 0:
                    raise RuntimeError('grad_req in simple_bind cannot be an empty dict')
                provided_grad_req_names = []
                provided_grad_req_types = []
                for k, v in grad_req.items():
                    provided_grad_req_names.append(k)
                    provided_grad_req_types.append(v)
                provided_grad_req_names = c_str_array(provided_grad_req_names)
                provided_req_type_list_len = len(provided_grad_req_types)
            provided_grad_req_types = c_str_array(provided_grad_req_types)

        num_ctx_map_keys = mx_uint(0)
        ctx_map_keys = ctypes.POINTER(ctypes.c_char_p)()
        ctx_map_dev_types = ctypes.POINTER(ctypes.c_int)()
        ctx_map_dev_ids = ctypes.POINTER(ctypes.c_int)()
        if group2ctx is not None:
            ctx_map_keys = []
            ctx_map_dev_types = []
            ctx_map_dev_ids = []
            for key, val in group2ctx.items():
                ctx_map_keys.append(key)
                ctx_map_dev_types.append(val.device_typeid)
                ctx_map_dev_ids.append(val.device_id)
            num_ctx_map_keys = mx_uint(len(ctx_map_keys))
            ctx_map_keys = c_str_array(ctx_map_keys)
            ctx_map_dev_types = c_array(ctypes.c_int, array('i', ctx_map_dev_types))
            ctx_map_dev_ids = c_array(ctypes.c_int, array('i', ctx_map_dev_ids))

        # prepare param names
        shared_arg_name_list = []
        if shared_arg_names is not None:
            if not isinstance(shared_arg_names, list):
                raise ValueError('shared_arg_names in simple_bind must be a list or None')
            shared_arg_name_list = shared_arg_names

        # prepare shared_buffer
        if shared_buffer is None:
            shared_buffer_len = ctypes.c_int(-1)
            shared_buffer_names = ctypes.POINTER(ctypes.c_char_p)()
            shared_buffer_handles = ctypes.POINTER(NDArrayHandle)()
        else:
            if not isinstance(shared_buffer, dict):
                raise ValueError('shared_buffer in simple_bind must be dict or None')
            buffer_names = shared_buffer.keys()
            buffer_arrays = shared_buffer.values()
            for v in buffer_arrays:
                assert(v.stype == 'default'), \
                    "shared_buffer is expected to only contain NDArrays with default storage"
            shared_buffer_names = c_str_array(buffer_names)
            shared_buffer_len = ctypes.c_int(len(buffer_arrays))
            shared_buffer_handles = c_handle_array(buffer_arrays)
        updated_shared_buffer_names = ctypes.POINTER(ctypes.c_char_p)()
        updated_shared_buffer_handles = ctypes.POINTER(NDArrayHandle)()

        # prepare shared_exec_handle
        shared_exec_handle = shared_exec.handle if shared_exec is not None else ExecutorHandle()

        # prepare current executor handle
        exe_handle = ExecutorHandle()

        # prepare current executor's in_args, arg_grads, and aux_states
        num_in_args = ctypes.c_uint()
        in_arg_handles = ctypes.POINTER(NDArrayHandle)()
        arg_grad_handles = ctypes.POINTER(NDArrayHandle)()
        num_aux_states = ctypes.c_uint()
        aux_state_handles = ctypes.POINTER(NDArrayHandle)()

        try:
            check_call(_LIB.MXExecutorSimpleBindEx(self.handle,
                                                   ctypes.c_int(ctx.device_typeid),
                                                   ctypes.c_int(ctx.device_id),
                                                   num_ctx_map_keys,
                                                   ctx_map_keys,
                                                   ctx_map_dev_types,
                                                   ctx_map_dev_ids,
                                                   mx_uint(provided_req_type_list_len),
                                                   provided_grad_req_names,
                                                   provided_grad_req_types,
                                                   mx_uint(len(provided_arg_shape_names)),
                                                   c_str_array(provided_arg_shape_names),
                                                   c_array_buf(mx_int,
                                                               array('I', provided_arg_shape_data)),
                                                   c_array_buf(mx_uint,
                                                               array('i', provided_arg_shape_idx)),
                                                   num_provided_arg_types,
                                                   provided_arg_type_names,
                                                   provided_arg_type_data,
                                                   num_provided_arg_stypes,
                                                   provided_arg_stype_names,
                                                   provided_arg_stype_data,
                                                   mx_uint(len(shared_arg_name_list)),
                                                   c_str_array(shared_arg_name_list),
                                                   ctypes.byref(shared_buffer_len),
                                                   shared_buffer_names,
                                                   shared_buffer_handles,
                                                   ctypes.byref(updated_shared_buffer_names),
                                                   ctypes.byref(updated_shared_buffer_handles),
                                                   ctypes.byref(num_in_args),
                                                   ctypes.byref(in_arg_handles),
                                                   ctypes.byref(arg_grad_handles),
                                                   ctypes.byref(num_aux_states),
                                                   ctypes.byref(aux_state_handles),
                                                   shared_exec_handle,
                                                   ctypes.byref(exe_handle)))
        except MXNetError as e:
            error_msg = "simple_bind error. Arguments:\n"
            for k, v in kwargs.items():
                error_msg += "%s: %s\n" % (k, v)
            error_msg += "%s" % e
            raise RuntimeError(error_msg)

        # update shared_buffer
        if shared_buffer is not None:
            for i in range(shared_buffer_len.value):
                k = py_str(updated_shared_buffer_names[i])
                v = NDArray(NDArrayHandle(updated_shared_buffer_handles[i]))
                shared_buffer[k] = v

        # create in_args, arg_grads, and aux_states for the current executor
        arg_arrays = [_ndarray_cls(NDArrayHandle(in_arg_handles[i]))
                      for i in range(num_in_args.value)]
        grad_arrays = [_ndarray_cls(NDArrayHandle(arg_grad_handles[i]))
                       if arg_grad_handles[i] is not None
                       else None for i in range(num_in_args.value)]
        aux_arrays = [_ndarray_cls(NDArrayHandle(aux_state_handles[i]))
                      for i in range(num_aux_states.value)]

        executor = Executor(exe_handle, self, ctx, grad_req, group2ctx)
        executor.arg_arrays = arg_arrays
        executor.grad_arrays = grad_arrays
        executor.aux_arrays = aux_arrays
        return executor

    def bind(self, ctx, args, args_grad=None, grad_req='write',
             aux_states=None, group2ctx=None, shared_exec=None):
        """Binds the current symbol to an executor and returns it.

        We first declare the computation and then bind to the data to run.
        This function returns an executor which provides method `forward()` method for evaluation
        and a `outputs()` method to get all the results.

        Example
        -------
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
            reqs_array = c_array_buf(mx_uint,
                                     array('I', [_GRAD_REQ_MAP[grad_req]] * len(listed_arguments)))
        elif isinstance(grad_req, list):
            reqs_array = c_array_buf(mx_uint,
                                     array('I', [_GRAD_REQ_MAP[item] for item in grad_req]))
        elif isinstance(grad_req, dict):
            req_array = []
            for name in listed_arguments:
                if name in grad_req:
                    req_array.append(_GRAD_REQ_MAP[grad_req[name]])
                else:
                    req_array.append(0)
            reqs_array = c_array_buf(mx_uint, array('I', req_array))

        ctx_map_keys = []
        ctx_map_dev_types = []
        ctx_map_dev_ids = []

        if group2ctx:
            for key, val in group2ctx.items():
                ctx_map_keys.append(key)
                ctx_map_dev_types.append(val.device_typeid)
                ctx_map_dev_ids.append(val.device_id)

        handle = ExecutorHandle()
        shared_handle = shared_exec.handle if shared_exec is not None else ExecutorHandle()
        check_call(_LIB.MXExecutorBindEX(self.handle,
                                         ctypes.c_int(ctx.device_typeid),
                                         ctypes.c_int(ctx.device_id),
                                         mx_uint(len(ctx_map_keys)),
                                         c_str_array(ctx_map_keys),
                                         c_array_buf(ctypes.c_int, array('i', ctx_map_dev_types)),
                                         c_array_buf(ctypes.c_int, array('i', ctx_map_dev_ids)),
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

    def gradient(self, wrt):
        """Gets the autodiff of current symbol.

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
        c_wrt = c_str_array(wrt)
        check_call(_LIB.MXSymbolGrad(self.handle,
                                     mx_uint(len(wrt)),
                                     c_wrt,
                                     ctypes.byref(handle)))
        return Symbol(handle)

    # pylint: enable= no-member

    def eval(self, ctx=None, **kwargs):
        """Evaluates a symbol given arguments.

        The `eval` method combines a call to `bind` (which returns an executor)
        with a call to `forward` (executor method).
        For the common use case, where you might repeatedly evaluate with same arguments,
        eval is slow.
        In that case, you should call `bind` once and then repeatedly call forward.
        This function allows simpler syntax for less cumbersome introspection.

        Example
        -------
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
        if ctx is None:
            ctx = current_context()
        return self.bind(ctx, kwargs).forward()

    def reshape(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reshape`.

        The arguments are the same as for :py:func:`reshape`, with
        this array as data.
        """
        return op.reshape(self, *args, **kwargs)

    def reshape_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reshape_like`.

        The arguments are the same as for :py:func:`reshape_like`, with
        this array as data.
        """
        return op.reshape_like(self, *args, **kwargs)

    def astype(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cast`.

        The arguments are the same as for :py:func:`cast`, with
        this array as data.
        """
        return op.cast(self, *args, **kwargs)

    def zeros_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`zeros_like`.

        The arguments are the same as for :py:func:`zeros_like`, with
        this array as data.
        """
        return op.zeros_like(self, *args, **kwargs)

    def ones_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ones_like`.

        The arguments are the same as for :py:func:`ones_like`, with
        this array as data.
        """
        return op.ones_like(self, *args, **kwargs)

    def broadcast_axes(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`broadcast_axes`.

        The arguments are the same as for :py:func:`broadcast_axes`, with
        this array as data.
        """
        return op.broadcast_axes(self, *args, **kwargs)

    def repeat(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`repeat`.

        The arguments are the same as for :py:func:`repeat`, with
        this array as data.
        """
        return op.repeat(self, *args, **kwargs)

    def pad(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pad`.

        The arguments are the same as for :py:func:`pad`, with
        this array as data.
        """
        return op.pad(self, *args, **kwargs)

    def swapaxes(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`swapaxes`.

        The arguments are the same as for :py:func:`swapaxes`, with
        this array as data.
        """
        return op.swapaxes(self, *args, **kwargs)

    def split(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split`.

        The arguments are the same as for :py:func:`split`, with
        this array as data.
        """
        return op.split(self, *args, **kwargs)

    def split_v2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`split_v2`.

        The arguments are the same as for :py:func:`split_v2`, with
        this array as data.
        """
        return split_v2(self, *args, **kwargs)

    def slice(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice`.

        The arguments are the same as for :py:func:`slice`, with
        this array as data.
        """
        return op.slice(self, *args, **kwargs)

    def slice_axis(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_axis`.

        The arguments are the same as for :py:func:`slice_axis`, with
        this array as data.
        """
        return op.slice_axis(self, *args, **kwargs)

    def slice_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`slice_like`.

        The arguments are the same as for :py:func:`slice_like`, with
        this array as data.
        """
        return op.slice_like(self, *args, **kwargs)

    def take(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`take`.

        The arguments are the same as for :py:func:`take`, with
        this array as data.
        """
        return op.take(self, *args, **kwargs)

    def one_hot(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`one_hot`.

        The arguments are the same as for :py:func:`one_hot`, with
        this array as data.
        """
        return op.one_hot(self, *args, **kwargs)

    def pick(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`pick`.

        The arguments are the same as for :py:func:`pick`, with
        this array as data.
        """
        return op.pick(self, *args, **kwargs)

    def sort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sort`.

        The arguments are the same as for :py:func:`sort`, with
        this array as data.
        """
        return op.sort(self, *args, **kwargs)

    def topk(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`topk`.

        The arguments are the same as for :py:func:`topk`, with
        this array as data.
        """
        return op.topk(self, *args, **kwargs)

    def argsort(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argsort`.

        The arguments are the same as for :py:func:`argsort`, with
        this array as data.
        """
        return op.argsort(self, *args, **kwargs)

    def argmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax`.

        The arguments are the same as for :py:func:`argmax`, with
        this array as data.
        """
        return op.argmax(self, *args, **kwargs)

    def argmax_channel(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmax_channel`.

        The arguments are the same as for :py:func:`argmax_channel`, with
        this array as data.
        """
        return op.argmax_channel(self, *args, **kwargs)

    def argmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`argmin`.

        The arguments are the same as for :py:func:`argmin`, with
        this array as data.
        """
        return op.argmin(self, *args, **kwargs)

    def clip(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`clip`.

        The arguments are the same as for :py:func:`clip`, with
        this array as data.
        """
        return op.clip(self, *args, **kwargs)

    def abs(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`abs`.

        The arguments are the same as for :py:func:`abs`, with
        this array as data.
        """
        return op.abs(self, *args, **kwargs)

    def sign(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sign`.

        The arguments are the same as for :py:func:`sign`, with
        this array as data.
        """
        return op.sign(self, *args, **kwargs)

    def flatten(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`flatten`.

        The arguments are the same as for :py:func:`flatten`, with
        this array as data.
        """
        return op.flatten(self, *args, **kwargs)

    def shape_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`shape_array`.

        The arguments are the same as for :py:func:`shape_op`, with
        this array as data.
        """
        return op.shape_array(self, *args, **kwargs)

    def size_array(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`size_array`.

        The arguments are the same as for :py:func:`size_array`, with
        this array as data.
        """
        return op.size_array(self, *args, **kwargs)

    def expand_dims(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expand_dims`.

        The arguments are the same as for :py:func:`expand_dims`, with
        this array as data.
        """
        return op.expand_dims(self, *args, **kwargs)

    def broadcast_to(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`broadcast_to`.

        The arguments are the same as for :py:func:`broadcast_to`, with
        this array as data.
        """
        return op.broadcast_to(self, *args, **kwargs)

    def broadcast_like(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`broadcast_like`.

        The arguments are the same as for :py:func:`broadcast_like`, with
        this array as data.
        """
        return op.broadcast_like(self, *args, **kwargs)

    def tile(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tile`.

        The arguments are the same as for :py:func:`tile`, with
        this array as data.
        """
        return op.tile(self, *args, **kwargs)

    def transpose(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`transpose`.

        The arguments are the same as for :py:func:`transpose`, with
        this array as data.
        """
        return op.transpose(self, *args, **kwargs)

    def flip(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`flip`.

        The arguments are the same as for :py:func:`flip`, with
        this array as data.
        """
        return op.flip(self, *args, **kwargs)

    def depth_to_space(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`depth_to_space`.

        The arguments are the same as for :py:func:`depth_to_space`, with
        this array as data.
        """
        return op.depth_to_space(self, *args, **kwargs)

    def space_to_depth(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`space_to_depth`.

        The arguments are the same as for :py:func:`space_to_depth`, with
        this array as data.
        """
        return op.space_to_depth(self, *args, **kwargs)

    def diag(self, k=0, **kwargs):
        """Convenience fluent method for :py:func:`diag`.

        The arguments are the same as for :py:func:`diag`, with
        this array as data.
        """
        return op.diag(self, k, **kwargs)

    def sum(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sum`.

        The arguments are the same as for :py:func:`sum`, with
        this array as data.
        """
        return op.sum(self, *args, **kwargs)

    def nansum(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nansum`.

        The arguments are the same as for :py:func:`nansum`, with
        this array as data.
        """
        return op.nansum(self, *args, **kwargs)

    def prod(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`prod`.

        The arguments are the same as for :py:func:`prod`, with
        this array as data.
        """
        return op.prod(self, *args, **kwargs)

    def nanprod(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`nanprod`.

        The arguments are the same as for :py:func:`nanprod`, with
        this array as data.
        """
        return op.nanprod(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`mean`.

        The arguments are the same as for :py:func:`mean`, with
        this array as data.
        """
        return op.mean(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`max`.

        The arguments are the same as for :py:func:`max`, with
        this array as data.
        """
        return op.max(self, *args, **kwargs)

    def min(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`min`.

        The arguments are the same as for :py:func:`min`, with
        this array as data.
        """
        return op.min(self, *args, **kwargs)

    def norm(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`norm`.

        The arguments are the same as for :py:func:`norm`, with
        this array as data.
        """
        return op.norm(self, *args, **kwargs)

    def round(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`round`.

        The arguments are the same as for :py:func:`round`, with
        this array as data.
        """
        return op.round(self, *args, **kwargs)

    def rint(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rint`.

        The arguments are the same as for :py:func:`rint`, with
        this array as data.
        """
        return op.rint(self, *args, **kwargs)

    def fix(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`fix`.

        The arguments are the same as for :py:func:`fix`, with
        this array as data.
        """
        return op.fix(self, *args, **kwargs)

    def floor(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`floor`.

        The arguments are the same as for :py:func:`floor`, with
        this array as data.
        """
        return op.floor(self, *args, **kwargs)

    def ceil(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`ceil`.

        The arguments are the same as for :py:func:`ceil`, with
        this array as data.
        """
        return op.ceil(self, *args, **kwargs)

    def trunc(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`trunc`.

        The arguments are the same as for :py:func:`trunc`, with
        this array as data.
        """
        return op.trunc(self, *args, **kwargs)

    def sin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sin`.

        The arguments are the same as for :py:func:`sin`, with
        this array as data.
        """
        return op.sin(self, *args, **kwargs)

    def cos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cos`.

        The arguments are the same as for :py:func:`cos`, with
        this array as data.
        """
        return op.cos(self, *args, **kwargs)

    def tan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tan`.

        The arguments are the same as for :py:func:`tan`, with
        this array as data.
        """
        return op.tan(self, *args, **kwargs)

    def arcsin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsin`.

        The arguments are the same as for :py:func:`arcsin`, with
        this array as data.
        """
        return op.arcsin(self, *args, **kwargs)

    def arccos(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccos`.

        The arguments are the same as for :py:func:`arccos`, with
        this array as data.
        """
        return op.arccos(self, *args, **kwargs)

    def arctan(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctan`.

        The arguments are the same as for :py:func:`arctan`, with
        this array as data.
        """
        return op.arctan(self, *args, **kwargs)

    def degrees(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`degrees`.

        The arguments are the same as for :py:func:`degrees`, with
        this array as data.
        """
        return op.degrees(self, *args, **kwargs)

    def radians(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`radians`.

        The arguments are the same as for :py:func:`radians`, with
        this array as data.
        """
        return op.radians(self, *args, **kwargs)

    def sinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sinh`.

        The arguments are the same as for :py:func:`sinh`, with
        this array as data.
        """
        return op.sinh(self, *args, **kwargs)

    def cosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cosh`.

        The arguments are the same as for :py:func:`cosh`, with
        this array as data.
        """
        return op.cosh(self, *args, **kwargs)

    def tanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`tanh`.

        The arguments are the same as for :py:func:`tanh`, with
        this array as data.
        """
        return op.tanh(self, *args, **kwargs)

    def arcsinh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arcsinh`.

        The arguments are the same as for :py:func:`arcsinh`, with
        this array as data.
        """
        return op.arcsinh(self, *args, **kwargs)

    def arccosh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arccosh`.

        The arguments are the same as for :py:func:`arccosh`, with
        this array as data.
        """
        return op.arccosh(self, *args, **kwargs)

    def arctanh(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`arctanh`.

        The arguments are the same as for :py:func:`arctanh`, with
        this array as data.
        """
        return op.arctanh(self, *args, **kwargs)

    def exp(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`exp`.

        The arguments are the same as for :py:func:`exp`, with
        this array as data.
        """
        return op.exp(self, *args, **kwargs)

    def expm1(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`expm1`.

        The arguments are the same as for :py:func:`expm1`, with
        this array as data.
        """
        return op.expm1(self, *args, **kwargs)

    def log(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log`.

        The arguments are the same as for :py:func:`log`, with
        this array as data.
        """
        return op.log(self, *args, **kwargs)

    def log10(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log10`.

        The arguments are the same as for :py:func:`log10`, with
        this array as data.
        """
        return op.log10(self, *args, **kwargs)

    def log2(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log2`.

        The arguments are the same as for :py:func:`log2`, with
        this array as data.
        """
        return op.log2(self, *args, **kwargs)

    def log1p(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log1p`.

        The arguments are the same as for :py:func:`log1p`, with
        this array as data.
        """
        return op.log1p(self, *args, **kwargs)

    def sqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sqrt`.

        The arguments are the same as for :py:func:`sqrt`, with
        this array as data.
        """
        return op.sqrt(self, *args, **kwargs)

    def rsqrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rsqrt`.

        The arguments are the same as for :py:func:`rsqrt`, with
        this array as data.
        """
        return op.rsqrt(self, *args, **kwargs)

    def cbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`cbrt`.

        The arguments are the same as for :py:func:`cbrt`, with
        this array as data.
        """
        return op.cbrt(self, *args, **kwargs)

    def rcbrt(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`rcbrt`.

        The arguments are the same as for :py:func:`rcbrt`, with
        this array as data.
        """
        return op.rcbrt(self, *args, **kwargs)

    def square(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`square`.

        The arguments are the same as for :py:func:`square`, with
        this array as data.
        """
        return op.square(self, *args, **kwargs)

    def reciprocal(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`reciprocal`.

        The arguments are the same as for :py:func:`reciprocal`, with
        this array as data.
        """
        return op.reciprocal(self, *args, **kwargs)

    def relu(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`relu`.

        The arguments are the same as for :py:func:`relu`, with
        this array as data.
        """
        return op.relu(self, *args, **kwargs)

    def sigmoid(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`sigmoid`.

        The arguments are the same as for :py:func:`sigmoid`, with
        this array as data.
        """
        return op.sigmoid(self, *args, **kwargs)

    def softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmax`.

        The arguments are the same as for :py:func:`softmax`, with
        this array as data.
        """
        return op.softmax(self, *args, **kwargs)

    def log_softmax(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`log_softmax`.

        The arguments are the same as for :py:func:`log_softmax`, with
        this array as data.
        """
        return op.log_softmax(self, *args, **kwargs)

    def softmin(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`softmin`.

        The arguments are the same as for :py:func:`softmin`, with
        this array as data.
        """
        return op.softmin(self, *args, **kwargs)

    def squeeze(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`squeeze`.

        The arguments are the same as for :py:func:`squeeze`, with
        this array as data.
        """
        return op.squeeze(self, *args, **kwargs)

    def get_backend_symbol(self, backend):
        """Return symbol for target backend.

        Parameters
        ----------
        backend : str
            The backend names.

        Returns
        -------
        out : Symbol
            The created Symbol for target backend.
        """
        out = SymbolHandle()
        check_call(_LIB.MXGenBackendSubgraph(self.handle, c_str(backend), ctypes.byref(out)))
        return Symbol(out)

    def wait_to_read(self):
        raise NotImplementedForSymbol(self.wait_to_read, None)

    def asnumpy(self):
        raise NotImplementedForSymbol(self.asnumpy, None)

    def asscalar(self):
        raise NotImplementedForSymbol(self.asscalar, None)

    def copy(self):
        raise NotImplementedForSymbol(self.copy, None)

    def as_in_context(self):
        raise NotImplementedForSymbol(self.as_in_context, None)

    def detach(self):
        raise NotImplementedForSymbol(self.detach, None)

    def backward(self):
        raise NotImplementedForSymbol(self.backward, None)

def var(name, attr=None, shape=None, lr_mult=None, wd_mult=None, dtype=None,
        init=None, stype=None, **kwargs):
    """Creates a symbolic variable with specified name.

    Example
    -------
    >>> data = mx.sym.Variable('data', attr={'a': 'b'})
    >>> data
    <Symbol data>
    >>> csr_data = mx.sym.Variable('csr_data', stype='csr')
    >>> csr_data
    <Symbol csr_data>
    >>> row_sparse_weight = mx.sym.Variable('weight', stype='row_sparse')
    >>> row_sparse_weight
    <Symbol weight>

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
    stype : str
        The storage type of the variable, such as 'row_sparse', 'csr', 'default', etc
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
    if not hasattr(AttrScope._current, "value"):
        AttrScope._current.value = AttrScope()
    attr = AttrScope._current.value.get(attr)
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
        if not isinstance(init, string_types):
            init = init.dumps()
        attr['__init__'] = init
    if stype is not None:
        attr['__storage_type__'] = str(_STORAGE_TYPE_STR_TO_ID[stype])
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


def Group(symbols, create_fn=Symbol):
    """Creates a symbol that contains a collection of other symbols, grouped together.
    A classic symbol (`mx.sym.Symbol`) will be returned if all the symbols in the list
    are of that type; a numpy symbol (`mx.sym.np._Symbol`) will be returned if all the
    symbols in the list are of that type. A type error will be raised if a list of mixed
    classic and numpy symbols are provided.

    Example
    -------
    >>> a = mx.sym.Variable('a')
    >>> b = mx.sym.Variable('b')
    >>> mx.sym.Group([a,b])
    <Symbol Grouped>

    Parameters
    ----------
    symbols : list
        List of symbols to be grouped.

    create_fn : mx.sym.Symbol or mx.sym.np._Symbol
        Symbol class for creating the grouped symbol.

    Returns
    -------
    sym : Symbol
        A group symbol.
     """
    if not symbols or any(not isinstance(sym, Symbol) for sym in symbols):
        raise TypeError('Expected a list of symbols as input')
    handle = SymbolHandle()
    check_call(_LIB.MXSymbolCreateGroup(
        mx_uint(len(symbols)),
        c_handle_array(symbols), ctypes.byref(handle)))
    return create_fn(handle)


def load(fname):
    """Loads symbol from a JSON file.

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
    """Loads symbol from json string.

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


# pylint: disable=no-member
# pylint: disable=redefined-builtin
def pow(base, exp):
    """Returns element-wise result of base element raised to powers from exp element.

    Both inputs can be Symbol or scalar number.
    Broadcasting is not supported. Use `broadcast_pow` instead.

    `sym.pow` is being deprecated, please use `sym.power` instead.

    Parameters
    ---------
    base : Symbol or scalar
        The base symbol
    exp : Symbol or scalar
        The exponent symbol

    Returns
    -------
    Symbol or scalar
        The bases in x raised to the exponents in y.

    Examples
    --------
    >>> mx.sym.pow(2, 3)
    8
    >>> x = mx.sym.Variable('x')
    >>> y = mx.sym.Variable('y')
    >>> z = mx.sym.pow(x, 2)
    >>> z.eval(x=mx.nd.array([1,2]))[0].asnumpy()
    array([ 1.,  4.], dtype=float32)
    >>> z = mx.sym.pow(3, y)
    >>> z.eval(y=mx.nd.array([2,3]))[0].asnumpy()
    array([  9.,  27.], dtype=float32)
    >>> z = mx.sym.pow(x, y)
    >>> z.eval(x=mx.nd.array([3,4]), y=mx.nd.array([2,3]))[0].asnumpy()
    array([  9.,  64.], dtype=float32)
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


def power(base, exp):
    """Returns element-wise result of base element raised to powers from exp element.

    Both inputs can be Symbol or scalar number.
    Broadcasting is not supported. Use `broadcast_pow` instead.

    Parameters
    ---------
    base : Symbol or scalar
        The base symbol
    exp : Symbol or scalar
        The exponent symbol

    Returns
    -------
    Symbol or scalar
        The bases in x raised to the exponents in y.

    Examples
    --------
    >>> mx.sym.power(2, 3)
    8
    >>> x = mx.sym.Variable('x')
    >>> y = mx.sym.Variable('y')
    >>> z = mx.sym.power(x, 2)
    >>> z.eval(x=mx.nd.array([1,2]))[0].asnumpy()
    array([ 1.,  4.], dtype=float32)
    >>> z = mx.sym.power(3, y)
    >>> z.eval(y=mx.nd.array([2,3]))[0].asnumpy()
    array([  9.,  27.], dtype=float32)
    >>> z = mx.sym.power(x, y)
    >>> z.eval(x=mx.nd.array([3,4]), y=mx.nd.array([2,3]))[0].asnumpy()
    array([  9.,  64.], dtype=float32)
    """
    return pow(base, exp)


# pylint: disable=no-member
# pylint: disable=redefined-builtin
def maximum(left, right):
    """Returns element-wise maximum of the input elements.

    Both inputs can be Symbol or scalar number. Broadcasting is not supported.

    Parameters
    ---------
    left : Symbol or scalar
        First symbol to be compared.
    right : Symbol or scalar
        Second symbol to be compared.

    Returns
    -------
    Symbol or scalar
        The element-wise maximum of the input symbols.

    Examples
    --------
    >>> mx.sym.maximum(2, 3.5)
    3.5
    >>> x = mx.sym.Variable('x')
    >>> y = mx.sym.Variable('y')
    >>> z = mx.sym.maximum(x, 4)
    >>> z.eval(x=mx.nd.array([3,5,2,10]))[0].asnumpy()
    array([  4.,   5.,   4.,  10.], dtype=float32)
    >>> z = mx.sym.maximum(x, y)
    >>> z.eval(x=mx.nd.array([3,4]), y=mx.nd.array([10,2]))[0].asnumpy()
    array([ 10.,   4.], dtype=float32)
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
    """Returns element-wise minimum of the input elements.

    Both inputs can be Symbol or scalar number. Broadcasting is not supported.

    Parameters
    ---------
    left : Symbol or scalar
        First symbol to be compared.
    right : Symbol or scalar
        Second symbol to be compared.

    Returns
    -------
    Symbol or scalar
        The element-wise minimum of the input symbols.

    Examples
    --------
    >>> mx.sym.minimum(2, 3.5)
    2
    >>> x = mx.sym.Variable('x')
    >>> y = mx.sym.Variable('y')
    >>> z = mx.sym.minimum(x, 4)
    >>> z.eval(x=mx.nd.array([3,5,2,10]))[0].asnumpy()
    array([ 3.,  4.,  2.,  4.], dtype=float32)
    >>> z = mx.sym.minimum(x, y)
    >>> z.eval(x=mx.nd.array([3,4]), y=mx.nd.array([10,2]))[0].asnumpy()
    array([ 3.,  2.], dtype=float32)
    """
    if isinstance(left, Symbol) and isinstance(right, Symbol):
        return _internal._Minimum(left, right)
    if isinstance(left, Symbol) and isinstance(right, Number):
        return _internal._MinimumScalar(left, scalar=right)
    if isinstance(left, Number) and isinstance(right, Symbol):
        return _internal._MinimumScalar(right, scalar=left)
    if isinstance(left, Number) and isinstance(right, Number):
        return left if left < right else right
    else:
        raise TypeError('types (%s, %s) not supported' % (str(type(left)), str(type(right))))


# pylint: disable=no-member
# pylint: disable=redefined-builtin
def hypot(left, right):
    """Given the "legs" of a right triangle, returns its hypotenuse.

    Equivalent to :math:`\\sqrt(left^2 + right^2)`, element-wise.
    Both inputs can be Symbol or scalar number. Broadcasting is not supported.

    Parameters
    ---------
    left : Symbol or scalar
        First leg of the triangle(s).
    right : Symbol or scalar
        Second leg of the triangle(s).

    Returns
    -------
    Symbol or scalar
        The hypotenuse of the triangle(s)

    Examples
    --------
    >>> mx.sym.hypot(3, 4)
    5.0
    >>> x = mx.sym.Variable('x')
    >>> y = mx.sym.Variable('y')
    >>> z = mx.sym.hypot(x, 4)
    >>> z.eval(x=mx.nd.array([3,5,2]))[0].asnumpy()
    array([ 5.,  6.40312433,  4.47213602], dtype=float32)
    >>> z = mx.sym.hypot(x, y)
    >>> z.eval(x=mx.nd.array([3,4]), y=mx.nd.array([10,2]))[0].asnumpy()
    array([ 10.44030666,   4.47213602], dtype=float32)
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


def eye(N, M=0, k=0, dtype=None, **kwargs):
    """Returns a new symbol of 2-D shpae, filled with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N: int
        Number of rows in the output.
    M: int, optional
        Number of columns in the output. If 0, defaults to N.
    k: int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal,
        and a negative value to a lower diagonal.
    dtype : str or numpy.dtype, optional
        The value type of the inner value, default to ``np.float32``.

    Returns
    -------
    out : Symbol
        The created Symbol.
    """
    if dtype is None:
        dtype = _numpy.float32
    return _internal._eye(N, M, k, dtype=dtype, **kwargs)

def zeros(shape, dtype=None, **kwargs):
    """Returns a new symbol of given shape and type, filled with zeros.

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
    """Returns a new symbol of given shape and type, filled with ones.

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


def full(shape, val, dtype=None, **kwargs):
    """Returns a new array of given shape and type, filled with the given value `val`.

    Parameters
    ----------
    shape :  int or sequence of ints
        Shape of the new array.
    val : scalar
        Fill value.
    dtype : str or numpy.dtype, optional
        The value type of the inner value, default to ``np.float32``.

    Returns
    -------
    out : Symbol
        The created Symbol
    """
    if dtype is None:
        dtype = _numpy.float32
    return _internal._full(shape=shape, dtype=dtype, value=float(val), **kwargs)

# pylint: disable=redefined-outer-name
def arange(start, stop=None, step=1.0, repeat=1, infer_range=False, name=None, dtype=None):
    """Returns evenly spaced values within a given interval.

    Values are generated within the half-open interval [`start`, `stop`). In other
    words, the interval includes `start` but excludes `stop`. The function is
    similar to the built-in Python function `range` and to `numpy.arange`,
    but returns a `Symbol`.

    Parameters
    ----------
    start : number, optional
        Start of interval. The interval includes this value. The default start value is 0.
    stop : number
        End of interval. The interval does not include this value.
    step : number, optional
        Spacing between values.
    repeat : int, optional
        "The repeating time of all elements.
        E.g repeat=3, the element a will be repeated three times --> a, a, a.
    infer_range : boolean, optional
        When set to True, infer the stop position from the start, step,
        repeat, and output tensor size.
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
                             infer_range=infer_range, name=name, dtype=dtype)

def linspace(start, stop, num, endpoint=True, name=None, dtype=None):
    """Return evenly spaced numbers within a specified interval.

    Values are generated within the half-open interval [`start`, `stop`) or
    closed interval [start, stop] depending on whether `endpoint` is True or
    False. The function is similar to `numpy.linspace`, but returns a `Symbol`.

    Parameters
    ----------
    start : number
        Start of interval.
    stop : number
        End of interval, unless endpoint is set to False.  In that case,
        the sequence consists of all but the last of `num + 1` evenly spaced
        samples, so that stop is excluded. Note that the step size changes
        when endpoint is False.
    num : number
        Number of samples to generate. Must be non-negative.
    endpoint : bool
        If True, stop is the last sample. Otherwise, it is not included.
        The default is True.
    ctx : Context, optional
        Device context. Default context is the current default context.
    dtype : str or numpy.dtype, optional
        The data type of the `NDArray`. The default datatype is `np.float32`.

    Returns
    -------
    out : Symbol
        The created Symbol
    """
    if dtype is None:
        dtype = _numpy.float32
    return _internal._linspace(start=start, stop=stop, num=num, endpoint=endpoint,
                               name=name, dtype=dtype)

def histogram(a, bins=10, range=None, **kwargs):
    """Compute the histogram of the input data.

    Parameters
    ----------
    a : NDArray
        Input data. The histogram is computed over the flattened array.
    bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the
        given range (10, by default). If bins is a sequence, it defines the bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.
    range : (float, float), required if bins is an integer
        The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()).
        Values outside the range are ignored. The first element of the range must be less than or
        equal to the second. range affects the automatic bin computation as well, the range will
        be equally divided by the number of bins.

    Returns
    -------
    out : Symbol
        The created Symbol
    """
    if isinstance(bins, Symbol):
        return _internal._histogram(data=a, bins=bins, **kwargs)
    elif isinstance(bins, integer_types):
        if range is None:
            raise ValueError("null range is not supported in symbol mode")
        return _internal._histogram(data=a, bin_cnt=bins, range=range, **kwargs)
    raise ValueError("bins argument should be either an integer or an NDArray")

def split_v2(ary, indices_or_sections, axis=0, squeeze_axis=False):
    """Split an array into multiple sub-arrays.

    Parameters
    ----------
    ary : NDArray
        Array to be divided into sub-arrays.
    indices_or_sections : int or tuple of ints
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.
        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in
        - ary[:2]
        - ary[2:3]
        - ary[3:]
        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.
    squeeze_axis: boolean, optional
        Whether to squeeze the axis of sub-arrays or not, only useful when size
        of the sub-arrays are 1 on the `axis`. Default is False.

    Returns
    -------
    out : Symbol
        The created Symbol
    """
    indices = []
    sections = 0
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
    elif isinstance(indices_or_sections, tuple):
        indices = [0] + list(indices_or_sections)
    else:
        raise ValueError('indices_or_sections must either int or tuple of ints')
    return _internal._split_v2(ary, indices, axis, squeeze_axis, sections)

_set_symbol_class(Symbol)
