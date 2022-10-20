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
"""general utility functions"""

import ctypes
import functools
import inspect
import threading
import tempfile
import platform
from contextlib import contextmanager

from struct import calcsize
from .base import (_LIB, check_call, c_str, py_str,
                   numeric_types, integer_types, long,
                   _MAX_VALUE_64_BIT_UNSIGNED_,
                   _MAX_VALUE_64_BIT_SIGNED_,
                   _MAX_VALUE_FLOAT32_REPRESENT_)


_np_ufunc_default_kwargs = {
    'where': True,
    'casting': 'same_kind',
    'order': 'K',
    'dtype': None,
    'subok': True,
}

_set_np_shape_logged = False
_set_np_array_logged = False
_set_np_default_dtype_logged = False


def get_gpu_count():
    size = ctypes.c_int()
    check_call(_LIB.MXGetGPUCount(ctypes.byref(size)))
    return size.value


def get_gpu_memory(gpu_dev_id):
    free_mem = ctypes.c_uint64(0)
    total_mem = ctypes.c_uint64(0)
    check_call(_LIB.MXGetGPUMemoryInformation64(gpu_dev_id, ctypes.byref(free_mem), ctypes.byref(total_mem)))
    return free_mem.value, total_mem.value


def set_np_shape(active):
    """Turns on/off NumPy shape semantics, in which `()` represents the shape of scalar tensors,
    and tuples with `0` elements, for example, `(0,)`, `(1, 0, 2)`, represent the shapes
    of zero-size tensors. This is turned off by default for keeping backward compatibility.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy within this semantics.

    Parameters
    ----------
    active : bool
        Indicates whether to turn on/off NumPy shape semantics.

    Returns
    -------
        A bool value indicating the previous state of NumPy shape semantics.

    Example
    -------
    >>> import mxnet as mx
    >>> prev_state = mx.set_np_shape(True)
    >>> print(prev_state)
    False
    >>> print(mx.is_np_shape())
    True
    """
    global _set_np_shape_logged
    if active:
        if not _set_np_shape_logged:
            import logging
            logging.info('NumPy-shape semantics has been activated in your code. '
                         'This is required for creating and manipulating scalar and zero-size '
                         'tensors, which were not supported in MXNet before, as in the official '
                         'NumPy library. Please DO NOT manually deactivate this semantics while '
                         'using `mxnet.numpy` and `mxnet.numpy_extension` modules.')
            _set_np_shape_logged = True
    elif is_np_array():
        raise ValueError('Deactivating NumPy shape semantics while NumPy array semantics is still'
                         ' active is not allowed. Please consider calling `npx.reset_np()` to'
                         ' deactivate both of them.')
    prev = ctypes.c_int()
    check_call(_LIB.MXSetIsNumpyShape(ctypes.c_int(active), ctypes.byref(prev)))
    return bool(prev.value)


def is_np_shape():
    """Checks whether the NumPy shape semantics is currently turned on.
    In NumPy shape semantics, `()` represents the shape of scalar tensors,
    and tuples with `0` elements, for example, `(0,)`, `(1, 0, 2)`, represent
    the shapes of zero-size tensors. This is turned off by default for keeping
    backward compatibility.

    In the NumPy shape semantics, `-1` indicates an unknown size. For example,
    `(-1, 2, 2)` means that the size of the first dimension is unknown. Its size
    may be inferred during shape inference.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy within this semantics.

    Returns
    -------
        A bool value indicating whether the NumPy shape semantics is currently on.

    Example
    -------
    >>> import mxnet as mx
    >>> prev_state = mx.set_np_shape(True)
    >>> print(prev_state)
    False
    >>> print(mx.is_np_shape())
    True
    """
    curr = ctypes.c_bool()
    check_call(_LIB.MXIsNumpyShape(ctypes.byref(curr)))
    return curr.value


class _NumpyShapeScope(object):
    """Scope for managing NumPy shape semantics.
    In NumPy shape semantics, `()` represents the shape of scalar tensors,
    and tuples with `0` elements, for example, `(0,)`, `(1, 0, 2)`, represent
    the shapes of zero-size tensors.

    Do not use this class directly. Use `np_shape(active)` instead.

    Example::

        with _NumpyShapeScope(True):
            y = model(x)
            backward([y])

    """
    def __init__(self, is_np_shape):  #pylint: disable=redefined-outer-name
        self._enter_is_np_shape = is_np_shape
        self._prev_is_np_shape = None

    def __enter__(self):
        if self._enter_is_np_shape is not None:
            self._prev_is_np_shape = set_np_shape(self._enter_is_np_shape)

    def __exit__(self, ptype, value, trace):
        if self._enter_is_np_shape is not None and self._prev_is_np_shape != self._enter_is_np_shape:
            set_np_shape(self._prev_is_np_shape)


def np_shape(active=True):
    """Returns an activated/deactivated NumPy shape scope to be used in 'with' statement
    and captures code that needs the NumPy shape semantics, i.e. support of scalar and
    zero-size tensors.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy even within this scope.

    Parameters
    ----------
    active : bool
        Indicates whether to activate NumPy-shape semantics.

    Returns
    -------
    _NumpyShapeScope
        A scope object for wrapping the code w/ or w/o NumPy-shape semantics.

    Example::

        with mx.np_shape(active=True):
            # A scalar tensor's shape is `()`, whose `ndim` is `0`.
            scalar = mx.nd.ones(shape=())
            assert scalar.shape == ()

            # If NumPy shape semantics is enabled, 0 in a shape means that
            # dimension contains zero elements.
            data = mx.sym.var("data", shape=(0, 2, 3))
            ret = mx.sym.sin(data)
            arg_shapes, out_shapes, _ = ret.infer_shape()
            assert arg_shapes[0] == (0, 2, 3)
            assert out_shapes[0] == (0, 2, 3)

            # -1 means unknown shape dimension size in the new NumPy shape definition
            data = mx.sym.var("data", shape=(-1, 2, 3))
            ret = mx.sym.sin(data)
            arg_shapes, out_shapes, _ = ret.infer_shape_partial()
            assert arg_shapes[0] == (-1, 2, 3)
            assert out_shapes[0] == (-1, 2, 3)

            # When a shape is completely unknown when NumPy shape semantics is on, it is
            # represented as `None` in Python.
            data = mx.sym.var("data")
            ret = mx.sym.sin(data)
            arg_shapes, out_shapes, _ = ret.infer_shape_partial()
            assert arg_shapes[0] is None
            assert out_shapes[0] is None

        with mx.np_shape(active=False):
            # 0 means unknown shape dimension size in the legacy shape definition.
            data = mx.sym.var("data", shape=(0, 2, 3))
            ret = mx.sym.sin(data)
            arg_shapes, out_shapes, _ = ret.infer_shape_partial()
            assert arg_shapes[0] == (0, 2, 3)
            assert out_shapes[0] == (0, 2, 3)

            # When a shape is completely unknown in the legacy mode (default), its ndim is
            # equal to 0 and it is represented as `()` in Python.
            data = mx.sym.var("data")
            ret = mx.sym.sin(data)
            arg_shapes, out_shapes, _ = ret.infer_shape_partial()
            assert arg_shapes[0] == ()
            assert out_shapes[0] == ()
    """
    return _NumpyShapeScope(active)


def use_np_shape(func):
    """A decorator wrapping a function or class with activated NumPy-shape semantics.
    When `func` is a function, this ensures that the execution of the function is scoped with NumPy
    shape semantics, such as the support for zero-dim and zero size tensors. When
    `func` is a class, it ensures that all the methods, static functions, and properties
    of the class are executed with the NumPy shape semantics.

    .. code-block:: python

        import mxnet as mx
        @mx.use_np_shape
        def scalar_one():
            return mx.nd.ones(())
        print(scalar_one())

        @np.use_np_shape
        class ScalarTensor(object):
            def __init__(self, val=None):
                if val is None:
                    val = ScalarTensor.random().value
                self._scalar = mx.nd.ones(()) * val

            def __repr__(self):
                print("Is __repr__ in np_shape semantics? {}!".format(str(np.is_np_shape())))
                return str(self._scalar.asnumpy())

            @staticmethod
            def random():
                val = mx.nd.random.uniform().asnumpy().item()
                return ScalarTensor(val)

            @property
            def value(self):
                print("Is value property in np_shape semantics? {}!".format(str(np.is_np_shape())))
                return self._scalar.asnumpy().item()

        print("Is global scope of np_shape activated? {}!".format(str(np.is_np_shape())))
        scalar_tensor = ScalarTensor()
        print(scalar_tensor)


    Parameters
    ----------
    func : a user-provided callable function or class to be scoped by the NumPy-shape semantics.

    Returns
    -------
    Function or class
        A function or class wrapped in the NumPy-shape scope.
    """

    if inspect.isclass(func):
        for name, method in inspect.getmembers(
                func,
                predicate=
                lambda f: inspect.isfunction(f) or inspect.ismethod(f) or isinstance(f, property)):
            if isinstance(method, property):
                setattr(func, name, property(use_np_shape(method.__get__),
                                             method.__set__,
                                             method.__delattr__,
                                             method.__doc__))
            else:
                setattr(func, name, use_np_shape(method))
        return func
    elif callable(func):
        @functools.wraps(func)
        def _with_np_shape(*args, **kwargs):
            with np_shape(active=True):
                return func(*args, **kwargs)
        return _with_np_shape
    else:
        raise TypeError('use_np_shape can only decorate classes and callable objects, '
                        'while received a {}'.format(str(type(func))))


def _sanity_check_params(func_name, unsupported_params, param_dict):
    for param_name in unsupported_params:
        if param_name in param_dict:
            raise NotImplementedError("function {} does not support parameter {}"
                                      .format(func_name, param_name))


def set_module(module):
    """Decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module('mxnet.numpy')
        def example():
            pass

        assert example.__module__ == 'numpy'
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator


class _NumpyArrayScope(object):
    """Scope for managing NumPy array creation. This is often used
    with `is_np_array=True` in initializer to enforce array creation
    as type `mxnet.numpy.ndarray`, instead of `mx.nd.NDArray` in Gluon.

    Do not use this class directly. Use `np_array(active)` instead.
    """
    _current = threading.local()

    def __init__(self, is_np_array):  # pylint: disable=redefined-outer-name
        self._old_scope = None
        self._is_np_array = is_np_array

    def __enter__(self):
        if not hasattr(_NumpyArrayScope._current, "value"):
            _NumpyArrayScope._current.value = _NumpyArrayScope(False)
        self._old_scope = _NumpyArrayScope._current.value
        _NumpyArrayScope._current.value = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope
        _NumpyArrayScope._current.value = self._old_scope


def np_array(active=True):
    """Returns an activated/deactivated NumPy-array scope to be used in 'with' statement
    and captures code that needs the NumPy-array semantics.

    Currently, this is used in Gluon to enforce array creation in `Block`s as type
    `mxnet.numpy.ndarray`, instead of `mx.nd.NDArray`.

    It is recommended to use the decorator `use_np_array` to decorate the classes
    that need this semantics, instead of using this function in a `with` statement
    unless you know exactly what has been scoped by this semantics.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy even within this scope.

    Parameters
    ----------
    active : bool
        Indicates whether to activate NumPy-array semantics.

    Returns
    -------
    _NumpyShapeScope
        A scope object for wrapping the code w/ or w/o NumPy-shape semantics.
    """
    return _NumpyArrayScope(active)


def is_np_array():
    """Checks whether the NumPy-array semantics is currently turned on.
    This is currently used in Gluon for checking whether an array of type `mxnet.numpy.ndarray`
    or `mx.nd.NDArray` should be created. For example, at the time when a parameter
    is created in a `Block`, an `mxnet.numpy.ndarray` is created if this returns true; else
    an `mx.nd.NDArray` is created.

    Normally, users are not recommended to use this API directly unless you known exactly
    what is going on under the hood.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy within this semantics.

    Returns
    -------
        A bool value indicating whether the NumPy-array semantics is currently on.
    """
    return _NumpyArrayScope._current.value._is_np_array if hasattr(
        _NumpyArrayScope._current, "value") else False


def use_np_array(func):
    """A decorator wrapping Gluon `Block`s and all its methods, properties, and static functions
    with the semantics of NumPy-array, which means that where ndarrays are created,
    `mxnet.numpy.ndarray`s should be created, instead of legacy ndarrays of type `mx.nd.NDArray`.
    For example, at the time when a parameter is created in a `Block`, an `mxnet.numpy.ndarray`
    is created if it's decorated with this decorator.

    .. code-block:: python

        import mxnet as mx
        from mxnet import gluon, nd, np
        from mxnet.gluon import Parameter

        class TestHybridBlock1(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock1, self).__init__()
                self.w = Parameter('w', shape=(2, 2))

            def forward(self, x):
                return nd.dot(x, self.w.data())

        x = mx.nd.ones((2, 2))
        net1 = TestHybridBlock1()
        net1.initialize()
        out = net1.forward(x)
        for _, v in net1.collect_params().items():
            assert type(v.data()) is mx.nd.NDArray
        assert type(out) is mx.nd.NDArray

        @mx.util.use_np_array
        class TestHybridBlock2(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock2, self).__init__()
                self.w = Parameter('w', shape=(2, 2))

            def forward(self, x):
                return np.dot(x, self.w.data())

        x = np.ones((2, 2))
        net2 = TestHybridBlock2()
        net2.initialize()
        out = net2.forward(x)
        for _, v in net2.collect_params().items():
            print(type(v.data()))
            assert type(v.data()) is np.ndarray
        assert type(out) is np.ndarray

    Parameters
    ----------
    func : a user-provided callable function or class to be scoped by the NumPy-array semantics.

    Returns
    -------
    Function or class
        A function or class wrapped in the NumPy-array scope.
    """
    if inspect.isclass(func):
        for name, method in inspect.getmembers(
                func,
                predicate=
                lambda f: inspect.isfunction(f) or inspect.ismethod(f) or isinstance(f, property)):
            if isinstance(method, property):
                setattr(func, name, property(use_np_array(method.__get__),
                                             method.__set__,
                                             method.__delattr__,
                                             method.__doc__))
            else:
                setattr(func, name, use_np_array(method))
        return func
    elif callable(func):
        @functools.wraps(func)
        def _with_np_array(*args, **kwargs):
            with np_array(active=True):
                return func(*args, **kwargs)
        return _with_np_array
    else:
        raise TypeError('use_np_array can only decorate classes and callable objects, '
                        'while received a {}'.format(str(type(func))))


def use_np(func):
    """A convenience decorator for wrapping user provided functions and classes in the scope of
    both NumPy-shape and NumPy-array semantics, which means that ``(1)`` empty tuples ``()`` and
    tuples with zeros, such as ``(0, 1)``, ``(1, 0, 2)``, will be treated as scalar tensors' shapes and
    zero-size tensors' shapes in shape inference functions of operators, instead of as unknown
    in legacy mode; (2) ndarrays of type :class:`mxnet.numpy.ndarray` should be created instead of
    :class:`mx.nd.NDArray`.

    .. code-block:: python

        import mxnet as mx
        from mxnet import gluon, nd, np
        from mxnet.gluon import Parameter

        class TestHybridBlock1(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock1, self).__init__()
                self.w = Parameter('w', shape=(2, 2))

            def forward(self, x):
                return nd.dot(x, self.w.data()) + nd.ones((1,))

        x = mx.nd.ones((2, 2))
        net1 = TestHybridBlock1()
        net1.initialize()
        out = net1.forward(x)
        for _, v in net1.collect_params().items():
            assert type(v.data()) is mx.nd.NDArray
        assert type(out) is mx.nd.NDArray

        @mx.util.use_np
        class TestHybridBlock2(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock2, self).__init__()
                self.w = Parameter('w', shape=(2, 2))

            def forward(self, x):
                return np.dot(x, self.w.data()) + np.ones(())

        x = np.ones((2, 2))
        net2 = TestHybridBlock2()
        net2.initialize()
        out = net2.forward(x)
        for _, v in net2.collect_params().items():
            print(type(v.data()))
            assert type(v.data()) is np.ndarray
        assert type(out) is np.ndarray


    Parameters
    ----------
    func : a user-provided callable function or class to be scoped by the
        NumPy-shape and NumPy-array semantics.

    Returns
    -------
    Function or class
        A function or class wrapped in the Numpy-shape and NumPy-array scope.
    """
    return use_np_shape(use_np_array(func))


def np_ufunc_legal_option(key, value):
    """Checking if ufunc arguments are legal inputs

    Parameters
    ----------
    key : string
        the key of the ufunc argument.
    value : string
        the value of the ufunc argument.

    Returns
    -------
    legal : boolean
        Whether or not the argument is a legal one. True when the key is one of the ufunc
        arguments and value is an allowed value. False when the key is not one of the ufunc
        arugments or the value is not an allowed value even when the key is a legal one.
    """
    if key == 'where':
        return True
    elif key == 'casting':
        return (value in set(['no', 'equiv', 'safe', 'same_kind', 'unsafe']))
    elif key == 'order':
        if isinstance(value, str):
            return True
    elif key == 'dtype':
        import numpy as _np
        return (value in set([_np.int8, _np.uint8, _np.int32, _np.int64,
                              _np.float16, _np.float32, _np.float64,
                              'int8', 'uint8', 'int32', 'int64',
                              'float16', 'float32', 'float64']))
    elif key == 'subok':
        return isinstance(value, bool)
    return False


def wrap_np_unary_func(func):
    """A convenience decorator for wrapping numpy-compatible unary ufuncs to provide uniform
    error handling.

    Parameters
    ----------
    func : a numpy-compatible unary function to be wrapped for better error handling.

    Returns
    -------
    Function
        A function wrapped with proper error handling.
    """
    @functools.wraps(func)
    def _wrap_np_unary_func(x, out=None, **kwargs):
        if len(kwargs) != 0:
            for key, value in kwargs.items():
                # if argument is not in the set of ufunc arguments
                if key not in _np_ufunc_default_kwargs:
                    raise TypeError("{} is an invalid keyword to function \'{}\'".format(key, func.__name__))
                # if argument is one of the ufunc arguments, but not with the default value
                if value != _np_ufunc_default_kwargs[key]:
                    # if the provided value of the argument is a legal option, raise NotImplementedError
                    if np_ufunc_legal_option(key, value):
                        raise NotImplementedError("{}={} is not implemented yet for operator {}"
                                                  .format(key, str(value), func.__name__))
                    # otherwise raise TypeError with not understood error message
                    raise TypeError("{}={} not understood for operator {}"
                                    .format(key, value, func.__name__))
        return func(x, out=out)
    return _wrap_np_unary_func


def wrap_np_binary_func(func):
    """A convenience decorator for wrapping numpy-compatible binary ufuncs to provide uniform
    error handling.

    Parameters
    ----------
    func : a numpy-compatible binary function to be wrapped for better error handling.

    Returns
    -------
    Function
        A function wrapped with proper error handling.
    """
    @functools.wraps(func)
    def _wrap_np_binary_func(x1, x2, out=None, **kwargs):
        if len(kwargs) != 0:
            for key, value in kwargs.items():
                # if argument is not in the set of ufunc arguments
                if key not in _np_ufunc_default_kwargs:
                    raise TypeError("{} is an invalid keyword to function \'{}\'".format(key, func.__name__))
                # if argument is one of the ufunc arguments, but not with the default value
                if value != _np_ufunc_default_kwargs[key]:
                    # if the provided value of the argument is a legal option, raise NotImplementedError
                    if np_ufunc_legal_option(key, value):
                        raise NotImplementedError("{}={} is not implemented yet".format(key, str(value)))
                    # otherwise raise TypeError with not understood error message
                    raise TypeError("{} {} not understood".format(key, value))
        return func(x1, x2, out=out)
    return _wrap_np_binary_func

def wrap_data_api_statical_func(func):
    """
    A convenience decorator for wrapping data apis standardized statical functions to provide
    context keyward backward compatibility
    Parameters
    ----------
    func : a numpy-compatible array statical function to be wrapped for context keyward change.
    Returns
    -------
    Function
    A function wrapped with context keyward changes.
    """

    @functools.wraps(func)
    def _wrap_api_creation_func(*args, **kwargs):
        if len(kwargs) != 0:
            correction = kwargs.pop('ddof', None)
            if correction is not None:
                kwargs['correction'] = correction
        return func(*args, **kwargs)

    return _wrap_api_creation_func

def wrap_data_api_linalg_func(func):
    """
    A convenience decorator for wrapping data apis standardized linalg functions to provide
    context keyward backward compatibility
    Parameters
    ----------
    func : a numpy-compatible array linalg function to be wrapped for context keyward change.
    Returns
    -------
    Function
    A function wrapped with context keyward changes.
    """

    @functools.wraps(func)
    def _wrap_linalg_func(*args, **kwargs):
        if len(kwargs) != 0:
            upper = kwargs.pop('UPLO', None)
            rcond = kwargs.pop('rcond', None)
            tol = kwargs.pop('tol', None)
            if upper is not None:
                if upper == 'U':
                    kwargs['upper'] = True
                else:
                    kwargs['upper'] = False
            if rcond is not None:
                kwargs['rtol'] = rcond
            if tol is not None:
                kwargs['rtol'] = tol
        return func(*args, **kwargs)

    return _wrap_linalg_func


def wrap_sort_functions(func):
    """A convenience decorator for wrapping sort functions

    Parameters
    ----------
    func : a numpy-compatible array creation function to be wrapped for parameter keyword change.

    Returns
    -------
    Function
        A function wrapped with changed keywords.
    """
    @functools.wraps(func)
    def _wrap_sort_func(*args, **kwargs):
        if len(kwargs) != 0:
            kind = kwargs.pop('kind', None)
            order = kwargs.pop('order', None)
            if kind is not None:
                kwargs['stable'] = kind == 'stable'
            if order is not None:
                raise NotImplementedError("order not supported here")
        return func(*args, **kwargs)
    return _wrap_sort_func


def wrap_ctx_to_device_func(func):
    """A convenience decorator for converting ctx to device keyward backward compatibility

    Parameters
    ----------
    func : a function to be wrapped for context keyward change.

    Returns
    -------
    Function
        A function wrapped with context keyward changes.
    """
    @functools.wraps(func)
    def _wrap_func_with_ctx(*args, **kwargs):
        if len(kwargs) != 0:
            device = kwargs.pop('ctx', None)
            if device is not None:
                kwargs['device'] = device
        return func(*args, **kwargs)
    return _wrap_func_with_ctx


# pylint: disable=exec-used
def numpy_fallback(func):
    """decorator for falling back to offical numpy for a specific function"""
    def get_device(device, new_device):
        if device is None:
            return new_device
        else:
            if new_device is None:
                new_device = device
            assert device == new_device, f"inconsistent device {str(device)} and {str(new_device)}"
            return device

    def _as_official_np_array(object):
        device = None
        if hasattr(object, 'asnumpy'):
            return object.asnumpy(), object.device
        elif isinstance(object, (list, tuple)):
            tmp = []
            for arr in object:
                new_arr, new_device = _as_official_np_array(arr)
                device = get_device(device, new_device)
                tmp.append(new_arr)
            return object.__class__(tmp), device
        elif isinstance(object, dict):
            tmp = {}
            for k, v in object.items():
                new_v, new_device = _as_official_np_array(v)
                device = get_device(device, new_device)
                tmp[k] = new_v
            return tmp, device
        else:
            return object, None

    from .ndarray import from_numpy
    from .numpy import array
    from .device import current_device
    def _as_mx_np_array(object, device=current_device()):
        import numpy as _np
        if isinstance(object, _np.ndarray):
            try:
                ret = from_numpy(object).as_np_ndarray()
            except ValueError:
                ret = array(object, dtype=object.dtype, device=device)
            return (ret if ('cpu' in str(device)) else ret.to_device(device))
        elif isinstance(object, (list, tuple)):
            tmp = [_as_mx_np_array(arr, device) for arr in object]
            return object.__class__(tmp)
        elif isinstance(object, dict):
            return {k:_as_mx_np_array(v, device) for k, v in object}
        else:
            return object

    import re
    func_name = func.__name__
    func_doc = func.__doc__
    func_source = inspect.getsource(func)
    func_source = re.sub(r'np\.', 'onp.', func_source)
    func_source = func_source.split('\n')[1:]
    indentation = func_source[0].find('def')
    if indentation == -1:
        raise ValueError("should wrap a function")
    stripped = []
    for line in func_source:
        stripped.append(line[indentation:])
    stripped.insert(1, '    import numpy as onp')
    func_source = '\n'.join(stripped)
    local = {}
    exec(func_source, None, local)
    func = local[func_name]
    func.__doc__ = func_doc

    @functools.wraps(func)
    def _fallback_to_official_np(*args, **kwargs):
        # for every ndarray input, fallback
        new_args, device0 = _as_official_np_array(args)
        new_kwargs, device1 = _as_official_np_array(kwargs)
        device = get_device(device0, device1)
        ret = func(*new_args, **new_kwargs)
        if ret is None:
            raise ValueError("Only functions with return values are allowed to use this decorator")
        ret = _as_mx_np_array(ret, device=device)
        return ret

    return _fallback_to_official_np
# pylint: enable=exec-used


def _set_np_array(active):
    """Turns on/off NumPy array semantics for the current thread in which `mxnet.numpy.ndarray`
    is expected to be created, instead of the legacy `mx.nd.NDArray`.

    Parameters
    ---------
    active : bool
        A boolean value indicating whether the NumPy-array semantics should be turned on or off.

    Returns
    -------
        A bool value indicating the previous state of NumPy array semantics.
    """
    global _set_np_array_logged
    if active:
        if not _set_np_array_logged:
            import logging
            logging.info('NumPy array semantics has been activated in your code. This allows you'
                         ' to use operators from MXNet NumPy and NumPy Extension modules as well'
                         ' as MXNet NumPy `ndarray`s.')
            _set_np_array_logged = True
    cur_state = is_np_array()
    _NumpyArrayScope._current.value = _NumpyArrayScope(active)
    return cur_state


def set_np(shape=True, array=True, dtype=False):
    """Setting NumPy shape and array semantics at the same time.
    It is required to keep NumPy shape semantics active while activating NumPy array semantics.
    Deactivating NumPy shape semantics while NumPy array semantics is still active is not allowed.
    It is highly recommended to set these two flags to `True` at the same time to fully enable
    NumPy-like behaviors. Please refer to the Examples section for a better understanding.

    Parameters
    ----------
    shape : bool
        A boolean value indicating whether the NumPy-shape semantics should be turned on or off.
        When this flag is set to `True`, zero-size and zero-dim shapes are all valid shapes in
        shape inference process, instead of treated as unknown shapes in legacy mode.
    array : bool
        A boolean value indicating whether the NumPy-array semantics should be turned on or off.
        When this flag is set to `True`, it enables Gluon code flow to use or generate `mxnet.numpy.ndarray`s
        instead of `mxnet.ndarray.NDArray`. For example, a `Block` would create parameters of type
        `mxnet.numpy.ndarray`.
    dtype : bool
         A boolean value indicating whether the NumPy-dtype semantics should be turned on or off.
         When this flag is set to `True`, default dtype is float64.
         When this flag is set to `False`, default dtype is float32.
    Examples
    --------
    >>> import mxnet as mx

    Creating zero-dim ndarray in legacy mode would fail at shape inference.

    >>> mx.nd.ones(shape=())
    mxnet.base.MXNetError: Operator _ones inferring shapes failed.

    >>> mx.nd.ones(shape=(2, 0, 3))
    mxnet.base.MXNetError: Operator _ones inferring shapes failed.

    In legacy mode, Gluon layers would create parameters and outputs of type `mx.nd.NDArray`.

    >>> from mxnet.gluon import nn
    >>> dense = nn.Dense(2)
    >>> dense.initialize()
    >>> dense(mx.nd.ones(shape=(3, 2)))
    [[0.01983214 0.07832371]
     [0.01983214 0.07832371]
     [0.01983214 0.07832371]]
    <NDArray 3x2 @cpu(0)>

    >>> [p.data() for p in dense.collect_params().values()]
    [
    [[0.0068339  0.01299825]
     [0.0301265  0.04819721]]
    <NDArray 2x2 @cpu(0)>,
    [0. 0.]
    <NDArray 2 @cpu(0)>]

    When the `shape` flag is `True`, both shape inferences are successful.

    >>> from mxnet import np, npx
    >>> npx.set_np()  # this is required to activate NumPy-like behaviors

    >>> np.ones(shape=())
    array(1.)
    >>> np.ones(shape=(2, 0, 3))
    array([], shape=(2, 0, 3))

    When the `array` flag is `True`, Gluon layers would create parameters and outputs of type `mx.np.ndarray`.

    >>> dense = nn.Dense(2)
    >>> dense.initialize()
    >>> dense(np.ones(shape=(3, 2)))
    array([[0.01983214, 0.07832371],
           [0.01983214, 0.07832371],
           [0.01983214, 0.07832371]])

    >>> [p.data() for p in dense.collect_params().values()]
    [array([[0.0068339 , 0.01299825],
           [0.0301265 , 0.04819721]]), array([0., 0.])]

    >>> npx.set_np(dtype=True)
    >>> np.ones(shape=()).dtype
    dtype('float64')
    """
    if not shape and array:
        raise ValueError('NumPy Shape semantics is required in using NumPy array semantics.')
    _set_np_array(array)
    set_np_shape(shape)
    set_np_default_dtype(dtype)


def reset_np():
    """Deactivate NumPy shape and array and deafult dtype semantics at the same time."""
    set_np(shape=False, array=False, dtype=False)


_CUDA_SUCCESS = 0


def get_cuda_compute_capability(device):
    """Returns the cuda compute capability of the input `device`.

    Parameters
    ----------
    device : Device
        GPU context whose corresponding cuda compute capability is to be retrieved.

    Returns
    -------
    cuda_compute_capability : int
        CUDA compute capability. For example, it returns 70 for CUDA arch equal to `sm_70`.

    References
    ----------
    https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549#file-cuda_check-py
    """
    if device.device_type != 'gpu':
        raise ValueError('Expecting a gpu context to get cuda compute capability, '
                         'while received device {}'.format(str(device)))

    libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + ' '.join(libnames))

    # Some constants taken from cuda.h

    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cuda_device = ctypes.c_int()
    error_str = ctypes.c_char_p()

    ret = cuda.cuInit(0)
    if ret != _CUDA_SUCCESS:
        cuda.cuGetErrorString(ret, ctypes.byref(error_str))
        raise RuntimeError('cuInit failed with erro code {}: {}'
                           .format(ret, error_str.value.decode()))

    ret = cuda.cuDeviceGet(ctypes.byref(cuda_device), device.device_id)
    if ret != _CUDA_SUCCESS:
        cuda.cuGetErrorString(ret, ctypes.byref(error_str))
        raise RuntimeError('cuDeviceGet failed with error code {}: {}'
                           .format(ret, error_str.value.decode()))
    ret = cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), cuda_device)
    if ret != _CUDA_SUCCESS:
        cuda.cuGetErrorString(ret, ctypes.byref(error_str))
        raise RuntimeError('cuDeviceComputeCapability failed with error code {}: {}'
                           .format(ret, error_str.value.decode()))
    return cc_major.value * 10 + cc_minor.value


def default_array(source_array, device=None, dtype=None):
    """Creates an array from any object exposing the default(nd or np) array interface.

    Parameters
    ----------
    source_array : array_like
        An object exposing the array interface, an object whose `__array__`
        method returns an array, or any (nested) sequence.
    device : Device, optional
        Device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        The data type of the output array. The default dtype is ``source_array.dtype``
        if `source_array` is an `NDArray`, `float32` otherwise.

    Returns
    -------
    NDArray
        An `NDArray`(nd or np) with the same contents as the `source_array`.
    """
    from . import nd as _mx_nd
    from . import np as _mx_np
    if is_np_array():
        return _mx_np.array(source_array, device=device, dtype=dtype)
    else:
        return _mx_nd.array(source_array, ctx=device, dtype=dtype)

class _NumpyDefaultDtypeScope(object):
    """Scope for managing NumPy default dtype semantics.
    In NumPy default dtype semantics, default dtype is 'float64',
    i.e. np.array([1, 2, 3]).dtype = np.float64
    Original default dtype without this semantic is 'float32'.

    Do not use this class directly. Use `np_shape(active)` instead.

    Example::

        with _NumpyDefaultDtypeScope(True):
            y = model(x)
            backward([y])

    """
    def __init__(self, is_np_default_dtype):  #pylint: disable=redefined-outer-name
        self._enter_is_np_default_dtype = is_np_default_dtype
        self._prev_is_np_default_dtype = None

    def __enter__(self):
        if self._enter_is_np_default_dtype is not None:
            self._prev_is_np_default_dtype = set_np_default_dtype(self._enter_is_np_default_dtype)

    def __exit__(self, ptype, value, trace):
        if self._enter_is_np_default_dtype is not None and\
           self._prev_is_np_default_dtype != self._enter_is_np_default_dtype:
            set_np_default_dtype(self._prev_is_np_default_dtype)

def np_default_dtype(active=True):
    """Returns an activated/deactivated NumPy-default_dtype scope to be used in 'with' statement
    and captures code that needs the NumPy default dtype semantics. i.e. default dtype is float64.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy even within this scope.

    Parameters
    ----------
    active : bool
        Indicates whether to activate NumPy default dtype semantics.

    Returns
    -------
    _NumpyDefaultDtypeScope
        A scope object for wrapping the code w/ or w/o NumPy-default_dtype semantics.

    Example::

        with mx.np_default_dtype(active=True):
            # Default Dtype is 'float64', consistent with offical NumPy behavior.
            arr = mx.np.array([1, 2, 3])
            assert arr.dtype == 'float64'

        with mx.np_default_dtype(active=False):
            # Default Dtype is 'float32' in the legacy default dtype definition.
            arr = mx.np.array([1, 2, 3])
            assert arr.dtype == 'float32'

    """
    return _NumpyDefaultDtypeScope(active)

def use_np_default_dtype(func):
    """A decorator wrapping a function or class with activated NumPy-default_dtype semantics.
    When `func` is a function, this ensures that the execution of the function is scoped with NumPy
    default dtype semantics, with the support for float64 as default dtype.
    When`func` is a class, it ensures that all the methods, static functions, and properties
    of the class are executed with the NumPy-default_dtype semantics.

    .. code-block:: python

        import mxnet as mx
        @mx.use_np_default_dtype
        def float64_one():
            return mx.nd.ones(()).dtype
        print(float64_one())

        @np.use_np_default_dtype
        class Float64Tensor(object):
            def __init__(self, data=None):
                if data is None:
                    data = Float64Tensor.random().data
                self._data = data

            def __repr__(self):
                print("Is __repr__ in np_default_dtype semantics? {}!".format(str(np.is_np_deafult_dtype())))
                return str(self._data.asnumpy())

            @staticmethod
            def random():
                data = mx.nd.random.uniform(shape=(2,2))
                return ScalarTensor(data)

            @property
            def value(self):
                print("Is value property in np_dafault_dtype semantics? {}!".format(str(np.is_np_default_dtype())))
                return self._data.asnumpy()

        print("Is global scope of np_default_dtype activated? {}!".format(str(np.is_np_default_dtype())))
        float64_tensor = Float64Tensor()
        print(float64_tensor)


    Parameters
    ----------
    func : a user-provided callable function or class to be scoped by the NumPy-default_dtype semantics.

    Returns
    -------
    Function or class
        A function or class wrapped in the NumPy-default_dtype scope.
    """
    if inspect.isclass(func):
        for name, method in inspect.getmembers(
                func,
                predicate=
                lambda f: inspect.isfunction(f) or inspect.ismethod(f) or isinstance(f, property)):
            if isinstance(method, property):
                setattr(func, name, property(use_np_default_dtype(method.__get__),
                                             method.__set__,
                                             method.__delattr__,
                                             method.__doc__))
            else:
                setattr(func, name, use_np_default_dtype(method))
        return func
    elif callable(func):
        @functools.wraps(func)
        def _with_np_default_dtype(*args, **kwargs):
            with np_default_dtype(active=True):
                return func(*args, **kwargs)
        return _with_np_default_dtype
    else:
        raise TypeError('use_np_default_dtype can only decorate classes and callable objects, '
                        'while received a {}'.format(str(type(func))))

def is_np_default_dtype():
    """Checks whether the NumPy default dtype semantics is currently turned on.
    In NumPy default dtype semantics, default dtype is float64.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy even within this scope.

    Returns
    -------
        A bool value indicating whether the NumPy default dtype semantics is currently on.

    See Also
    --------
    set_np_default_dtype : Set default dtype equals to offical numpy
    set_np : npx.set_np(dtype=True) has equal performance to npx.set_np_default_dtype(True)

    Example
    -------
    >>> import mxnet as mx
    >>> from mxnet import npx
    >>> prev_state = npx.set_np_default_dtype(True)
    >>> print(prev_state)
    False
    >>> print(npx.is_np_default_dtype())
    True
    """
    curr = ctypes.c_bool()
    check_call(_LIB.MXIsNumpyDefaultDtype(ctypes.byref(curr)))
    return curr.value

def set_np_default_dtype(is_np_default_dtype=True):  # pylint: disable=redefined-outer-name
    """Turns on/off NumPy default dtype semantics, because mxnet.numpy.ndarray use
    32 bit data storage as default (e.g. float32 and int 32) while offical NumPy use
    64 bit data storage as default (e.g. float64 and int64).
    This is turned off by default for keeping backward compatibility.

    Please note that this is designed as an infrastructure for the incoming
    MXNet-NumPy operators. Legacy operators registered in the modules
    `mx.nd` and `mx.sym` are not guaranteed to behave like their counterparts
    in NumPy within this semantics.

    Parameters
    ----------
    active : bool
        Indicates whether to turn on/off NumPy default dtype semantics.

    Returns
    -------
        A bool value indicating the previous state of NumPy default dtype semantics.

    Example
    -------
    >>> import mxnet as mx
    >>> from mxnet import npx
    >>> prev_state = npx.set_np_default_dtype(True)
    >>> print(prev_state)
    False
    >>> print(npx.is_np_default_dtype())
    True
    """
    global _set_np_default_dtype_logged
    if is_np_default_dtype:
        if not _set_np_default_dtype_logged:
            import logging
            logging.info('NumPy array default dtype has been changed from flaot32 to float64 in your code.')
            _set_np_default_dtype_logged = True
    prev = ctypes.c_bool()
    check_call(_LIB.MXSetIsNumpyDefaultDtype(ctypes.c_bool(is_np_default_dtype), ctypes.byref(prev)))
    return prev.value


def getenv(name):
    """Get the setting of an environment variable from the C Runtime.

    Parameters
    ----------
    name : string type
        The environment variable name

    Returns
    -------
    value : string
        The value of the environment variable, or None if not set
    """
    ret = ctypes.c_char_p()
    check_call(_LIB.MXGetEnv(c_str(name), ctypes.byref(ret)))
    return None if ret.value is None else py_str(ret.value)


def setenv(name, value):
    """Set an environment variable in the C Runtime.

    Parameters
    ----------
    name : string type
        The environment variable name
    value : string type
        The desired value to set the environment value to
    """
    passed_value = None if value is None else c_str(value)
    check_call(_LIB.MXSetEnv(c_str(name), passed_value))


def get_max_supported_compute_capability():
    """Get the maximum compute capability (SM arch) supported by the nvrtc compiler
    """
    max_supported_cc = ctypes.c_int()
    check_call(_LIB.MXGetMaxSupportedArch(ctypes.byref(max_supported_cc)))
    return max_supported_cc.value


def get_rtc_compile_opts(device):
    """Get the compile ops suitable for the context, given the toolkit/driver config
    """
    device_cc = get_cuda_compute_capability(device)
    max_supported_cc = get_max_supported_compute_capability()

    # CUDA toolkits starting with 11.1 (first to support arch 86) can compile directly to SASS
    can_compile_to_SASS = max_supported_cc >= 86
    should_compile_to_SASS = can_compile_to_SASS and \
                             device_cc <= max_supported_cc
    device_cc_as_used = min(device_cc, max_supported_cc)
    arch_opt = "--gpu-architecture={}_{}".format("sm" if should_compile_to_SASS else "compute",
                                                 device_cc_as_used)
    return [arch_opt]

def set_flush_denorms(value):
    """Change floating-point calculations on CPU when dealing with denormalized values.
       This is only applicable to architectures which supports flush-to-zero.
       Denormalized values are positive and negative values that are very close to 0
       (exponent is the smallest possible value).
       Flushing denormalized values to 0 can speedup calculations if such values occurs,
       but if fulfilling whole IEEE 754 standard is required this option should be disabled.
       Flushing denormalized values is enabled in MXNet by default.

    Parameters
    ----------
    value : bool
        State of flush-to-zero and denormals-are-zero in MXCSR register

    Returns
    -------
    prev_state : bool
        Previous state of flush-to-zero in MXCSR register
    """
    ret = ctypes.c_bool()
    passed_value = ctypes.c_bool(value)
    check_call(_LIB.MXSetFlushDenorms(passed_value, ctypes.byref(ret)))
    return ret.value


def dtype_from_number(number):
    """Get the data type from the given int or float number
    """
    assert isinstance(number, numeric_types),\
        "The input number should be either int for float types"
    import numpy as _np
    if isinstance(number, (int, long)):
        if number > _MAX_VALUE_64_BIT_UNSIGNED_:
            raise OverflowError("Integer out of bounds")
        if number > _MAX_VALUE_64_BIT_SIGNED_:
            return _np.uint64
        elif calcsize("P") == 8:
            return _np.int64
        else:
            return _np.int32
    elif isinstance(number, float):
        if abs(number) > _MAX_VALUE_FLOAT32_REPRESENT_ or \
            ((not _np.isnan(number)) and \
                (_np.float32(number) == int(number)) and \
                    (number != int(number))):
            return _np.float64
        else:
            return _np.float64 if is_np_default_dtype() else _np.float32
    elif isinstance(number, _np.generic):
        return number.dtype
    raise TypeError('type {} not supported'.format(str(type(number))))

# This is a wrapping of tempfile.TemporaryDirectory(), known to have cleanup issues on Windows.
# The problem is partially handled as of Python 3.10 by the adding of a 'ignore_cleanup_errors'
# parameter.  Once MXNet's Python version is forced to be >= 3.10, a simplification of this
# function to use 'ignore_cleanup_errors' would be possible.  Until the fundamental Windows
# issues are resolved, best to use this routine instead of tempfile.TemporaryDirectory().
@contextmanager
def TemporaryDirectory(*args, **kwargs):
    """A context wrapper of tempfile.TemporaryDirectory() that ignores cleanup errors on Windows.
    """
    dir = tempfile.TemporaryDirectory(*args, **kwargs)
    try:
        yield dir.name
    finally:
        try:
            dir.cleanup()
        except PermissionError:
            if platform.system() != 'Windows':
                raise
