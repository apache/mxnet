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
import os
import sys
import functools
import itertools
import inspect
import threading

from .base import _LIB, check_call


def makedirs(d):
    """Create directories recursively if they don't exist. os.makedirs(exist_ok=True) is not
    available in Python2"""
    if sys.version_info[0] < 3:
        from distutils.dir_util import mkpath
        mkpath(d)
    else:
        os.makedirs(d, exist_ok=True)  # pylint: disable=unexpected-keyword-arg


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
    """
    Turns on/off NumPy shape semantics, in which `()` represents the shape of scalar tensors,
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


def wraps_safely(wrapped, assigned=functools.WRAPPER_ASSIGNMENTS):
    """This function is safe version of `functools.wraps` in Python2 which skips wrapping functions
    for the attributes that do not exist."""
    if sys.version_info[0] > 2:
        return functools.wraps(wrapped)
    else:
        return functools.wraps(wrapped,
                               assigned=itertools.ifilter(
                                   functools.partial(hasattr, wrapped), assigned))


def use_np_shape(func):
    """A decorator wrapping a function or class with activated NumPy-shape semantics.
    When `func` is a function, this ensures that the execution of the function is scoped with NumPy
    shape semantics, such as the support for zero-dim and zero size tensors. When
    `func` is a class, it ensures that all the methods, static functions, and properties
    of the class are executed with the NumPy shape semantics.

    Example::
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
        @wraps_safely(func)
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

    def __init__(self, is_np_array):  #pylint: disable=redefined-outer-name
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

    Example::
        import mxnet as mx
        from mxnet import gluon, np


        class TestHybridBlock1(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock1, self).__init__()
                self.w = self.params.get('w', shape=(2, 2))

            def hybrid_forward(self, F, x, w):
                return F.dot(x, w)


        x = mx.nd.ones((2, 2))
        net1 = TestHybridBlock1()
        net1.initialize()
        out = net1.forward(x)
        for _, v in net1.collect_params().items():
            assert type(v.data()) is mx.nd.NDArray
        assert type(out) is mx.nd.NDArray


        @np.use_np_array
        class TestHybridBlock2(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock2, self).__init__()
                self.w = self.params.get('w', shape=(2, 2))

            def hybrid_forward(self, F, x, w):
                return F.np.dot(x, w)


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
        @wraps_safely(func)
        def _with_np_array(*args, **kwargs):
            with np_array(active=True):
                return func(*args, **kwargs)
        return _with_np_array
    else:
        raise TypeError('use_np_array can only decorate classes and callable objects, '
                        'while received a {}'.format(str(type(func))))


def use_np(func):
    """A convenience decorator for wrapping user provided functions and classes in the scope of
    both NumPy-shape and NumPy-array semantics, which means that (1) empty tuples `()` and tuples
    with zeros, such as `(0, 1)`, `(1, 0, 2)`, will be treated as scalar tensors' shapes and
    zero-size tensors' shapes in shape inference functions of operators, instead of as unknown
    in legacy mode; (2) ndarrays of type `mxnet.numpy.ndarray` should be created instead of
    `mx.nd.NDArray`.

    Example::
        import mxnet as mx
        from mxnet import gluon, np


        class TestHybridBlock1(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock1, self).__init__()
                self.w = self.params.get('w', shape=(2, 2))

            def hybrid_forward(self, F, x, w):
                return F.dot(x, w) + F.ones((1,))


        x = mx.nd.ones((2, 2))
        net1 = TestHybridBlock1()
        net1.initialize()
        out = net1.forward(x)
        for _, v in net1.collect_params().items():
            assert type(v.data()) is mx.nd.NDArray
        assert type(out) is mx.nd.NDArray


        @np.use_np
        class TestHybridBlock2(gluon.HybridBlock):
            def __init__(self):
                super(TestHybridBlock2, self).__init__()
                self.w = self.params.get('w', shape=(2, 2))

            def hybrid_forward(self, F, x, w):
                return F.np.dot(x, w) + F.np.ones(())


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
    return use_np_array(use_np_shape(func))
