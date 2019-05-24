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


def set_np_compat(active):
    """
    Turns on/off NumPy compatibility. NumPy-compatibility is turned off by default in backend.

    Parameters
    ----------
    active : bool
        Indicates whether to turn on/off NumPy compatibility.

    Returns
    -------
        A bool value indicating the previous state of NumPy compatibility.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXSetIsNumpyCompatible(ctypes.c_int(active), ctypes.byref(prev)))
    return bool(prev.value)


def is_np_compat():
    """
    Checks whether the NumPy compatibility is currently turned on.
    NumPy-compatibility is turned off by default in backend.

    Returns
    -------
        A bool value indicating whether the NumPy compatibility is currently on.
    """
    curr = ctypes.c_bool()
    check_call(_LIB.MXIsNumpyCompatible(ctypes.byref(curr)))
    return curr.value


class _NumpyCompatibilityStateScope(object):
    """Scope for managing numpy compatibility state.
    Do not use this class directly. Use `np_compat(active)` instead.

    Example::

        with _NumpyCompatibilityStateScope(True):
            y = model(x)
            backward([y])

    """
    def __init__(self, is_np_compat):  #pylint: disable=redefined-outer-name
        self._enter_is_np_compat = is_np_compat
        self._prev_is_np_compat = None

    def __enter__(self):
        if self._enter_is_np_compat is not None:
            self._prev_is_np_compat = set_np_compat(self._enter_is_np_compat)

    def __exit__(self, ptype, value, trace):
        if self._enter_is_np_compat is not None and self._prev_is_np_compat != self._enter_is_np_compat:
            set_np_compat(self._prev_is_np_compat)


def np_compat(active=True):
    """Returns an activated/deactivated NumPy compatibility state scope to be used in 'with' statement
    and captures code that needs the compatibility.

    Example::

        with mx.np_compat(active=True):
            # A scalar tensor's shape is `()`, whose `ndim` is `0`.
            scalar = mx.nd.ones(shape=())
            assert scalar.shape == ()

            # In NumPy compatible mode, 0 in a shape means that dimension contains zero elements.
            data = mx.sym.var("data", shape=(0, 2, 3))
            ret = mx.sym.sin(data)
            arg_shapes, out_shapes, _ = ret.infer_shape()
            assert arg_shapes[0] == (0, 2, 3)
            assert out_shapes[0] == (0, 2, 3)

            # -1 means unknown shape dimension size in the new NumPy-compatible shape definition
            data = mx.sym.var("data", shape=(-1, 2, 3))
            ret = mx.sym.sin(data)
            arg_shapes, out_shapes, _ = ret.infer_shape_partial()
            assert arg_shapes[0] == (-1, 2, 3)
            assert out_shapes[0] == (-1, 2, 3)

            # When a shape is completely unknown in NumPy-compatible mode, it is
            # represented as `None` in Python.
            data = mx.sym.var("data")
            ret = mx.sym.sin(data)
            arg_shapes, out_shapes, _ = ret.infer_shape_partial()
            assert arg_shapes[0] is None
            assert out_shapes[0] is None

        with mx.np_compat(active=False):
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
    return _NumpyCompatibilityStateScope(active)


def wraps_safely(wrapped, assigned=functools.WRAPPER_ASSIGNMENTS):
    """This function is safe version of `functools.wraps` in Python2 which skips wrapping functions
    for the attributes that do not exist."""
    if sys.version_info[0] > 2:
        return functools.wraps(wrapped)
    else:
        return functools.wraps(wrapped,
                               assigned=itertools.ifilter(
                                   functools.partial(hasattr, wrapped), assigned))


def use_np_compat(func):
    """A decorator wrapping a function or class with an activated NumPy-compatibility scope.
    When `func` is a function, this ensures that the execution of the function is scoped with NumPy
    compatible-semantics, such as the support for zero-dim and zero size tensors. When
    `func` is a class, it ensures that all the methods, static functions, and properties
    of the class are executed with the NumPy-compatible semantics.

    Example::
        import mxnet as mx
        @mx.use_np_compat
        def scalar_one():
            return mx.nd.ones(())
        print(scalar_one())

        @np.use_np_compat
        class ScalarTensor(object):
            def __init__(self, val=None):
                if val is None:
                    val = ScalarTensor.random().value
                self._scalar = mx.nd.ones(()) * val

            def __repr__(self):
                print("Is __repr__ numpy compatible? {}!".format(str(np.is_np_compat())))
                return str(self._scalar.asnumpy())

            @staticmethod
            def random():
                val = mx.nd.random.uniform().asnumpy().item()
                return ScalarTensor(val)

            @property
            def value(self):
                print("Is value property numpy compatible? {}!".format(str(np.is_np_compat())))
                return self._scalar.asnumpy().item()


        print("Is global scope numpy compatible? {}!".format(str(np.is_np_compat())))
        scalar_tensor = ScalarTensor()
        print(scalar_tensor)

    Parameters
    ----------
    func : a user-provided callable function or class to be scoped by the NumPy compatibility state.

    Returns
    -------
    Function or class
        A function or class wrapped in the NumPy compatibility scope.
    """

    if inspect.isclass(func):
        for name, method in inspect.getmembers(
                func,
                predicate=
                lambda f: inspect.isfunction(f) or inspect.ismethod(f) or isinstance(f, property)):
            if isinstance(method, property):
                setattr(func, name, property(use_np_compat(method.__get__),
                                             method.__set__,
                                             method.__delattr__,
                                             method.__doc__))
            else:
                setattr(func, name, use_np_compat(method))
        return func
    elif callable(func):
        @wraps_safely(func)
        def _with_np_compat(*args, **kwargs):
            with np_compat(active=True):
                return func(*args, **kwargs)
        return _with_np_compat
    else:
        raise TypeError('use_np_compat can only decorate classes and callable objects, '
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
