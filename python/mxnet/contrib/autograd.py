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
"""Autograd for NDArray."""
from __future__ import absolute_import
from __future__ import division

from array import array
import ctypes
import functools
from ..base import _LIB, check_call, string_types
from ..base import mx_uint, NDArrayHandle, c_array, c_array_buf, c_handle_array
# pylint: disable= unused-import
from ..ndarray import NDArray, zeros_like, _GRAD_REQ_MAP


def set_is_training(is_train):
    """Set status to training/not training. When training, graph will be constructed
    for gradient computation. Operators will also run with ctx.is_train=True. For example,
    Dropout will drop inputs randomly when is_train=True while simply passing through
    if is_train=False.

    Parameters
    ----------
    is_train: bool

    Returns
    -------
    previous state before this set.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXAutogradSetIsTraining(
        ctypes.c_int(is_train), ctypes.byref(prev)))
    check_call(_LIB.MXAutogradSetIsRecording(
        ctypes.c_int(is_train), ctypes.byref(prev)))
    return bool(prev.value)


class TrainingStateScope(object):
    """Scope for managing training state.

    Example::
        with TrainingStateScope(True):
            y = model(x)
            compute_gradient([y])
    """
    def __init__(self, enter_state):
        self._enter_state = enter_state
        self._prev = None

    def __enter__(self):
        self._prev = set_is_training(self._enter_state)

    def __exit__(self, ptype, value, trace):
        if self._prev != self._enter_state:
            set_is_training(self._prev)


def train_section():
    """Returns a training scope context to be used in 'with' statement
    and captures training code.

    Example::
        with autograd.train_section():
            y = model(x)
            compute_gradient([y])
        metric.update(...)
        optim.step(...)
    """
    return TrainingStateScope(True)


def test_section():
    """Returns a testing scope context to be used in 'with' statement
    and captures testing code.

    Example::
        with autograd.train_section():
            y = model(x)
            compute_gradient([y])
            with autograd.test_section():
                # testing, IO, gradient updates...
    """
    return TrainingStateScope(False)


def mark_variables(variables, gradients, grad_reqs='write'):
    """Mark NDArrays as variables to compute gradient for autograd.

    Parameters
    ----------
    variables: list of NDArray
    gradients: list of NDArray
    grad_reqs: list of string
    """
    if isinstance(grad_reqs, string_types):
        grad_reqs = [_GRAD_REQ_MAP[grad_reqs]]*len(variables)
    else:
        grad_reqs = [_GRAD_REQ_MAP[i] for i in grad_reqs]

    check_call(_LIB.MXAutogradMarkVariables(
        len(variables),
        c_handle_array(variables),
        c_array_buf(mx_uint, array('I', grad_reqs)),
        c_handle_array(gradients)))


def backward(outputs, out_grads=None, retain_graph=False):
    """Compute the gradients of outputs w.r.t variables.

    Parameters
    ----------
    outputs: list of NDArray
    out_grads: list of NDArray or None
    """
    assert isinstance(outputs, (list, tuple)), \
        "outputs must be a list or tuple of NDArrays"

    if out_grads is None:
        check_call(_LIB.MXAutogradBackward(
            len(outputs),
            c_handle_array(outputs),
            ctypes.c_void_p(0),
            ctypes.c_int(retain_graph)))
        return

    ograd_handles = []
    for arr in out_grads:
        if arr is not None:
            ograd_handles.append(arr.handle)
        else:
            ograd_handles.append(NDArrayHandle(0))
    assert len(ograd_handles) == len(outputs), \
        "outputs and out_grads must have the same length"

    check_call(_LIB.MXAutogradBackward(
        len(outputs),
        c_handle_array(outputs),
        c_array(NDArrayHandle, ograd_handles),
        ctypes.c_int(retain_graph)))


def compute_gradient(outputs):
    """Deprecated. Please use backward"""
    backward(outputs)


def grad_and_loss(func, argnum=None):
    """Return function that computes both gradient of arguments and loss value.

    Parameters
    ----------
    func: a python function
        The forward (loss) function.
    argnum: an int or a list of int
        The index of argument to calculate gradient for.

    Returns
    -------
    grad_and_loss_func: a python function
        A function that would compute both the gradient of arguments and loss value.
    """
    @functools.wraps(func)
    def wrapped(*args):
        """Wrapped function."""
        variables = args
        if argnum is not None:
            argnum_ = argnum if isinstance(argnum, list) else [argnum]
            variables = [args[i] for i in argnum_]
        for x in variables:
            assert isinstance(x, NDArray), "type of autograd input should NDArray."
        grads = [zeros_like(x) for x in variables]
        mark_variables(variables, grads)
        with train_section():
            outputs = func(*args)
        compute_gradient([outputs] if isinstance(outputs, NDArray) else outputs)
        return grads, outputs
    return wrapped

def grad(func, argnum=None):
    """Return function that computes gradient of arguments.

    Parameters
    ----------
    func: a python function
        The forward (loss) function.
    argnum: an int or a list of int
        The index of argument to calculate gradient for.

    Returns
    -------
    grad_func: a python function
        A function that would compute the gradient of arguments.

    Examples
    --------
    >>> # autograd supports dynamic graph which is changed
    >>> # every instance
    >>> def func(x):
    >>>     r = random.randint(0, 1)
    >>>     if r % 2:
    >>>         return x**2
    >>>     else:
    >>>         return x/3
    >>> # use `grad(func)` to get the gradient function
    >>> for x in range(10):
    >>>     grad_func = grad(func)
    >>>     inputs = nd.array([[1, 2, 3], [4, 5, 6]])
    >>>     grad_vals = grad_func(inputs)
    """
    grad_with_loss_func = grad_and_loss(func, argnum)
    @functools.wraps(grad_with_loss_func)
    def wrapped(*args):
        return grad_with_loss_func(*args)[0]
    return wrapped
