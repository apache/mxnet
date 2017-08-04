# coding: utf-8
"""Autograd for NDArray."""
from __future__ import absolute_import
from __future__ import division

import ctypes
from .base import _LIB, check_call, string_types
from .base import mx_uint, NDArrayHandle, c_array
from .ndarray import NDArray
from .symbol import _GRAD_REQ_MAP


def set_recording(is_recording):
    """Set status to recording/not recording. When recording, graph will be constructed
    for gradient computation.

    Parameters
    ----------
    is_recording: bool

    Returns
    -------
    previous state before this set.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXAutogradSetIsRecording(
        ctypes.c_int(is_recording), ctypes.byref(prev)))
    return bool(prev.value)

def set_training(is_train):
    """Set status to training/not training. This affects ctx.is_train in operator
    running context. For example, Dropout will drop inputs randomly when
    is_train=True while simply passing through if is_train=False.

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
    return bool(prev.value)


class RecordingStateScope(object):
    """Scope for managing training state.

    Example::
        with RecordingStateScope(True, True):
            y = model(x)
            backward([y])
    """
    def __init__(self, enter_state, is_train):
        self._enter_state = enter_state
        self._enter_is_train = is_train
        self._prev = None
        self._prev_is_train = None

    def __enter__(self):
        self._prev = set_recording(self._enter_state)
        self._prev_is_train = set_training(self._enter_is_train)

    def __exit__(self, ptype, value, trace):
        if self._prev != self._enter_state:
            set_recording(self._prev)
        if self._prev_is_train != self._enter_is_train:
            set_training(self._prev_is_train)


def record(is_train=True):
    """Returns a training scope context to be used in 'with' statement
    and captures training code.

    .. note:: When forwarding with is_train=False, the corresponding backward
              should also use is_train=False, otherwise gradient is undefined.

    Example::
        with autograd.record():
            y = model(x)
            backward([y])
        metric.update(...)
        optim.step(...)

    Parameters
    ----------
    is_train: bool, default True
        Whether to do forward for training or inference.
    """
    return RecordingStateScope(True, is_train)


def pause(is_train=False):
    """Returns a testing scope context to be used in 'with' statement
    and captures testing code.

    Example::
        with autograd.record():
            y = model(x)
            backward([y])
            with autograd.pause():
                # testing, IO, gradient updates...

    Parameters
    ----------
    is_train: bool, default False
        Whether to do forward for training or inference.
    """
    return RecordingStateScope(False, is_train)


def mark_variables(variables, gradients, grad_reqs='write'):
    """Mark NDArrays as variables to compute gradient for autograd.

    Parameters
    ----------
    variables: NDArray or list of NDArray
    gradients: NDArray or list of NDArray
    grad_reqs: str or list of str
    """
    if isinstance(variables, NDArray):
        assert isinstance(gradients, NDArray)
        variables = [variables]
        gradients = [gradients]

    variable_handles = []
    gradient_handles = []
    for var, gradvar in zip(variables, gradients):
        variable_handles.append(var.handle)
        gradient_handles.append(gradvar.handle)
    if isinstance(grad_reqs, string_types):
        grad_reqs = [_GRAD_REQ_MAP[grad_reqs]]*len(variables)
    else:
        grad_reqs = [_GRAD_REQ_MAP[i] for i in grad_reqs]

    check_call(_LIB.MXAutogradMarkVariables(
        len(variable_handles),
        c_array(NDArrayHandle, variable_handles),
        c_array(mx_uint, grad_reqs),
        c_array(NDArrayHandle, gradient_handles)))


def backward(heads, head_grads=None, retain_graph=False, is_train=True):
    """Compute the gradients of heads w.r.t previously marked variables.

    Parameters
    ----------
    heads: NDArray or list of NDArray
        Output NDArray(s)
    head_grads: NDArray or list of NDArray or None
        Gradients with respect to heads.
    is_train: bool, optional
        Whether to do backward for training or inference.
    """
    if isinstance(heads, NDArray):
        assert head_grads is None or isinstance(head_grads, NDArray)
        heads = [heads]
        head_grads = [head_grads] if head_grads is not None else None

    output_handles = []
    for arr in heads:
        output_handles.append(arr.handle)

    if head_grads is None:
        check_call(_LIB.MXAutogradBackwardEx(
            len(output_handles),
            c_array(NDArrayHandle, output_handles),
            ctypes.c_void_p(0),
            ctypes.c_int(retain_graph),
            ctypes.c_int(is_train)))
        return

    ograd_handles = []
    for arr in head_grads:
        if arr is not None:
            ograd_handles.append(arr.handle)
        else:
            ograd_handles.append(NDArrayHandle(0))
    assert len(ograd_handles) == len(output_handles), \
        "heads and head_grads must have the same length"

    check_call(_LIB.MXAutogradBackwardEx(
        len(output_handles),
        c_array(NDArrayHandle, output_handles),
        c_array(NDArrayHandle, ograd_handles),
        ctypes.c_int(retain_graph),
        ctypes.c_int(is_train)))
