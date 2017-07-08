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
    for gradient computation. Operators will also run with ctx.is_train=True. For example,
    Dropout will drop inputs randomly when is_train=True while simply passing through
    if is_train=False.

    Parameters
    ----------
    is_recording: bool

    Returns
    -------
    previous state before this set.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXAutogradSetIsTraining(
        ctypes.c_int(is_recording), ctypes.byref(prev)))
    return bool(prev.value)


class TrainingStateScope(object):
    """Scope for managing training state.

    Example::
        with TrainingStateScope(True):
            y = model(x)
            backward([y])
    """
    def __init__(self, enter_state):
        self._enter_state = enter_state
        self._prev = None

    def __enter__(self):
        self._prev = set_recording(self._enter_state)

    def __exit__(self, ptype, value, trace):
        if self._prev != self._enter_state:
            set_recording(self._prev)


def record():
    """Returns a training scope context to be used in 'with' statement
    and captures training code.

    Example::
        with autograd.record():
            y = model(x)
            backward([y])
        metric.update(...)
        optim.step(...)
    """
    return TrainingStateScope(True)


def pause():
    """Returns a testing scope context to be used in 'with' statement
    and captures testing code.

    Example::
        with autograd.record():
            y = model(x)
            backward([y])
            with autograd.pause():
                # testing, IO, gradient updates...
    """
    return TrainingStateScope(False)


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


def backward(heads, head_grads=None, retain_graph=False):
    """Compute the gradients of heads w.r.t previously marked variables.

    Parameters
    ----------
    heads: NDArray or list of NDArray
        Output NDArray(s)
    head_grads: NDArray or list of NDArray or None
        Gradients with respect to heads.
    """
    if isinstance(heads, NDArray):
        assert head_grads is None or isinstance(head_grads, NDArray)
        heads = [heads]
        head_grads = [head_grads] if head_grads is not None else None

    output_handles = []
    for arr in heads:
        output_handles.append(arr.handle)

    if head_grads is None:
        check_call(_LIB.MXAutogradBackward(
            len(output_handles),
            c_array(NDArrayHandle, output_handles),
            ctypes.c_void_p(0),
            ctypes.c_int(retain_graph)))
        return

    ograd_handles = []
    for arr in head_grads:
        if arr is not None:
            ograd_handles.append(arr.handle)
        else:
            ograd_handles.append(NDArrayHandle(0))
    assert len(ograd_handles) == len(output_handles), \
        "heads and head_grads must have the same length"

    check_call(_LIB.MXAutogradBackward(
        len(output_handles),
        c_array(NDArrayHandle, output_handles),
        c_array(NDArrayHandle, ograd_handles),
        ctypes.c_int(retain_graph)))
