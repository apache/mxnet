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

import ctypes
from .base import _LIB, check_call, string_types
from .base import mx_uint, NDArrayHandle, c_array
from .ndarray import NDArray
from .symbol import _GRAD_REQ_MAP


def set_recording(is_recording): #pylint: disable=redefined-outer-name
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

def set_training(train_mode): #pylint: disable=redefined-outer-name
    """Set status to training/predicting. This affects ctx.is_train in operator
    running context. For example, Dropout will drop inputs randomly when
    train_mode=True while simply passing through if train_mode=False.

    Parameters
    ----------
    train_mode: bool

    Returns
    -------
    previous state before this set.
    """
    prev = ctypes.c_int()
    check_call(_LIB.MXAutogradSetIsTraining(
        ctypes.c_int(train_mode), ctypes.byref(prev)))
    return bool(prev.value)

def is_recording():
    """Get status on recording/not recording.

    Returns
    -------
    Current state of recording.
    """
    curr = ctypes.c_bool()
    check_call(_LIB.MXAutogradIsRecording(ctypes.byref(curr)))
    return curr.value

def is_training():
    """Get status on training/predicting.

    Returns
    -------
    Current state of training/predicting.
    """
    curr = ctypes.c_bool()
    check_call(_LIB.MXAutogradIsTraining(ctypes.byref(curr)))
    return curr.value


class _RecordingStateScope(object):
    """Scope for managing training state.

    Example::

        with _RecordingStateScope(True, True):
            y = model(x)
            backward([y])

    """
    def __init__(self, is_record, train_mode): #pylint: disable=redefined-outer-name
        self._enter_is_record = is_record
        self._enter_train_mode = train_mode
        self._prev_is_record = None
        self._prev_train_mode = None

    def __enter__(self):
        if self._enter_is_record is not None:
            self._prev_is_record = set_recording(self._enter_is_record)
        if self._enter_train_mode is not None:
            self._prev_train_mode = set_training(self._enter_train_mode)

    def __exit__(self, ptype, value, trace):
        if self._enter_is_record is not None and self._prev_is_record != self._enter_is_record:
            set_recording(self._prev_is_record)
        if self._enter_train_mode is not None and self._prev_train_mode != self._enter_train_mode:
            set_training(self._prev_train_mode)


def record(train_mode=True): #pylint: disable=redefined-outer-name
    """Returns an autograd recording scope context to be used in 'with' statement
    and captures code that needs gradients to be calculated.

    .. note:: When forwarding with train_mode=False, the corresponding backward
              should also use train_mode=False, otherwise gradient is undefined.

    Example::

        with autograd.record():
            y = model(x)
            backward([y])
        metric.update(...)
        optim.step(...)

    Parameters
    ----------
    train_mode: bool, default True
        Whether the forward pass is in training or predicting mode. This controls the behavior
        of some layers such as Dropout, BatchNorm.
    """
    return _RecordingStateScope(True, train_mode)


def pause(train_mode=False): #pylint: disable=redefined-outer-name
    """Returns a scope context to be used in 'with' statement for codes that do not need
    gradients to be calculated.

    Example::

        with autograd.record():
            y = model(x)
            backward([y])
            with autograd.pause():
                # testing, IO, gradient updates...

    Parameters
    ----------
    train_mode: bool, default False
        Whether to do forward for training or predicting.
    """
    return _RecordingStateScope(False, train_mode)


def train_mode():
    """Returns a scope context to be used in 'with' statement
    in which forward pass behavior is set to training mode,
    without changing the recording states.

    Example::

        y = model(x)
        with autograd.train_mode():
            y = dropout(y)

    """
    return _RecordingStateScope(None, True)


def predict_mode():
    """Returns a scope context to be used in 'with' statement
    in which forward pass behavior is set to inference mode,
    without changing the recording states.

    Example::

        with autograd.record():
            y = model(x)
            with autograd.predict_mode():
                y = sampling(y)
            backward([y])
    """
    return _RecordingStateScope(None, False)


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


def backward(heads, head_grads=None, retain_graph=False, train_mode=True): #pylint: disable=redefined-outer-name
    """Compute the gradients of heads w.r.t previously marked variables.

    Parameters
    ----------
    heads: NDArray or list of NDArray
        Output NDArray(s)
    head_grads: NDArray or list of NDArray or None
        Gradients with respect to heads.
    train_mode: bool, optional
        Whether to do backward for training or predicting.
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
            ctypes.c_int(train_mode)))
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
        ctypes.c_int(train_mode)))
