# coding: utf-8
"""Autograd for NDArray."""
from __future__ import absolute_import
from __future__ import division

import ctypes
import functools
from .base import _LIB, check_call
from .base import mx_uint, NDArrayHandle, c_array
from .ndarray import NDArray
from .ndarray import array as ndarray

def set_recording(recording):
    check_call(_LIB.MXAutogradSetRecording(
        ctypes.c_int(recording)))

def set_mark_for_record(arrays, mark):
    handles = []
    for arr in arrays:
        handles.append(arr.handle)
    check_call(_LIB.MXAutogradSetMarkForRecord(
        len(handles),
        c_array(NDArrayHandle, handles),
        ctypes.c_int(mark)))

def compute_gradient(inputs, outputs):
    input_handles = []
    for arr in inputs:
        input_handles.append(arr.handle)
    output_handles = []
    for arr in outputs:
        output_handles.append(arr.handle)

    num_grad = mx_uint()
    grad_handles = ctypes.POINTER(NDArrayHandle)()
    check_call(_LIB.MXAutogradComputeGradient(
        len(input_handles),
        c_array(NDArrayHandle, input_handles),
        len(output_handles),
        c_array(NDArrayHandle, output_handles),
        ctypes.byref(num_grad),
        ctypes.byref(grad_handles)))
    return [NDArray(NDArrayHandle(grad_handles[i])) for i in range(num_grad.value)]

def grad_and_loss(func, argnum=0):
    @functools.wraps(func)
    def wrapped(*args):
        inputs  = [a if isinstance(a, NDArray) else ndarray(a) for a in args]
        argnums = [argnum] if isinstance(argnum, int) else argnum
        marked_arrays = [inputs[i] for i in argnums]
        set_recording(True)
        set_mark_for_record(marked_arrays, True)
        outputs = func(*inputs)
        set_recording(False)
        grad_vals = compute_gradient(
            inputs, [outputs])
        set_mark_for_record(marked_arrays, False)
        if len(grad_vals) == 1:
            grad_vals = grad_vals[0]
        return grad_vals, outputs
    return wrapped

def grad(func, argnum=0):
    grad_with_loss_func = grad_and_loss(func, argnum)
    @functools.wraps(grad_with_loss_func)
    def wrapped(*args):
        return grad_with_loss_func(*args)[0]
    return wrapped
