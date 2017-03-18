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
    """Turn on or turn of operator recording.

    Parameters
    ----------
    recording: bool
    """
    check_call(_LIB.MXAutogradSetRecording(
        ctypes.c_int(recording)))

def compute_gradient(inputs, outputs):
    """Compute the gradients of outputs w.r.t inputs.

    Parameters
    ----------
    inputs: list of NDArray

    outputs: list of NDArray

    Returns
    -------
    gradients: list of NDArray
        A
    """
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

def grad_and_loss(func):
    """Return function that computes both gradient of arguments and loss value.

    Parameters
    ----------
    func: a python function
        The forward (loss) function.

    Returns
    -------
    grad_and_loss_func: a python function
        A function that would compute both the gradient of arguments and loss value.
    """
    @functools.wraps(func)
    def wrapped(*args):
        """Wrapped function."""
        inputs  = [a if isinstance(a, NDArray) else ndarray(a) for a in args]
        set_recording(True)
        outputs = func(*inputs)
        set_recording(False)
        grad_vals = compute_gradient(
            inputs, outputs if isinstance(outputs, list) else [outputs])
        return grad_vals, outputs
    return wrapped

def grad(func):
    """Return function that computes gradient of arguments.

    Parameters
    ----------
    func: a python function
        The forward (loss) function.

    Returns
    -------
    grad_func: a python function
        A function that would compute the gradient of arguments.
    """
    grad_with_loss_func = grad_and_loss(func)
    @functools.wraps(grad_with_loss_func)
    def wrapped(*args):
        return grad_with_loss_func(*args)[0]
    return wrapped
