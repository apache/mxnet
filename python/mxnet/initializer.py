import numpy as np
from .ndarray import NDArray
from . import random

def get_fan(shape):
    fan_in = shape[1]
    fan_out = shape[0]
    return fan_in, fan_out


def uniform(arr, scale=0.07):
    if isinstance(arr, NDArray):
        arr[:] = random.uniform(-scale, scale, arr.shape)
    else:
        raise TypeError("Input array must be NDArray")

def normal(arr, sigma=0.07):
    if isinstance(arr, NDArray):
        arr[:] = random.normal(0, sigma, arr.shape)
    else:
        raise TypeError("Input array must be NDArray")

def xavier(arr):
    if isinstance(arr, NDArray):
        fan_in, fan_out = get_fan(arr.shape)
        s = np.sqrt(6. / (fan_in + fan_out))
        uniform(arr, s)
    else:
        raise TypeError("Input array must be NDArray")
