import numpy as np
from .ndarray import NDArray
from . import random

class Initializer(object):
    def __init__(self, **kwargs):
        self.args = kwargs

    def __call__(self, state, arr):
        assert(isinstance(state, str))
        assert(isinstance(arr, NDArray))
        if "weight" in state:
            self.init_factory(arr)
        if "bias" in state:
            arr[:] = 0.0
        if "gamma" in state:
            arr[:] = 1.0
        if "beta" in state:
            arr[:] = 0.0

    def init_factory(self, arr):
        # need a lot of check
        if self.args["init_type"] == "xavier":
            self.xavier(arr)
        elif self.args["init_type"] == "uniform":
            scale = float(self.args["scale"])
            self.uniform(arr, scale)
        elif self.args["init_type"] == "gaussian":
            sigma = float(self.args["sigma"])
            self.normal(arr, sigma)
        else:
            raise Exception("unknown")

    def get_fan(self, shape):
        fan_in = shape[1]
        fan_out = shape[0]
        return fan_in, fan_out


    def uniform(self, arr, scale=0.07):
        if isinstance(arr, NDArray):
            arr[:] = random.uniform(-scale, scale, arr.shape)
        else:
            raise TypeError("Input array must be NDArray")

    def normal(self, arr, sigma=0.07):
        if isinstance(arr, NDArray):
            arr[:] = random.normal(0, sigma, arr.shape)
        else:
            raise TypeError("Input array must be NDArray")

    def xavier(self, arr):
        if isinstance(arr, NDArray):
            fan_in, fan_out = self.get_fan(arr.shape)
            s = np.sqrt(6. / (fan_in + fan_out))
            self.uniform(arr, s)
        else:
            raise TypeError("Input array must be NDArray")
