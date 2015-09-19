import numpy as np
from .ndarray import NDArray
from . import random

class Initializer(object):
    """Base class for Initializer"""
    def __init__(self, **kwargs):
        """Constructor

        Parameters
        ----------
        kwargs: dict
            potential parameters for Initializer implmentation
        """
        self.args = kwargs

    def init_weight(self):
        """Abstruct method to Initialize weight"""
        raise NotImplementedError("Must override it")

    def __call__(self, state, arr):
        """Override () function to do Initialization

        Parameters:
        ----------
        state: str
            name of corrosponding ndarray
        arr: NDArray
            ndarray to be Initialized
        """
        assert(isinstance(state, str))
        assert(isinstance(arr, NDArray))
        if "weight" in state:
            self.init_weight(arr)
        if "bias" in state:
            arr[:] = 0.0
        if "gamma" in state:
            arr[:] = 1.0
        if "beta" in state:
            arr[:] = 0.0

    def get_fan(self, shape):
        """Get input/output from shape

        Parameter
        ---------
        shape: tuple
            shape of NDArray

        Returns
        -------
        fan_in: int
            input dim
        fan_out: int
            output dim
        """
        fan_in = shape[1]
        fan_out = shape[0]
        return fan_in, fan_out

class Uniform(Initializer):
    """Uniform Initializer"""
    def __init__(self, scale=0.07):
        """Constructor

        Parameter
        ---------
        scale: float (default=0.07)
            unifrom range [-scale, scale]
        """
        super().__init__(scale = scale)

    def init_weight(self, arr):
        """Implmentation of abs method

        Parameter
        --------
        arr: NDArray
            NDArray to be Initialized
        """
        if isinstance(arr, NDArray):
            arr[:] = random.uniform(-scale, scale, arr.shape)
        else:
            raise TypeError("Input array must be NDArray")

class Normal(Initializer):
    """Gaussian Initializer"""
    def __init__(self, sigma=0.01):
        """Constuctor of Normal Initializer
        Parameter
        --------
        sigma: float (default=0.01)
            sigma for gaussian distribution
        """
        super().__init__(sigma = sigma)
    def init_weight(self, arr):
        """Implmentation of abs method

        Parameter
        --------
        arr: NDArray
            NDArray to be Initialized
        """
        if isinstance(arr, NDArray):
            arr[:] = random.normal(0, sigma, arr.shape)
        else:
            raise TypeError("Input array must be NDArray")

class Xavier(Initializer):
    def init_weight(self, arr):
        """Implmentation of abs method

        Parameter
        --------
        arr: NDArray
            NDArray to be Initialized
        """
        if isinstance(arr, NDArray):
            fan_in, fan_out = self.get_fan(arr.shape)
            s = np.sqrt(6. / (fan_in + fan_out))
            arr[:] = random.uniform(-s, s, arr.shape)
        else:
            raise TypeError("Input array must be NDArray")
