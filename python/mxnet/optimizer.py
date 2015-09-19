# pylint: skip-file
from .ndarray import NDArray, zeros

def get_optimizer(name, batch_size=1, **kwargs):
    """Optimizer factory

    Parameters
    ----------
    name: str
        Name of required optimizer
    batch_size: int
        batch size, used to normalize gradient
    kwargs: dict
        Parameters for optimizer

    Return
    ----------
    A required optimizer object

    """
    if name == "sgd" or name == "SGD":
        return SGD(batch_size=batch_size, **kwargs)
    else:
        raise Exception("Not implemented")

class SGD(object):
    """A very simple SGD optimizer with Nesterov method"""
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0001, batch_size=1, **kwargs):
        """
        Parameter
        ----------
        learning_rate: float
            learning_rate value
        momentum: float
            momentum value
        weight_decay: float
            L2 regularization coefficient
        batch_size: int
            batch size, used to norm gradient
        """

        self.lr = learning_rate
        self.momentum = momentum
        self.wd = weight_decay
        self.batch_size = batch_size
        self.momentums = {}

    def __call__(self, weight, grad, states):
        """
        Parameter
        ---------
        weight: NDArray
            weight ndarray
        grad: NDArray
            grad ndarray
        states: str
            name of weight
        """
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        if states not in self.momentums:
            self.momentums[states] = zeros(grad.shape, grad.context)
        mom = self.momentums[states]
        mom[:] *= self.momentum
        mom[:] += -self.lr * (grad / self.batch_size + self.wd * weight)
        weight[:] += mom

