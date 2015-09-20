# pylint: skip-file
"""Common Optimization algorithms with regularizations."""
from .ndarray import NDArray, zeros

class Optimizer(object):
    """Base class of all optimizers."""
    def begin_round(self, iteration):
        """Function called to notify beginning of iteration.

        Parameters
        ----------
        iteration : int
            The iteration number.
        """
        pass


class SGD(Optimizer):
    """A very simple SGD optimizer with momentum and weight regularization.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    momentum : float, optional
       momentum value

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient.

    """
    def __init__(self, learning_rate=0.01, momentum=0.0,
                 wd=0.0001, rescale_grad=1):
        self.lr = learning_rate
        self.momentum = momentum
        self.wd = wd
        self.rescale_grad = rescale_grad
        self.momentums = {}

    def update(self, key, weight, grad):
        """Update the parameters.

        Parameters
        ----------
        key : str
           The name of the parameter.
        weight: NDArray
            weight ndarray
        grad: NDArray
            grad ndarray
        """
        # TODO(bing) implement wd_bias, wd_gamma, wd_beta
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))

        if key not in self.momentums:
            self.momentums[key] = zeros(grad.shape, grad.context)
        mom = self.momentums[key]
        mom[:] *= self.momentum
        mom[:] += -self.lr * (grad * self.rescale_grad + self.wd * weight)
        weight[:] += mom


def create(name, rescale_grad=1, **kwargs):
    """Create an optimizer with specified name.

    Parameters
    ----------
    name: str
        Name of required optimizer

    rescale_grad : float
        Rescaling factor on gradient.

    kwargs: dict
        Parameters for optimizer

    Returns
    -------
    opt : optimizer
        The result optimizer.
    """
    if name == 'sgd' or name == 'SGD':
        return SGD(rescale_grad=rescale_grad, **kwargs)
    else:
        raise ValueError('Cannot find optimizer %s' % name)
