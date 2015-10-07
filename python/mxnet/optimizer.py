# pylint: disable=fixme, invalid-name, unused-argument, too-many-arguments, no-name-in-module
"""Common Optimization algorithms with regularizations."""
from .ndarray import NDArray, zeros, clip

class Optimizer(object):
    """Base class of all optimizers."""
    def __init__(self):
        self.iteration = 0

    def begin_round(self, iteration):
        """Function called to notify beginning of iteration.

        Parameters
        ----------
        iteration : int
            The iteration number.
        """
        self.iteration = iteration


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

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """
    def __init__(self, learning_rate=0.01, momentum=0.0,
                 wd=0.0001, rescale_grad=1, clip_gradient=None,
                 lr_scheduler=None):
        super(SGD, self).__init__()
        self.lr = learning_rate
        self.momentum = momentum
        self.wd = wd
        self.rescale_grad = rescale_grad
        self.clip_gradient = clip_gradient
        self.lr_scheduler = lr_scheduler
        if lr_scheduler != None:
            self.lr_scheduler.base_lr = learning_rate
        self.momentums = {}

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        if self.momentum == 0.0:
            return None
        else:
            return zeros(weight.shape, weight.context)

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        # TODO(bing) implement wd_bias, wd_gamma, wd_beta
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))

        if self.lr_scheduler != None:
            lr = self.lr_scheduler(self.iteration)
        else:
            lr = self.lr
        if state:
            mom = state
            mom[:] *= self.momentum
            if self.clip_gradient == None:
                mom[:] += -lr * (grad * self.rescale_grad + self.wd * weight)
            else:
                mom[:] += -lr * (clip(grad * self.rescale_grad, -self.clip_gradient,
                                      self.clip_gradient) +
                                 self.wd * weight)
            weight[:] += mom
        else:
            assert self.momentum == 0.0
            weight[:] += -lr * (grad * self.rescale_grad + self.wd * weight)


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
