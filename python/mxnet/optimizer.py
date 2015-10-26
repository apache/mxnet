# pylint: disable=fixme, invalid-name, unused-argument, too-many-arguments, no-name-in-module
"""Common Optimization algorithms with regularizations."""
from .ndarray import NDArray, zeros, clip

class Optimizer(object):
    """Base class of all optimizers."""
    opt_registry = {}

    @staticmethod
    def register(klass):
        """Register optimizers to the optimizer factory"""
        assert(isinstance(klass, type))
        name = klass.__name__.lower()
        if name in Optimizer.opt_registry:
            print('WARNING: New optimizer %s.%s is overriding ' \
                  'existing optimizer %s.%s'%(
                      klass.__module__, klass.__name__,
                      Optimizer.opt_registry[name].__module__,
                      Optimizer.opt_registry[name].__name__))
        Optimizer.opt_registry[name] = klass
        return klass

    @staticmethod
    def create_optimizer(name, rescale_grad=1, **kwargs):
        """Create an optimizer with specified name.

        Parameters
        ----------
        name: str
            Name of required optimizer. Should be the name
            of a subclass of Optimizer. Case insensitive.

        rescale_grad : float
            Rescaling factor on gradient.

        kwargs: dict
            Parameters for optimizer

        Returns
        -------
        opt : Optimizer
            The result optimizer.
        """
        if name.lower() in Optimizer.opt_registry:
            return Optimizer.opt_registry[name.lower()](
                rescale_grad=rescale_grad,
                **kwargs)
        else:
            raise ValueError('Cannot find optimizer %s' % name)

    def __init__(self, rescale_grad=1):
        self.iteration = 0
        self.rescale_grad = rescale_grad

    def begin_round(self, iteration):
        """Function called to notify beginning of iteration.

        Parameters
        ----------
        iteration : int
            The iteration number.
        """
        self.iteration = iteration

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.
        override in implementations."""

    def update(self, index, weight, grad, state):
        """Update the parameters. override in implementations"""

#convenience wrapper for Optimizer.Register
register = Optimizer.register

@register
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
        super(SGD, self).__init__(rescale_grad)
        self.lr = learning_rate
        self.momentum = momentum
        self.wd = wd
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

        grad = grad * self.rescale_grad
        if self.clip_gradient != None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state
            mom[:] *= self.momentum
            mom[:] += -lr * (grad + self.wd * weight)
            weight[:] += mom
        else:
            assert self.momentum == 0.0
            weight[:] += -lr * (grad + self.wd * weight)

@register
class Test(Optimizer):
    """For test use"""
    def __init__(self, rescale_grad=1):
        super(Test, self).__init__(rescale_grad)

    # pylint: disable=no-self-use
    def create_state(self, index, weight):
        """Create a state to duplicate weight"""
        return zeros(weight.shape, weight.context)

    def update(self, index, weight, grad, state):
        """performs w += rescale_grad * grad"""
        weight[:] += grad * self.rescale_grad
        state[:] = weight

#backward compatibility wrapper for Optimizer.CreateOptimizer
create = Optimizer.create_optimizer

def get_updater(optimizer):
    """Return a clossure of the updater needed for kvstore

    Parameters
    ----------
    optimizer: Optimizer
         The optimizer

    Returns
    -------
    updater: function
         The clossure of the updater
    """
    states = dict()
    def updater(index, grad, weight):
        """updater for kvstore"""
        if index not in states:
            states[index] = optimizer.create_state(index, weight)
        optimizer.update(index, weight, grad, states[index])
    return updater
