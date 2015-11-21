# pylint: disable=fixme, invalid-name, unused-argument, too-many-arguments, no-name-in-module
"""Common Optimization algorithms with regularizations."""
from .ndarray import NDArray, zeros, clip, sqrt
import math

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
        self.rescale_grad = rescale_grad
        self.lr_scale = {}
        self.num_update = 0
        self._index_update_count = {}

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.
        override in implementations."""

    def update(self, index, weight, grad, state):
        """Update the parameters. override in implementations"""

    def set_lr_scale(self, args_lrscale):
        """Set individual learning rate scale for parameters

        Parameters
        ----------
        args_lrscale : dict of index to float
            set the lr multipler for index to float
        """
        self.lr_scale = args_lrscale.copy()

    def _update_count(self, index):
        """
        update num_update

        Parameters:
        index : int
            The index will be updated
        """
        if index not in self._index_update_count:
            self._index_update_count[index] = 0
        self._index_update_count[index] += 1
        self.num_update = max(self._index_update_count[index], self.num_update)


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
        if lr_scheduler is not None:
            self.lr_scheduler.base_lr = learning_rate

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
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.num_update)
            self._update_count(index)
        else:
            lr = self.lr
        lr *= self.lr_scale.get(index, 1.0)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
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
class Adam(Optimizer):
    """Adam optimizer as described in [King2014]_.

    .. [King2014] Diederik Kingma, Jimmy Ba,
       *Adam: A Method for Stochastic Optimization*,
       http://arxiv.org/abs/1412.6980

    the code in this class was adapted from
    https://github.com/mila-udem/blocks/blob/master/blocks/algorithms/__init__.py#L765

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.002.
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
        Default value is set to 0.9.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
        Default value is set to 0.999.
    epsilon : float, optional
        Default value is set to 1e-8.
    decay_factor : float, optional
        Default value is set to 1 - 1e-8.

    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """
    def __init__(self, learning_rate=0.002,
                 beta1=0.9, beta2=0.999, epsilon=1e-8,
                 decay_factor=(1 - 1e-8),
                 wd=0.,
                 rescale_grad=1, clip_gradient=None,
                 lr_scheduler=None):
        super(Adam, self).__init__(rescale_grad)
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.wd = wd
        self.clip_gradient = clip_gradient
        self.lr_scheduler = lr_scheduler
        if lr_scheduler is not None:
            self.lr_scheduler.base_lr = learning_rate
        self.time = 0
        self.time_first_index = None

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        self.time_first_index = None  # time is incremented only on the first index
        return (zeros(weight.shape, weight.context),  # mean
                zeros(weight.shape, weight.context))  # variance

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
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.num_update)
            self._update_count(index)
        else:
            lr = self.lr
        lr *= self.lr_scale.get(index, 1.0)

        mean, variance = state

        # increment time only when the first parameters is called
        if self.time_first_index is None:
            self.time_first_index = index
            self.time = 0  # all parameters share the same time
        elif self.time_first_index == index:
            self.time += 1

        t1 = self.time + 1
        learning_rate = (lr *
                         math.sqrt(1. - self.beta2**t1) /
                         (1. - self.beta1**t1))
        beta_1t = self.beta1 * self.decay_factor ** (t1 - 1)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        mean_t = beta_1t * mean + (1. - beta_1t) * grad
        variance_t = (self.beta2 * variance +
                      (1. - self.beta2) * grad * grad)
        step = (learning_rate * mean_t /
                (sqrt(variance_t) + self.epsilon))
        if self.wd > 0.:
            step += lr * self.wd * weight

        weight[:] += -step
        mean[:] = mean_t
        variance[:] = variance_t

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
