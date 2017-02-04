# pylint: disable=fixme, invalid-name, unused-argument, too-many-arguments, no-name-in-module
"""Common Optimization algorithms with regularizations."""
import math
import pickle
from .ndarray import NDArray, zeros, clip, sqrt
from .ndarray import sgd_update, sgd_mom_update, adam_update, rmsprop_update
from .random import normal


class Optimizer(object):
    """Base class of all optimizers."""
    opt_registry = {}

    @staticmethod
    def register(klass):
        """Register optimizers to the optimizer factory"""
        assert(isinstance(klass, type))
        name = klass.__name__.lower()
        if name in Optimizer.opt_registry:
            print('WARNING: New optimizer %s.%s is overriding '
                  'existing optimizer %s.%s' % (
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
            Rescaling factor on gradient. Normally should be 1/batch_size.

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

    def __init__(self, rescale_grad=1., param_idx2name=None, wd=0.,
                 clip_gradient=None, learning_rate=0.01,
                 lr_scheduler=None, sym=None, begin_num_update=0):
        self.rescale_grad = rescale_grad
        self.lr = learning_rate
        self.lr_scheduler = lr_scheduler
        if lr_scheduler is not None:
            self.lr_scheduler.base_lr = learning_rate

        self.wd = wd
        self.lr_mult = {}
        self.wd_mult = {}
        self.begin_num_update = begin_num_update
        self.num_update = begin_num_update
        self._index_update_count = {}
        self.clip_gradient = clip_gradient

        if param_idx2name is None:
            param_idx2name = {}
        assert isinstance(param_idx2name, dict), \
            'param_idx2name should be a dict of param indexes to names.'
        self.idx2name = param_idx2name.copy()
        self.sym = sym

        self.set_lr_mult({})
        self.set_wd_mult({})

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.
        override in implementations."""

    def update(self, index, weight, grad, state):
        """Update the parameters. override in implementations"""

    # pylint: disable=no-self-use
    def set_lr_scale(self, args_lrscale):
        """set lr scale is deprecated. Use set_lr_mult instead."""
        raise DeprecationWarning

    def set_lr_mult(self, args_lr_mult):
        """Set individual learning rate multipler for parameters

        Parameters
        ----------
        args_lr_mult : dict of string/int to float
            set the lr multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        self.lr_mult = {}
        if self.sym is not None:
            attr = self.sym.attr_dict()
            for name in self.sym.list_arguments():
                if name in attr and '__lr_mult__' in attr[name]:
                    self.lr_mult[name] = float(attr[name]['__lr_mult__'])
        self.lr_mult.update(args_lr_mult)

    def set_wd_mult(self, args_wd_mult):
        """Set individual weight decay multipler for parameters.
        By default wd multipler is 0 for all params whose name doesn't
        end with _weight, if param_idx2name is provided.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            set the wd multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        self.wd_mult = {}
        for n in self.idx2name.values():
            if not (n.endswith('_weight') or n.endswith('_gamma')):
                self.wd_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.attr_dict()
            for name in self.sym.list_arguments():
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def _update_count(self, index):
        """
        update num_update

        Parameters:
        index : int
            The index will be updated
        """
        if index not in self._index_update_count:
            self._index_update_count[index] = self.begin_num_update
        self._index_update_count[index] += 1
        self.num_update = max(self._index_update_count[index], self.num_update)

    def _get_lr(self, index):
        """get learning rate for index.

        Parameters
        ----------
        index : int
            The index for weight

        Returns
        -------
        lr : float
            learning rate for this index
        """
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.num_update)
        else:
            lr = self.lr

        if index in self.lr_mult:
            lr *= self.lr_mult[index]
        elif index in self.idx2name:
            lr *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lr

    def _get_wd(self, index):
        """get weight decay for index.
        Returns 0 for non-weights if the name of weights are provided for __init__.

        Parameters
        ----------
        index : int
            The index for weight

        Returns
        -------
        wd : float
            weight decay for this index
        """
        wd = self.wd
        if index in self.wd_mult:
            wd *= self.wd_mult[index]
        elif index in self.idx2name:
            wd *= self.wd_mult.get(self.idx2name[index], 1.0)
        return wd

# convenience wrapper for Optimizer.Register
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
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
    """
    def __init__(self, momentum=0.0, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            self.kwargs['momentum'] = self.momentum
        if self.clip_gradient:
            self.kwargs['clip_gradient'] = self.clip_gradient

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
            return zeros(weight.shape, weight.context, dtype=weight.dtype)

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
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        if state:
            sgd_mom_update(weight, grad, state, out=weight,
                           lr=lr, wd=wd, **self.kwargs)
        else:
            sgd_update(weight, grad, out=weight,
                       lr=lr, wd=wd, **self.kwargs)

@register
class DCASGD(Optimizer):
    """ DCASGD optimizer with momentum and weight regularization.

    implement paper "Asynchronous Stochastic Gradient Descent with
                    Delay Compensation for Distributed Deep Learning"

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    momentum : float, optional
       momentum value

    lamda : float, optional
       scale DC value

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
    """
    def __init__(self, momentum=0.0, lamda=0.04, **kwargs):
        super(DCASGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.weight_previous = {}
        self.lamda = lamda

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
            return zeros(weight.shape, weight.context, dtype=weight.dtype)

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
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state
            mom[:] *= self.momentum
            if self.weight_previous.has_key(index):
                mom[:] += -lr * (grad + wd * weight + self.lamda \
                                    * grad * grad * (weight - self.weight_previous[index]))
                self.weight_previous[index] = weight
            else:
                mom[:] += -lr * (grad + wd * weight)
                self.weight_previous[index] = weight
            weight[:] += mom
        else:
            assert self.momentum == 0.0
            if self.weight_previous.has_key(index):
                weight[:] += -lr * (grad + wd * weight + self.lamda \
                                    * grad * grad * (weight - self.weight_previous[index]))
                self.weight_previous[index] = weight
            else:
                weight[:] += -lr * (grad + wd * weight)
                self.weight_previous[index] = weight

@register
class NAG(SGD):
    """SGD with nesterov
    It is implemented according to
    https://github.com/torch/optim/blob/master/sgd.lua
    """
    def __init__(self, **kwargs):
        super(NAG, self).__init__(**kwargs)

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
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state
            mom[:] *= self.momentum
            grad += wd * weight
            mom[:] += grad
            grad[:] += self.momentum * mom
            weight[:] += -lr * grad
        else:
            assert self.momentum == 0.0
            weight[:] += -lr * (grad + wd * weight)


@register
class SGLD(Optimizer):
    """Stochastic Langevin Dynamics Updater to sample from a distribution.

    Parameters
    ----------
    learning_rate : float, optional
        learning_rate of SGD

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]

    param_idx2name : dict of string/int to float, optional
        special treat weight decay in parameter ends with bias, gamma, and beta
    """
    def __init__(self, **kwargs):
        super(SGLD, self).__init__(**kwargs)

    def create_state(self, index, weight):
        """Create additional optimizer state such as momentum.

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        return None

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
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        weight[:] += - lr/2 * (grad + wd * weight) + normal(0, math.sqrt(lr),
                                                            weight.shape, weight.context)


@register
class ccSGD(SGD):
    """[Deprecated] Same as sgd. Left here for backward compatibility."""
    def __init__(self, *args, **kwargs):
        super(ccSGD, self).__init__(*args, **kwargs)

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
        rescaling factor of gradient. Normally should be 1/batch_size.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 decay_factor=(1 - 1e-8), **kwargs):
        super(Adam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.kwargs = {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon,
                       'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            self.kwargs['clip_gradient'] = self.clip_gradient

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

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
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        t = self._index_update_count[index]
        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1

        mean, var = state
        adam_update(weight, grad, mean, var, out=weight,
                    lr=lr, wd=wd, **self.kwargs)


@register
class AdaGrad(Optimizer):
    """AdaGrad optimizer of Duchi et al., 2011,

    This code follows the version in http://arxiv.org/pdf/1212.5701v1.pdf  Eq(5)
    by Matthew D. Zeiler, 2012. AdaGrad will help the network to converge faster
    in some cases.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.05.

    wd : float, optional
        L2 regularization coefficient add to all the weights

    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.

    eps: float, optional
        A small float number to make the updating processing stable
        Default value is set to 1e-7.

    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """
    def __init__(self, eps=1e-7, **kwargs):
        super(AdaGrad, self).__init__(**kwargs)
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        return zeros(weight.shape, weight.context)  # history

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        history = state
        history[:] += (grad * grad)
        weight[:] += -lr * (grad / sqrt(history + self.float_stable_eps) + wd * weight)


@register
class RMSProp(Optimizer):
    """RMSProp optimizer of Tieleman & Hinton, 2012,

    This code follows the version in  http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45)
    by Alex Graves, 2013.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.002.
    gamma1: float, optional
        decay factor of moving average for gradient, gradient^2.
        Default value is set to 0.95.
    gamma2: float, optional
        "momentum" factor.
        Default value if set to 0.9.
    epsilon : float, optional
        Default value is set to 1e-8.
    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.
    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """
    def __init__(self, learning_rate=0.001, gamma1=0.95, gamma2=0.9,
                 epsilon=1e-8, **kwargs):
        super(RMSProp, self).__init__(learning_rate=learning_rate, **kwargs)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.kwargs = {'gamma1': gamma1, 'gamma2': gamma2, 'epsilon': epsilon,
                       'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            self.kwargs['clip_gradient'] = self.clip_gradient

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        return (zeros(weight.shape, weight.context),  # n
                zeros(weight.shape, weight.context),  # g
                zeros(weight.shape, weight.context))  # delta

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
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        n, g, delta = state
        rmsprop_update(weight, grad, n, g, delta, out=weight,
                       lr=lr, wd=wd, **self.kwargs)

@register
class AdaDelta(Optimizer):
    """
    AdaDelta optimizer as described in
    Zeiler, M. D. (2012).
    *ADADELTA: An adaptive learning rate method.*

    http://arxiv.org/abs/1212.5701

    Parameters
    ----------
    rho: float
        Decay rate for both squared gradients and delta x
    epsilon : float
        The constant as described in the thesis
    wd : float
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient. Normally should be 1/batch_size.
    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """
    def __init__(self, rho=0.90, epsilon=1e-5, **kwargs):
        super(AdaDelta, self).__init__(**kwargs)
        self.rho = rho
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context),  # accumulated g
                zeros(weight.shape, weight.context))  # accumulated delta

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        wd = self._get_wd(index)
        self._update_count(index)

        # preprocess grad
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # accumulated g and delta initlization
        acc_g, acc_delta = state

        # update g, delta
        acc_g[:] = self.rho * acc_g + (1. - self.rho) * grad * grad
        current_delta = sqrt(acc_delta + self.epsilon) / sqrt(acc_g + self.epsilon) * grad
        acc_delta[:] = self.rho * acc_delta + (1. - self.rho) * current_delta * current_delta

        # update weight
        weight[:] -= current_delta + wd * weight


@register
class Test(Optimizer):
    """For test use"""
    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)

    # pylint: disable=no-self-use
    def create_state(self, index, weight):
        """Create a state to duplicate weight"""
        return zeros(weight.shape, weight.context)

    def update(self, index, weight, grad, state):
        """performs w += rescale_grad * grad"""
        weight[:] += grad * self.rescale_grad
        state[:] = weight

# backward compatibility wrapper for Optimizer.CreateOptimizer
create = Optimizer.create_optimizer

class Updater(object):
    """updater for kvstore"""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.states = {}

    def __call__(self, index, grad, weight):
        """Update weight given gradient and index"""
        if index not in self.states:
            self.states[index] = self.optimizer.create_state(index, weight)
        self.optimizer.update(index, weight, grad, self.states[index])

    def set_states(self, states):
        """set updater states"""
        self.states = pickle.loads(states)

    def get_states(self):
        """get updater states"""
        return pickle.dumps(self.states)

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
    return Updater(optimizer)
