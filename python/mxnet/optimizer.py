"""Weight updating functions."""
import math
import pickle
import logging
from .ndarray import NDArray, zeros, clip, sqrt, sign
from .ndarray import sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update
from .random import normal


class Optimizer(object):
    """The base class inherited by all optimizers.

    Parameters
    ----------
    rescale_grad : float, optional
        Multiply the gradient with `rescale_grad` before updating. Often
        choose to be ``1.0/batch_size``.

    param_idx2name : dict from int to string, optional
        A dictionary that maps int index to string name.

    clip_gradient : float, optional
        Clip the gradient by projecting onto the box ``[-clip_gradient, clip_gradient]``.

    learning_rate : float, optional
        The initial learning rate.

    lr_scheduler : LRScheduler, optional
        The learning rate scheduler.

    wd : float, optional
        The weight decay (or L2 regularization) coefficient. Modifies objective
        by adding a penalty for having large weights.

    sym: Symbol, optional
        The Symbol this optimizer is applying to.

    begin_num_update : int, optional
        The initial number of updates
    """
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

    opt_registry = {}

    @staticmethod
    def register(klass):
        """Register a new optimizer.

        Once an optimizer is registered, we can create an instance of this
        optimizer with `create_optimizer` later.

        Examples
        --------

        >>> @mx.optimizer.Optimizer.register
        ... class MyOptimizer(mx.optimizer.Optimizer):
        ...     pass
        >>> optim = mx.optimizer.Optimizer.create_optimizer('MyOptimizer')
        >>> print(type(optim))
        <class '__main__.MyOptimizer'>
        """
        assert(isinstance(klass, type))
        name = klass.__name__.lower()
        if name in Optimizer.opt_registry:
            logging.warning('WARNING: New optimizer %s.%s is overriding '
                            'existing optimizer %s.%s',
                            klass.__module__, klass.__name__,
                            Optimizer.opt_registry[name].__module__,
                            Optimizer.opt_registry[name].__name__)
        Optimizer.opt_registry[name] = klass
        return klass

    @staticmethod
    def create_optimizer(name, **kwargs):
        """Instantiate an optimizer with a given name and kwargs.

        Notes
        -----
        We can use the alias `create` for ``Optimizer.create_optimizer``

        Parameters
        ----------
        name: str
            Name of the optimizer. Should be the name
            of a subclass of Optimizer. Case insensitive.

        kwargs: dict
            Parameters for the optimizer.

        Returns
        -------
        Optimizer
            An instantiated optimizer.

        Examples
        --------
        >>> sgd = mx.optimizer.Optimizer.create_optimizer('sgd')
        >>> type(sgd)
        <class 'mxnet.optimizer.SGD'>
        >>> adam = mx.optimizer.create('adam', learning_rate=.1)
        >>> type(adam)
        <class 'mxnet.optimizer.Adam'>
        """
        if name.lower() in Optimizer.opt_registry:
            return Optimizer.opt_registry[name.lower()](**kwargs)
        else:
            raise ValueError('Cannot find optimizer %s' % name)


    def create_state(self, index, weight):
        """Create auxiliary state for a given weight

        Some optimizers require additional states, e.g. as momentum, in addition
        to gradients in order to update weights. This function creates state
        for a given weight which will be used in `update`. This function is
        called only once for each weight.

        Parameters
        ----------
        index : int
            An unique index to identify the weight.
        weight : NDArray
            The weight.

        Returns
        -------
        state : any obj
            The state associated with the weight.
        """

    def update(self, index, weight, grad, state):
        """Updates the given parameter using the corresponding gradient and state.

        Parameters
        ----------
        index : int
            The unique index of the parameter into the individual learning
            rates and weight decays. Learning rates and weight decay
            may be set via `set_lr_mult()` and `set_wd_mult()`, respectively.
        weight : NDArray
            The parameter to be updated.
        grad : NDArray
            The gradient of the objective with respect to this parameter.
        state : any obj
            The state returned by `create_state()`.
        """
        raise NotImplementedError()

    def set_lr_scale(self, args_lrscale): # pylint: disable=unused-argument
        """[DEPRECATED] set lr scale. Use set_lr_mult instead."""
        raise DeprecationWarning

    def set_lr_mult(self, args_lr_mult):
        """Sets an individual learning rate multiplier for each parameter.

        If you specify a learning rate multiplier for a parameter, then
        the learning rate for the parameter will be set as the product of
        the global learning rate `self.lr` and its multiplier.

        .. note:: The default learning rate multiplier of a `Variable`
            can be set with `lr_mult` argument in the constructor.

        Parameters
        ----------
        args_lr_mult : dict of str/int to float
            For each of its key-value entries, the learning rate multipler for the
            parameter specified in the key will be set as the given value.

            You can specify the parameter with either its name or its index.
            If you use the name, you should pass `sym` in the constructor,
            and the name you specified in the key of `args_lr_mult` should match
            the name of the parameter in `sym`. If you use the index, it should
            correspond to the index of the parameter used in the `update` method.

            Specifying a parameter by its index is only supported for backward
            compatibility, and we recommend to use the name instead.
        """
        self.lr_mult = {}
        if self.sym is not None:
            attr = self.sym.attr_dict()
            for name in self.sym.list_arguments():
                if name in attr and '__lr_mult__' in attr[name]:
                    self.lr_mult[name] = float(attr[name]['__lr_mult__'])
        self.lr_mult.update(args_lr_mult)

    def set_wd_mult(self, args_wd_mult):
        """Sets an individual weight decay multiplier for each parameter.

        By default, if `param_idx2name` was provided in the
        constructor, the weight decay multipler is set as 0 for all
        parameters whose name don't end with ``_weight`` or
        ``_gamma``.

        .. note:: The default weight decay multiplier for a `Variable`
            can be set with its `wd_mult` argument in the constructor.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            For each of its key-value entries, the weight decay multipler for the
            parameter specified in the key will be set as the given value.

            You can specify the parameter with either its name or its index.
            If you use the name, you should pass `sym` in the constructor,
            and the name you specified in the key of `args_lr_mult` should match
            the name of the parameter in `sym`. If you use the index, it should
            correspond to the index of the parameter used in the `update` method.

            Specifying a parameter by its index is only supported for backward
            compatibility, and we recommend to use the name instead.
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
        """Update num_update

        Parameters:
        index : int
            The index to be updated.
        """
        if index not in self._index_update_count:
            self._index_update_count[index] = self.begin_num_update
        self._index_update_count[index] += 1
        self.num_update = max(self._index_update_count[index], self.num_update)

    def _get_lr(self, index):
        """Get the learning rate given the index of the weight.

        Parameters
        ----------
        index : int
            The index corresponding to the weight.

        Returns
        -------
        lr : float
            Learning rate for this index.
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
        Returns 0 for non-weights if the name of weights are provided for `__init__`.

        Parameters
        ----------
        index : int
            The index for weight.

        Returns
        -------
        wd : float
            Weight decay for this index.
        """
        wd = self.wd
        if index in self.wd_mult:
            wd *= self.wd_mult[index]
        elif index in self.idx2name:
            wd *= self.wd_mult.get(self.idx2name[index], 1.0)
        return wd

# convenience wrapper for Optimizer.Register
register = Optimizer.register   # pylint: disable=invalid-name

@register
class SGD(Optimizer):
    """The SGD optimizer with momentum and weight decay.

    The optimizer updates the weight by::

        state = momentum * state + lr * rescale_grad * clip(grad, clip_gradient) + wd * weight
        weight = weight - state

    For details of the update algorithm see :class:`~mxnet.ndarray.sgd_update` and
    :class:`~mxnet.ndarray.sgd_mom_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`:

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
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
        if self.momentum == 0.0:
            return None
        else:
            return zeros(weight.shape, weight.context, dtype=weight.dtype)

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        if state is not None:
            sgd_mom_update(weight, grad, state, out=weight,
                           lr=lr, wd=wd, **self.kwargs)
        else:
            sgd_update(weight, grad, out=weight,
                       lr=lr, wd=wd, **self.kwargs)

@register
class DCASGD(Optimizer):
    """The DCASGD optimizer

    This class implements the optimizer described in *Asynchronous Stochastic Gradient Descent with
    Delay Compensation for Distributed Deep Learning*, available at https://arxiv.org/abs/1609.08326

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`:

    Parameters
    ----------
    momentum : float, optional
       The momentum value.

    lamda : float, optional
       Scale DC value.
    """
    def __init__(self, momentum=0.0, lamda=0.04, **kwargs):
        super(DCASGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.weight_previous = {}
        self.lamda = lamda

    def create_state(self, index, weight):
        if self.momentum == 0.0:
            return (None,
                    weight.copy())  # previous weight
        else:
            return (zeros(weight.shape, weight.context, dtype=weight.dtype), # momentum
                    weight.copy())  # previous weight

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        mom, previous_weight = state
        if mom:
            mom[:] *= self.momentum
            mom[:] += -lr * (grad + wd * weight + self.lamda \
                      * grad * grad * (weight - previous_weight))
        else:
            assert(self.momentum == 0.0)
            mom = -lr * (grad + wd * weight + self.lamda \
                      * grad * grad * (weight - previous_weight))
        previous_weight[:] = weight
        weight[:] += mom

@register
class NAG(SGD):
    """Nesterov accelerated SGD.

    This optimizer updates each weight by:

        state = momentum * state + grad + wd * weight
        weight = weight - (lr * (grad + momentum * state))

    This optimizer accepts the same arguments as :class:`.SGD`.
    """
    def __init__(self, **kwargs):
        super(NAG, self).__init__(**kwargs)

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        if state is not None:
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
    """Stochastic Gradient Riemannian Langevin Dynamics.

    This class implements the optimizer described in the paper *Stochastic Gradient
    Riemannian Langevin Dynamics on the Probability Simplex*, available at
    https://papers.nips.cc/paper/4883-stochastic-gradient-riemannian-langevin-dynamics-on-the-probability-simplex.pdf

    """
    def __init__(self, **kwargs):
        super(SGLD, self).__init__(**kwargs)

    def create_state(self, index, weight):
        return None

    def update(self, index, weight, grad, state):
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


@register  # pylint: disable=invalid-name
class ccSGD(SGD):
    """[Deprecated] Same as sgd. Left here for backward compatibility."""
    def __init__(self, *args, **kwargs):
        super(ccSGD, self).__init__(*args, **kwargs)

@register
class Adam(Optimizer):
    """The Adam optimizer.

    This class implements the optimizer described in *Adam: A Method for
    Stochastic Optimization*, available at http://arxiv.org/abs/1412.6980

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`:

    For details of the update algorithm, see :class:`ndarray.adam_update`.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 **kwargs):
        super(Adam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.kwargs = {'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon,
                       'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            self.kwargs['clip_gradient'] = self.clip_gradient

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
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
    """AdaGrad optimizer

    This calss implements the AdaGrad optiizer described in *Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization*, and available at
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`:

    Parameters
    ----------
    eps: float, optional
        Small value to avoid division by 0.
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
    """The RMSProp optimizer.

    Two versions of RMSProp are implemented:

    If ``centered=False``, we follow
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf by
    Tieleman & Hinton, 2012.
    For details of the update algorithm see :class:`~mxnet.ndarray.rmsprop_update`

    If ``centered=True``, we follow http://arxiv.org/pdf/1308.0850v5.pdf (38)-(45)
    by Alex Graves, 2013.
    For details of the update algorithm see :class:`~mxnet.ndarray.rmspropalex_update`

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`:

    Parameters
    ----------
    gamma1: float, optional
        A decay factor of moving average over past squared gradient.
    gamma2: float, optional
        A "momentum" factor. Only used if `centered`=``True``.
    epsilon : float, optional
        Small value to avoid division by 0.
    centered : bool, optional
        Flag to control which version of RMSProp to use.
        ``True`` will use Graves's version of `RMSProp`,
        ``False`` will use Tieleman & Hinton's version of `RMSProp`.
    clip_weights : float, optional
        Clips weights into range ``[-clip_weights, clip_weights]``
    """
    def __init__(self, learning_rate=0.001, gamma1=0.9, gamma2=0.9,
                 epsilon=1e-8, centered=False, clip_weights=None, **kwargs):
        super(RMSProp, self).__init__(learning_rate=learning_rate, **kwargs)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.centered = centered
        self.clip_weights = clip_weights
        self.kwargs = {'gamma1': gamma1, 'epsilon': epsilon,
                       'rescale_grad': self.rescale_grad}
        if self.centered:
            self.kwargs['gamma2'] = gamma2
        if self.clip_gradient:
            self.kwargs['clip_gradient'] = self.clip_gradient
        if self.clip_weights:
            self.kwargs['clip_weights'] = self.clip_weights

    def create_state(self, index, weight):
        if self.centered:
            return (
                zeros(weight.shape, weight.context),  # n
                zeros(weight.shape, weight.context),  # g
                zeros(weight.shape, weight.context))  # delta
        else:
            return (zeros(weight.shape, weight.context), )  # n

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        if not self.centered:
            (n, ) = state
            rmsprop_update(
                weight, grad, n, out=weight, lr=lr, wd=wd, **self.kwargs)
        else:
            n, g, delta = state
            rmspropalex_update(weight, grad, n, g, delta, out=weight,
                               lr=lr, wd=wd, **self.kwargs)

@register
class AdaDelta(Optimizer):
    """The AdaDelta optimizer.

    This class implements AdaDelta, an optimizer described in  *ADADELTA: An adaptive
    learning rate method*, available at https://arxiv.org/abs/1212.5701

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`:

    Parameters
    ----------
    rho: float
        Decay rate for both squared gradients and delta.
    epsilon : float
        Small value to avoid division by 0.
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

#pylint: disable=invalid-name
@register
class Ftrl(Optimizer):
    """
    Reference:Ad Click Prediction: a View from the Trenches

    Parameters
    ----------
    lamda1 : float, optional
        L1 regularization coefficient.

    learning_rate : float, optional
        The initial learning rate.

    beta : float, optional
        Per-coordinate learning rate correlation parameter.
    eta_{t,i}=frac{learning_rate}{beta+sqrt{sum_{s=1^}tg_{s,i}^t}

    """

    def __init__(self, lamda1=0.01, learning_rate=0.1, beta=1, **kwargs):
        super(Ftrl, self).__init__(**kwargs)
        self.lamda1 = lamda1
        self.beta = beta
        self.lr = learning_rate

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context),  # dn
                zeros(weight.shape, weight.context))  # n

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        wd = self._get_wd(index)
        lr = self._get_lr(index)

        # preprocess grad
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # accumulated g and delta initlization
        dn, n = state

        #update dn, n
        dn += grad - (sqrt(n + grad * grad) - sqrt(n)) * weight / lr
        n += grad * grad

        # update weight
        weight[:] = (sign(dn) * self.lamda1 - dn) / \
            ((self.beta + sqrt(n)) / lr + wd) * (NDArray.abs(dn) > self.lamda1)

@register
class Test(Optimizer):
    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)

    def create_state(self, index, weight):
        """Create a state to duplicate weight"""
        return zeros(weight.shape, weight.context)

    def update(self, index, weight, grad, state):
        """performs w += rescale_grad * grad"""
        weight[:] += grad * self.rescale_grad
        state[:] = weight

# backward compatibility wrapper for Optimizer.CreateOptimizer
create = Optimizer.create_optimizer  # pylint: disable=invalid-name

class Updater(object):
    """Updater for kvstore."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.states = {}

    def __call__(self, index, grad, weight):
        """Update weight given gradient and index."""
        if index not in self.states:
            self.states[index] = self.optimizer.create_state(index, weight)
        self.optimizer.update(index, weight, grad, self.states[index])

    def set_states(self, states):
        """Set updater states."""
        self.states = pickle.loads(states)

    def get_states(self):
        """Get updater states."""
        return pickle.dumps(self.states)

def get_updater(optimizer):
    """Return a clossure of the updater needed for kvstore.

    Parameters
    ----------
    optimizer: Optimizer
         The optimizer.

    Returns
    -------
    updater: function
         The clossure of the updater.
    """
    return Updater(optimizer)
