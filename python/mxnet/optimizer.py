# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=too-many-lines
"""Weight updating functions."""
import math
import pickle
import warnings
import numpy
from .base import py_str
from .ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs)
from .ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
                      mp_sgd_update, mp_sgd_mom_update, square, ftrl_update, ftml_update,
                      signsgd_update, signum_update)
from .ndarray import _internal
from .ndarray import op
from .ndarray import sparse
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
        The initial number of updates.

    multi_precision : bool, optional
       Flag to control the internal precision of the optimizer.
       ``False`` results in using the same precision as the weights (default),
       ``True`` makes internal 32-bit copy of the weights and applies gradients
                in 32-bit precision even if actual weights used in the model have lower precision.
                Turning this on can improve convergence and accuracy when training with float16.

    Properties
    ----------
    learning_rate : float
        The current learning rate of the optimizer. Given an Optimizer object
        optimizer, its learning rate can be accessed as optimizer.learning_rate.
    """
    def __init__(self, rescale_grad=1., param_idx2name=None, wd=0.,
                 clip_gradient=None, learning_rate=0.01,
                 lr_scheduler=None, sym=None, begin_num_update=0,
                 multi_precision=False, param_dict=None):
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
        self.multi_precision = multi_precision

        if param_idx2name is None:
            param_idx2name = {}
        assert isinstance(param_idx2name, dict), \
            'param_idx2name should be a dict of param indexes to names.'
        self.idx2name = param_idx2name.copy()
        self.sym_info = (sym.attr_dict(), sym.list_arguments()) if sym is not None else ()
        self.param_dict = param_dict if param_dict else {}

        self.set_lr_mult({})
        self.set_wd_mult({})

    opt_registry = {}

    @staticmethod
    def register(klass):
        """Registers a new optimizer.

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
            warnings.warn('WARNING: New optimizer %s.%s is overriding '
                          'existing optimizer %s.%s' %
                          (klass.__module__, klass.__name__,
                           Optimizer.opt_registry[name].__module__,
                           Optimizer.opt_registry[name].__name__))
        Optimizer.opt_registry[name] = klass
        return klass

    @staticmethod
    def create_optimizer(name, **kwargs):
        """Instantiates an optimizer with a given name and kwargs.

        .. note:: We can use the alias `create` for ``Optimizer.create_optimizer``.

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

    @property
    def learning_rate(self):
        if self.lr_scheduler is not None:
            return self.lr_scheduler(self.num_update)
        else:
            return self.lr

    def create_state(self, index, weight):
        """Creates auxiliary state for a given weight.

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

    def create_state_multi_precision(self, index, weight):
        """Creates auxiliary state for a given weight, including FP32 high
        precision copy if original weight is FP16.

        This method is provided to perform automatic mixed precision training
        for optimizers that do not support it themselves.

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
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (weight_master_copy,) + (self.create_state(index, weight_master_copy),)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "optimizer")
        return self.create_state(index, weight)

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

    def update_multi_precision(self, index, weight, grad, state):
        """Updates the given parameter using the corresponding gradient and state.
        Mixed precision version.

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
        if self.multi_precision and weight.dtype == numpy.float16:
            # Wrapper for mixed precision
            weight_master_copy = state[0]
            original_state = state[1]
            grad32 = grad.astype(numpy.float32)
            self.update(index, weight_master_copy, grad32, original_state)
            cast(weight_master_copy, dtype=weight.dtype, out=weight)
        else:
            self.update(index, weight, grad, state)

    def set_learning_rate(self, lr):
        """Sets a new learning rate of the optimizer.

        Parameters
        ----------
        lr : float
            The new learning rate of the optimizer.
        """
        if self.lr_scheduler is not None:
            raise UserWarning("LRScheduler of the optimizer has already been "
                              "defined. Note that set_learning_rate can mutate "
                              "the value of the learning rate of the optimizer "
                              "only when the LRScheduler of the optimizer is "
                              "undefined.")
        else:
            self.lr = lr

    def set_lr_scale(self, args_lrscale): # pylint: disable=unused-argument
        """[DEPRECATED] Sets lr scale. Use set_lr_mult instead."""
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
        if self.sym_info:
            attr, arg_names = self.sym_info
            for name in arg_names:
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
        if self.sym_info:
            attr, arg_names = self.sym_info
            for name in arg_names:
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def _update_count(self, index):
        """Updates num_update.

        Parameters
        ----------
        index : int
            The index to be updated.
        """
        if index not in self._index_update_count:
            self._index_update_count[index] = self.begin_num_update
        self._index_update_count[index] += 1
        self.num_update = max(self._index_update_count[index], self.num_update)

    def _get_lr(self, index):
        """Gets the learning rate given the index of the weight.

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

        if index in self.param_dict:
            lr *= self.param_dict[index].lr_mult
        elif index in self.lr_mult:
            lr *= self.lr_mult[index]
        elif index in self.idx2name:
            lr *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lr

    def _get_wd(self, index):
        """Gets weight decay for index.
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
        if index in self.param_dict:
            wd *= self.param_dict[index].wd_mult
        elif index in self.wd_mult:
            wd *= self.wd_mult[index]
        elif index in self.idx2name:
            wd *= self.wd_mult.get(self.idx2name[index], 1.0)
        return wd

# convenience wrapper for Optimizer.Register
register = Optimizer.register   # pylint: disable=invalid-name

# pylint: disable=line-too-long
@register
class SGD(Optimizer):
    """The SGD optimizer with momentum and weight decay.

    If the storage types of weight and grad are both ``row_sparse``, and ``lazy_update`` is True, \
    **lazy updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = lr * rescale_grad * clip(grad[row], clip_gradient) + wd * weight[row]
            state[row] = momentum[row] * state[row] + rescaled_grad[row]
            weight[row] = weight[row] - state[row]

    The sparse update only updates the momentum for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    Otherwise, **standard updates** are applied by::

        rescaled_grad = lr * rescale_grad * clip(grad, clip_gradient) + wd * weight
        state = momentum * state + rescaled_grad
        weight = weight - state

    For details of the update algorithm see
    :class:`~mxnet.ndarray.sgd_update` and :class:`~mxnet.ndarray.sgd_mom_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
    lazy_update : bool, optional
       Default is True. If True, lazy updates are applied \
       if the storage types of weight and grad are both ``row_sparse``.
    multi_precision: bool, optional
       Flag to control the internal precision of the optimizer.
       ``False`` results in using the same precision as the weights (default),
       ``True`` makes internal 32-bit copy of the weights and applies gradients \
                in 32-bit precision even if actual weights used in the model have lower precision.\
                Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, lazy_update=True, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.lazy_update = lazy_update

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "SGD optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        stype = weight.stype if self.lazy_update else 'default'
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def _update_impl(self, index, weight, grad, state, multi_precision=False):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if not multi_precision:
            if state is not None:
                sgd_mom_update(weight, grad, state, out=weight,
                               lr=lr, wd=wd, **kwargs)
            else:
                sgd_update(weight, grad, out=weight,
                           lr=lr, wd=wd, **kwargs)
        else:
            if state[0] is not None:
                mp_sgd_mom_update(weight, grad, state[0], state[1], out=weight,
                                  lr=lr, wd=wd, **kwargs)
            else:
                mp_sgd_update(weight, grad, state[1], out=weight,
                              lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        use_multi_precision = self.multi_precision and weight.dtype == numpy.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

@register
class Signum(Optimizer):
    """The Signum optimizer that takes the sign of gradient or momentum.

    The optimizer updates the weight by:

        rescaled_grad = rescale_grad * clip(grad, clip_gradient) + wd * weight
        state = momentum * state + (1-momentum)*rescaled_grad
        weight = (1 - lr * wd_lh) * weight - lr * sign(state)

    See the original paper at: https://jeremybernste.in/projects/amazon/signum.pdf

    For details of the update algorithm see
    :class:`~mxnet.ndarray.signsgd_update` and :class:`~mxnet.ndarray.signum_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
    wd_lh : float, optional
       The amount of decoupled weight decay regularization, see details in the original paper at:\
       https://arxiv.org/abs/1711.05101
    """
    def __init__(self, learning_rate=0.01, momentum=0.9, wd_lh=0.0, **kwargs):
        super(Signum, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum
        self.wd_lh = wd_lh

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=weight.stype)
        return momentum

    def _update_impl(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient
        if self.wd_lh:
            kwargs['wd_lh'] = self.wd_lh

        if state is not None:
            signum_update(weight, grad, state, out=weight,
                          lr=lr, wd=wd, **kwargs)
        else:
            signsgd_update(weight, grad, out=weight,
                           lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state)

@register
class FTML(Optimizer):
    """The FTML optimizer.

    This class implements the optimizer described in
    *FTML - Follow the Moving Leader in Deep Learning*,
    available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional
        0 < beta1 < 1. Generally close to 0.5.
    beta2 : float, optional
        0 < beta2 < 1. Generally close to 1.
    epsilon : float, optional
        Small value to avoid division by 0.
    """
    def __init__(self, beta1=0.6, beta2=0.999, epsilon=1e-8, **kwargs):
        super(FTML, self).__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype), # d_0
                zeros(weight.shape, weight.context, dtype=weight.dtype), # v_0
                zeros(weight.shape, weight.context, dtype=weight.dtype)) # z_0

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        t = self._index_update_count[index]

        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad, 't': t}
        if self.clip_gradient:
            kwargs['clip_grad'] = self.clip_gradient

        prev_d, prev_v, prev_z = state
        ftml_update(weight, grad, prev_d, prev_v, prev_z, out=weight,
                    lr=lr, wd=wd, **kwargs)

@register
class LBSGD(Optimizer):
    """The Large Batch SGD optimizer with momentum and weight decay.

    The optimizer updates the weight by::

        state = momentum * state + lr * rescale_grad * clip(grad, clip_gradient) + wd * weight
        weight = weight - state

    For details of the update algorithm see :class:`~mxnet.ndarray.lbsgd_update` and
    :class:`~mxnet.ndarray.lbsgd_mom_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
    multi_precision: bool, optional
       Flag to control the internal precision of the optimizer.
       ``False`` results in using the same precision as the weights (default),
       ``True`` makes internal 32-bit copy of the weights and applies gradients
                in 32-bit precision even if actual weights used in the model have lower precision.`<
                Turning this on can improve convergence and accuracy when training with float16.
    warmup_strategy: string ('linear', 'power2', 'sqrt'. , 'lars'   default : 'linear')
    warmup_epochs: unsigned, default: 5
    batch_scale:   unsigned, default: 1 (same as batch size*numworkers)
    updates_per_epoch: updates_per_epoch (default: 32, Default might not reflect true number batches per epoch. Used for warmup.)
    begin_epoch: unsigned, default 0, starting epoch.
    """

    def __init__(self, momentum=0.0, multi_precision=False, warmup_strategy='linear',
                 warmup_epochs=5, batch_scale=1, updates_per_epoch=32, begin_epoch=0, num_epochs=60,
                 **kwargs):
        super(LBSGD, self).__init__(**kwargs)
        logging.info('Running Large-Batch SGD Algorithm')
        logging.info('(Batch_scale=%f, warmup_epochs=%d, warmup_strategy=%s, updates_per_epoch=%d)',
                     batch_scale, warmup_epochs, warmup_strategy, updates_per_epoch)
        self.momentum = momentum
        self.multi_precision = multi_precision
        # new user parameters for large batch
        self.warmup_strategy = warmup_strategy
        self.warmup_epochs = warmup_epochs
        self.batch_scale = batch_scale
        self.updates_per_epoch = updates_per_epoch
        self.init_updates = begin_epoch * updates_per_epoch
        self.num_epochs = num_epochs
        # addl internal usage parameters and storage
        self.lbmult = 1
        self.cumgrads = {}
        # for adaptive lr
        self.adaptive = False
        self.admult = 1  # adaptation constant

    def create_state(self, index, weight):
        momentum = None
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = array(weight, ctx=weight.context, dtype=numpy.float32)
            if self.momentum != 0.0:
                momentum = zeros(weight.shape, weight.context, dtype=numpy.float32,
                                 stype=weight.stype)
            return (momentum, weight_master_copy)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "SGD optimizer")
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=weight.stype)
        return momentum

    def _get_lbmult(self, nup):
        """Returns lr scaling factor for large batch according to warmup schedule
        (to be implemented)
        """
        nwup = self.warmup_epochs * self.updates_per_epoch
        strategy = self.warmup_strategy
        maxmult = float(self.batch_scale)
        if nup >= nwup:
            mult = maxmult
        elif nwup <= 1:
            mult = 1.0
        else:
            if (strategy == 'linear'):
                mult = 1.0 + (maxmult - 1) * nup / nwup
            elif (strategy == 'power2'):
                mult = 1.0 + (maxmult-1) * (nup*nup)/(nwup*nwup)
            elif (strategy == 'sqrt'):
                mult = 1.0 + (maxmult - 1) * math.sqrt(float(nup) / nwup)
            else:
                mult = 1.0
        return mult

    def _get_lars(self, weight, g, wd):
        """Returns a scaling factor for the learning rate for this layer
        default is 1
        """
        weight2 = self._l2norm(weight)
        grad2 = self._l2norm(g)
        lars = math.sqrt(weight2 / (grad2 + wd * weight2 + 1e-18))
        if lars < 0.01:
            lars = 0.01
        elif lars > 100:
            lars = 100
        return lars

    def _l2norm(self, v):
        "inner product implementation"
        norm = multiply(v, v).asnumpy().sum()
        return norm

    def _reset_cum_gradient(self, index):
        "called every macro-batch to reset cumulated gradients to 0 for a given index"
        self.cumgrads[index]['cum_grad'] = 0

    def _get_cum_gradient(self, index):
        "get the cumulated gradient for index"
        if index in self.cumgrads:
            return self.cumgrads[index]
        else:
            return {}

    def _put_cum_gradient(self, index, cgrad):
        "store cumulated gradient for index"
        self.cumgrads[index] = cgrad

    def _cumulate_gradient(self, grad, index):
        "Cumulate gradients for large-batch emulation. Cumulated by index (layer)"
        cgrad = self._get_cum_gradient(index)
        if cgrad:
            num_cums = cgrad['num_cums']
            if num_cums > 0:
                cum_grad = cgrad['cum_grad'] + grad
                num_cums += 1
            else:
                cum_grad = grad
                num_cums = self.init_updates + 1
        else:
            cum_grad = grad
            num_cums = self.init_updates + 1
        cgrad = {'cum_grad': cum_grad, 'num_cums': num_cums}
        self._put_cum_gradient(index, cgrad)
        return cgrad

    def update(self, index, weight, grad, state):
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))

        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        # new stuff for large batch
        cgrad = self._cumulate_gradient(grad, index)
        if (cgrad['num_cums'] % self.batch_scale) == 0:
            grad = cgrad['cum_grad'] / self.batch_scale
            if self.warmup_strategy == 'lars':
                lbmult = self._get_lars(weight, grad, wd)
            else:
                lbmult = self._get_lbmult(cgrad['num_cums'])
            lr = lr * lbmult
            # do the regular sgd update flow
            kwargs = {'rescale_grad': self.rescale_grad}
            if self.momentum > 0:
                kwargs['momentum'] = self.momentum
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            use_multi_precision = isinstance(state, (list, tuple))

            if not use_multi_precision:
                if state is not None:
                    sgd_mom_update(weight, grad, state, out=weight, lr=lr, wd=wd, **kwargs)
                else:
                    sgd_update(weight, grad, out=weight, lr=lr, wd=wd, **kwargs)
            else:
                if state[0] is not None:
                    mp_sgd_mom_update(weight, grad, state[0], state[1], out=weight, lr=lr, wd=wd,
                                      **kwargs)
                else:
                    mp_sgd_update(weight, grad, state[1], out=weight, lr=lr, wd=wd, **kwargs)
            # reset update count and cumulated gradient per large batch
            self._reset_cum_gradient(index)
        else:
            lr = 0.0
            kwargs = {}
            sgd_update(weight, grad, out=weight, lr=lr, wd=wd, **kwargs)

# pylint: enable=line-too-long
@register
class DCASGD(Optimizer):
    """The DCASGD optimizer.

    This class implements the optimizer described in *Asynchronous Stochastic Gradient Descent
    with Delay Compensation for Distributed Deep Learning*,
    available at https://arxiv.org/abs/1609.08326.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

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
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

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
@register
class NAG(Optimizer):
    """Nesterov accelerated SGD.

    This optimizer updates each weight by::

        state = momentum * state + grad + wd * weight
        weight = weight - (lr * (grad + momentum * state))

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
    multi_precision: bool, optional
       Flag to control the internal precision of the optimizer.
       ``False`` results in using the same precision as the weights (default),
       ``True`` makes internal 32-bit copy of the weights and applies gradients \
                in 32-bit precision even if actual weights used in the model have lower precision.\
                Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, **kwargs):
        super(NAG, self).__init__(**kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
        return momentum

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

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
    https://papers.nips.cc/paper/4883-stochastic-gradient-riemannian-langevin-dynamics-on-the-probability-simplex.pdf.

    """
    def __init__(self, **kwargs):
        super(SGLD, self).__init__(**kwargs)

    def create_state(self, index, weight):
        return None

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        weight[:] += - lr/2 * (grad + wd * weight) + normal(0, math.sqrt(lr), shape=weight.shape,
                                                            dtype=weight.dtype, ctx=weight.context)



@register  # pylint: disable=invalid-name
class ccSGD(SGD):
    """[DEPRECATED] Same as `SGD`. Left here for backward compatibility."""
    def __init__(self, *args, **kwargs):
        super(ccSGD, self).__init__(*args, **kwargs)

@register
class Adam(Optimizer):
    """The Adam optimizer.

    This class implements the optimizer described in *Adam: A Method for
    Stochastic Optimization*, available at http://arxiv.org/abs/1412.6980.

    If the storage types of weight and grad are both ``row_sparse``, and ``lazy_update`` is True, \
    **lazy updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = clip(grad[row] * rescale_grad + wd * weight[row], clip_gradient)
            m[row] = beta1 * m[row] + (1 - beta1) * rescaled_grad[row]
            v[row] = beta2 * v[row] + (1 - beta2) * (rescaled_grad[row]**2)
            w[row] = w[row] - learning_rate * m[row] / (sqrt(v[row]) + epsilon)

    The lazy update only updates the mean and var for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all indices.
    Compared with the original update, it can provide large improvements in model training
    throughput for some applications. However, it provides slightly different semantics than
    the original update, and may lead to different empirical results.

    Otherwise, **standard updates** are applied by::

        rescaled_grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        w = w - learning_rate * m / (sqrt(v) + epsilon)

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    For details of the update algorithm, see :class:`~mxnet.ndarray.adam_update`.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0.
    lazy_update : bool, optional
       Default is True. If True, lazy updates are applied \
       if the storage types of weight and grad are both ``row_sparse``.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 lazy_update=True, **kwargs):
        super(Adam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
        stype = weight.stype if self.lazy_update else 'default'
        return (zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype))  # variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]
        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1

        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        mean, var = state
        adam_update(weight, grad, mean, var, out=weight,
                    lr=lr, wd=wd, **kwargs)

@register
class AdaGrad(Optimizer):
    """AdaGrad optimizer.

    This class implements the AdaGrad optimizer described in *Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization*, and available at
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    eps: float, optional
        Small value to avoid division by 0.

    """
    def __init__(self, eps=1e-7, **kwargs):
        super(AdaGrad, self).__init__(**kwargs)
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        return zeros(weight.shape, weight.context, stype=weight.stype)  # history

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        is_sparse = True if weight.stype == 'row_sparse' and grad.stype == 'row_sparse' else False

        if is_sparse is True:
            grad_indices_count = len(grad.indices)

        grad = grad * self.rescale_grad

        if is_sparse is True:
            grad_indices = grad.indices
            # Make sure that the scalar multiply still has a sparse result
            assert grad_indices_count == len(grad_indices)

        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        history = state
        save_history_stype = history.stype

        if is_sparse:
            history[:] = sparse.elemwise_add(sparse.square(grad),
                                             sparse.retain(history, grad_indices))
            history_indices = history.indices
            assert len(history_indices) == grad_indices_count
            adjusted_add = _internal._scatter_plus_scalar(history, self.float_stable_eps)
            srt = op.sqrt(adjusted_add)
            div = _internal._scatter_elemwise_div(grad, srt)
            retained_weight = sparse.retain(weight, grad.indices)
            to_add = sparse.elemwise_add(div, _internal._mul_scalar(retained_weight, float(wd)))
            assert len(to_add.indices) == grad_indices_count
            weight[:] = sparse.elemwise_add(weight, _internal._mul_scalar(to_add, float(-lr)))
            state[:] = history
            assert state.stype == save_history_stype
            assert len(history_indices) == grad_indices_count
        else:
            history[:] += square(grad)
            div = grad / sqrt(history + self.float_stable_eps)
            weight[:] += (div + weight * wd) * -lr

@register
class RMSProp(Optimizer):
    """The RMSProp optimizer.

    Two versions of RMSProp are implemented:

    If ``centered=False``, we follow
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf by
    Tieleman & Hinton, 2012.
    For details of the update algorithm see :class:`~mxnet.ndarray.rmsprop_update`.

    If ``centered=True``, we follow http://arxiv.org/pdf/1308.0850v5.pdf (38)-(45)
    by Alex Graves, 2013.
    For details of the update algorithm see :class:`~mxnet.ndarray.rmspropalex_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

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
        Clips weights into range ``[-clip_weights, clip_weights]``.
    """
    def __init__(self, learning_rate=0.001, gamma1=0.9, gamma2=0.9,
                 epsilon=1e-8, centered=False, clip_weights=None, **kwargs):
        super(RMSProp, self).__init__(learning_rate=learning_rate, **kwargs)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.centered = centered
        self.epsilon = epsilon
        self.clip_weights = clip_weights

    def create_state(self, index, weight):
        if self.centered:
            return (
                zeros(weight.shape, weight.context, stype=weight.stype),  # n
                zeros(weight.shape, weight.context, stype=weight.stype),  # g
                zeros(weight.shape, weight.context, stype=weight.stype))  # delta
        else:
            return (zeros(weight.shape, weight.context, stype=weight.stype),)  # n

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        kwargs = {'gamma1': self.gamma1, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.centered:
            kwargs['gamma2'] = self.gamma2
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient
        if self.clip_weights:
            kwargs['clip_weights'] = self.clip_weights

        if not self.centered:
            (n, ) = state
            rmsprop_update(
                weight, grad, n, out=weight, lr=lr, wd=wd, **kwargs)
        else:
            n, g, delta = state
            rmspropalex_update(weight, grad, n, g, delta, out=weight,
                               lr=lr, wd=wd, **kwargs)

@register
class AdaDelta(Optimizer):
    """The AdaDelta optimizer.

    This class implements AdaDelta, an optimizer described in  *ADADELTA: An adaptive
    learning rate method*, available at https://arxiv.org/abs/1212.5701.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

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
#pylint: disable=line-too-long
@register
class Ftrl(Optimizer):
    """The Ftrl optimizer.

    Referenced from *Ad Click Prediction: a View from the Trenches*, available at
    http://dl.acm.org/citation.cfm?id=2488200.

    eta :
        .. math::
           \\eta_{t,i} = \\frac{learningrate}{\\beta+\\sqrt{\\sum_{s=1}^tg_{s,i}^2}}

    The optimizer updates the weight by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
        n += rescaled_grad**2
        w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)

    If the storage types of weight, state and grad are all ``row_sparse``, \
    **sparse updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
            z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
            n[row] += rescaled_grad[row]**2
            w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)

    The sparse update only updates the z and n for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    For details of the update algorithm, see :class:`~mxnet.ndarray.ftrl_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    lamda1 : float, optional
        L1 regularization coefficient.
    learning_rate : float, optional
        The initial learning rate.
    beta : float, optional
        Per-coordinate learning rate correlation parameter.
    """

    def __init__(self, lamda1=0.01, learning_rate=0.1, beta=1, **kwargs):
        super(Ftrl, self).__init__(**kwargs)
        self.lamda1 = lamda1
        self.beta = beta
        self.lr = learning_rate

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, stype=weight.stype),  # z
                zeros(weight.shape, weight.context, stype=weight.stype))  # n

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        wd = self._get_wd(index)
        lr = self._get_lr(index)

        kwargs = {'lamda1': self.lamda1, 'beta': self.beta, 'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        # accumulated g and delta initialization
        z, n = state
        ftrl_update(weight, grad, z, n, out=weight,
                    lr=lr, wd=wd, **kwargs)

# pylint: enable=line-too-long
@register
class Adamax(Optimizer):
    """The AdaMax optimizer.

    It is a variant of Adam based on the infinity norm
    available at http://arxiv.org/abs/1412.6980 Section 7.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    """
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, **kwargs):
        super(Adamax, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]
        lr /= (1. - self.beta1**t)

        # preprocess grad
        grad = grad * self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # update m_t and u_t
        m_t, u_t = state
        m_t[:] = self.beta1 * m_t + (1. - self.beta1) * grad
        u_t[:] = maximum(self.beta2 * u_t, NDabs(grad))

        # update weight
        weight[:] -= lr * m_t / u_t

@register
class Nadam(Optimizer):
    """The Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum available
    at http://cs229.stanford.edu/proj2015/054_report.pdf.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0.
    schedule_decay : float, optional
        Exponential decay rate for the momentum schedule
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 schedule_decay=0.004, **kwargs):
        super(Nadam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
        self.m_schedule = 1.

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]

        # preprocess grad
        grad = grad * self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # warming momentum schedule
        momentum_t = self.beta1 * (1. - 0.5 * (pow(0.96, t * self.schedule_decay)))
        momentum_t_1 = self.beta1 * (1. - 0.5 * (pow(0.96, (t + 1) * self.schedule_decay)))
        self.m_schedule = self.m_schedule * momentum_t
        m_schedule_next = self.m_schedule * momentum_t_1

        # update m_t and v_t
        m_t, v_t = state
        m_t[:] = self.beta1 * m_t + (1. - self.beta1) * grad
        v_t[:] = self.beta2 * v_t + (1. - self.beta2) * grad * grad

        grad_prime = grad / (1. - self.m_schedule)
        m_t_prime = m_t / (1. - m_schedule_next)
        v_t_prime = v_t / (1. - pow(self.beta2, t))
        m_t_bar = (1. - momentum_t) * grad_prime + momentum_t_1 * m_t_prime

        # update weight
        weight[:] -= lr * m_t_bar / (sqrt(v_t_prime) + self.epsilon)

@register
class Test(Optimizer):
    """The Test optimizer"""
    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)

    def create_state(self, index, weight):
        """Creates a state to duplicate weight."""
        return zeros(weight.shape, weight.context)

    def update(self, index, weight, grad, state):
        """Performs w += rescale_grad * grad."""
        weight[:] += grad * self.rescale_grad
        state[:] = weight

# backward compatibility wrapper for Optimizer.CreateOptimizer
create = Optimizer.create_optimizer  # pylint: disable=invalid-name

class Updater(object):
    """Updater for kvstore."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.states = {}
        self.states_synced = {}

    def __call__(self, index, grad, weight):
        """Updates weight given gradient and index."""
        # convert ctypes.char_p.value back to python str if needed
        if isinstance(index, bytes):
            index = py_str(index)
        if index not in self.states:
            self.states[index] = self.optimizer.create_state_multi_precision(index, weight)
            self.states_synced[index] = True
        elif not self.states_synced[index]:
            self.states[index] = \
                self.sync_state_context(self.states[index], weight.context)
            self.states_synced[index] = True
        self.optimizer.update_multi_precision(index, weight, grad, self.states[index])

    def sync_state_context(self, state, context):
        """sync state context."""
        if isinstance(state, NDArray):
            return state.as_in_context(context)
        elif isinstance(state, (tuple, list)):
            synced_state = (self.sync_state_context(i, context) for i in state)
            if isinstance(state, tuple):
                return tuple(synced_state)
            else:
                return list(synced_state)
        else:
            return state

    def set_states(self, states):
        """Sets updater states."""
        states = pickle.loads(states)
        if isinstance(states, tuple) and len(states) == 2:
            self.states, self.optimizer = states
        else:
            self.states = states
        self.states_synced = dict.fromkeys(self.states.keys(), False)

    def get_states(self, dump_optimizer=False):
        """Gets updater states.

        Parameters
        ----------
        dump_optimizer : bool, default False
            Whether to also save the optimizer itself. This would also save optimizer
            information such as learning rate and weight decay schedules.
        """
        return pickle.dumps((self.states, self.optimizer) if dump_optimizer else self.states)

def get_updater(optimizer):
    """Returns a closure of the updater needed for kvstore.

    Parameters
    ----------
    optimizer: Optimizer
         The optimizer.

    Returns
    -------
    updater: function
         The closure of the updater.
    """
    return Updater(optimizer)
