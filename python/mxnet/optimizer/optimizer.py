# coding: utf-8
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

# pylint: disable=too-many-lines
"""Weight updating functions."""
from __future__ import absolute_import
import logging
import math
import pickle
import warnings
import os
import numpy
from ..base import py_str
from ..ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply)
from ..ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
                       mp_sgd_update, mp_sgd_mom_update, square, ftrl_update, ftml_update,
                       signsgd_update, signum_update, nag_mom_update, mp_nag_mom_update,
                       multi_sgd_update, multi_sgd_mom_update, multi_mp_sgd_update,
                       multi_mp_sgd_mom_update)
from ..ndarray import sparse
from ..random import normal

__all__ = [
    'AdaDelta', 'AdaGrad', 'Adam', 'Adamax', 'DCASGD', 'FTML', 'Ftrl', 'LBSGD',
    'NAG', 'NDabs', 'Nadam', 'Optimizer', 'RMSProp', 'SGD', 'SGLD', 'Signum',
    'Test', 'Updater', 'ccSGD', 'create', 'get_updater', 'register'
]

def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

class Optimizer(object):
    """The base class inherited by all optimizers.

    Parameters
    ----------
    rescale_grad : float, optional, default 1.0
        Multiply the gradient with `rescale_grad` before updating. Often
        choose to be ``1.0/batch_size``.

    param_idx2name : dict from int to string, optional, default None
        A dictionary that maps int index to string name.

    clip_gradient : float, optional, default None
        Clip the gradient by projecting onto the box ``[-clip_gradient, clip_gradient]``.

    learning_rate : float, optional, default 0.01
        The initial learning rate.

    lr_scheduler : LRScheduler, optional, default None
        The learning rate scheduler.

    wd : float, optional, default 0.0
        The weight decay (or L2 regularization) coefficient. Modifies objective
        by adding a penalty for having large weights.

    sym: Symbol, optional, default None
        The Symbol this optimizer is applying to.

    begin_num_update : int, optional, default 0
        The initial number of updates.

    multi_precision : bool, optional, default False
       Flag to control the internal precision of the optimizer.
       False: results in using the same precision as the weights (default),
       True: makes internal 32-bit copy of the weights and applies gradients
       in 32-bit precision even if actual weights used in the model have lower precision.
       Turning this on can improve convergence and accuracy when training with float16.

    param_dict : dict of int -> gluon.Parameter, default None
        Dictionary of parameter index to gluon.Parameter, used to lookup parameter attributes
        such as lr_mult, wd_mult, etc. param_dict shall not be deep copied.

    Properties
    ----------
    learning_rate : float
        The current learning rate of the optimizer. Given an Optimizer object
        optimizer, its learning rate can be accessed as optimizer.learning_rate.
    """
    def __init__(self, rescale_grad=1., param_idx2name=None, wd=0.,
                 clip_gradient=None, learning_rate=0.01,
                 lr_scheduler=None, sym=None, begin_num_update=0,
                 multi_precision=False, param_dict=None, allow_np=False):
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
        self._all_index_update_counts = {0 : {}}
        self._index_update_count = self._all_index_update_counts[0]
        self.clip_gradient = clip_gradient
        self.multi_precision = multi_precision
        self.aggregate_num = 0

        if param_idx2name is None:
            param_idx2name = {}
        assert isinstance(param_idx2name, dict), \
            'param_idx2name should be a dict of param indexes to names.'
        self.idx2name = param_idx2name.copy()
        self.sym_info = (sym.attr_dict(), sym.list_arguments()) if sym is not None else ()
        self.param_dict = param_dict if param_dict else {}
        self.allow_np = allow_np

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
        if self.lr_scheduler is not None: # pylint: disable=no-else-raise
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

    def _set_current_context(self, device_id):
        """Sets the number of the currently handled device.

        Parameters
        ----------
        device_id : int
            The number of current device.
        """
        if device_id not in self._all_index_update_counts:
            self._all_index_update_counts[device_id] = {}
        self._index_update_count = self._all_index_update_counts[device_id]

    def _update_count(self, index):
        """Updates num_update.

        Parameters
        ----------
        index : int or list of int
            The index to be updated.
        """
        if not isinstance(index, (list, tuple)):
            index = [index]
        for idx in index:
            if idx not in self._index_update_count:
                self._index_update_count[idx] = self.begin_num_update
            self._index_update_count[idx] += 1
            self.num_update = max(self._index_update_count[idx], self.num_update)

    def _get_lrs(self, indices):
        """Gets the learning rates given the indices of the weights.

        Parameters
        ----------
        indices : list of int
            Indices corresponding to weights.

        Returns
        -------
        lrs : list of float
            Learning rates for those indices.
        """
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.num_update)
        else:
            lr = self.lr

        lrs = [lr for _ in indices]
        for i, index in enumerate(indices):
            if index in self.param_dict:
                lrs[i] *= self.param_dict[index].lr_mult
            elif index in self.lr_mult:
                lrs[i] *= self.lr_mult[index]
            elif index in self.idx2name:
                lrs[i] *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lrs

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
        return self._get_lrs([index])[0]

    def _get_wds(self, indices):
        """Gets weight decays for indices.
        Returns 0 for non-weights if the name of weights are provided for `__init__`.

        Parameters
        ----------
        indices : list of int
            Indices of weights.

        Returns
        -------
        wds : list of float
            Weight decays for those indices.
        """
        wds = [self.wd for _ in indices]
        for i, index in enumerate(indices):
            if index in self.param_dict:
                wds[i] *= self.param_dict[index].wd_mult
            elif index in self.wd_mult:
                wds[i] *= self.wd_mult[index]
            elif index in self.idx2name:
                wds[i] *= self.wd_mult.get(self.idx2name[index], 1.0)
        return wds

    def _get_wd(self, index):
        """Gets weight decay for index.
        Returns 0 for non-weights if the name of weights are provided for `__init__`.

        Parameters
        ----------
        index : int
            The index of weight.

        Returns
        -------
        wd : float
            Weight decay for this index.
        """
        return self._get_wds([index])[0]

    def __getstate__(self):
        ret = self.__dict__.copy()
        # do not include param_dict in the state
        del ret['param_dict']
        return ret

    def __setstate__(self, state):
        self.__dict__ = state
        # param_dict needs to be explicitly set by the trainer
        self.param_dict = {}

# convenience wrapper for Optimizer.Register
register = Optimizer.register   # pylint: disable=invalid-name

# pylint: disable=line-too-long
@register
class SGD(Optimizer):
    """The SGD optimizer with momentum and weight decay.

    If the storage types of grad is ``row_sparse`` and ``lazy_update`` is True, \
    **lazy updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = lr * (rescale_grad * clip(grad[row], clip_gradient) + wd * weight[row])
            state[row] = momentum[row] * state[row] + rescaled_grad[row]
            weight[row] = weight[row] - state[row]

    The sparse update only updates the momentum for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    In the case when ``update_on_kvstore`` is set to False (either globally via
    MXNET_UPDATE_ON_KVSTORE=0 environment variable or as a parameter in
    :class:`~mxnet.gluon.Trainer`) SGD optimizer can perform aggregated update
    of parameters, which may lead to improved performance. The aggregation size
    is controlled by MXNET_OPTIMIZER_AGGREGATION_SIZE environment variable and
    defaults to 4.

    Otherwise, **standard updates** are applied by::

        rescaled_grad = lr * (rescale_grad * clip(grad, clip_gradient) + wd * weight)
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
        False: results in using the same precision as the weights (default),
        True: makes internal 32-bit copy of the weights and applies gradients
        in 32-bit precision even if actual weights used in the model have lower precision.
        Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, lazy_update=True, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.lazy_update = lazy_update
        self.aggregate_num = int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', "4"))

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
        if self.momentum != 0.0:
            stype = weight.stype if self.lazy_update else 'default'
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def _update_impl(self, indices, weights, grads, states, multi_precision=False):
        aggregate = True
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        for weight, grad in zip(weights, grads):
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            if not multi_precision:
                if self.momentum > 0:
                    multi_sgd_mom_update(*_flatten_list(zip(weights, grads, states)), out=weights,
                                         num_weights=len(weights), lrs=lrs, wds=wds, **kwargs)
                else:
                    multi_sgd_update(*_flatten_list(zip(weights, grads)), out=weights,
                                     num_weights=len(weights), lrs=lrs, wds=wds, **kwargs)
            else:
                if self.momentum > 0:
                    multi_mp_sgd_mom_update(*_flatten_list(zip(weights, grads, *zip(*states))),
                                            out=weights, num_weights=len(weights),
                                            lrs=lrs, wds=wds, **kwargs)
                else:
                    multi_mp_sgd_update(*_flatten_list(zip(weights, grads,
                                                           list(zip(*states))[1])),
                                        out=weights, num_weights=len(weights),
                                        lrs=lrs, wds=wds, **kwargs)
        else:
            for weight, grad, state, lr, wd in zip(weights, grads, states, lrs, wds):
                if not multi_precision:
                    if state is not None:
                        sgd_mom_update(weight, grad, state, out=weight,
                                       lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)
                    else:
                        sgd_update(weight, grad, out=weight, lazy_update=self.lazy_update,
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
        if not isinstance(index, (tuple, list)):
            use_multi_precision = self.multi_precision and weight.dtype == numpy.float16
        else:
            use_multi_precision = self.multi_precision and weight[0].dtype == numpy.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

@register
class Signum(Optimizer):
    r"""The Signum optimizer that takes the sign of gradient or momentum.

    The optimizer updates the weight by::

        rescaled_grad = rescale_grad * clip(grad, clip_gradient) + wd * weight
        state = momentum * state + (1-momentum)*rescaled_grad
        weight = (1 - lr * wd_lh) * weight - lr * sign(state)

    References
    ----------
    Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli & Anima Anandkumar. (2018).
    signSGD: Compressed Optimisation for Non-Convex Problems. In ICML'18.

    See: https://arxiv.org/abs/1802.04434

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

    Denote time step by t. The optimizer updates the weight by::

        rescaled_grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        v = beta2 * v + (1 - beta2) * square(rescaled_grad)
        d_t = (1 - power(beta1, t)) / lr * square_root(v / (1 - power(beta2, t))) + epsilon)
        z = beta1 * z + (1 - beta1) * rescaled_grad - (d_t - beta1 * d_(t-1)) * weight
        weight = - z / d_t

    For details of the update algorithm, see :class:`~mxnet.ndarray.ftml_update`.

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

    For details of the update algorithm see :class:`~mxnet.ndarray.sgd_update`
    and :class:`~mxnet.ndarray.sgd_mom_update`.
    In addition to the SGD updates the LBSGD optimizer uses the LARS, Layer-wise
    Adaptive Rate Scaling, algorithm to have a separate learning rate for each
    layer of the network, which leads to better stability over large batch sizes.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
        The momentum value.
    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.
        False: results in using the same precision as the weights (default),
        True: makes internal 32-bit copy of the weights and applies gradients
        in 32-bit precision even if actual weights used in the model have lower precision.
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
class NAG(Optimizer):
    """Nesterov accelerated gradient.

    This optimizer updates each weight by::

        state = momentum * state + grad + wd * weight
        weight = weight - (lr * (grad + momentum * state))

    Parameters
    ----------
    momentum : float, optional
       The momentum value.
    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.
        False: results in using the same precision as the weights (default),
        True: makes internal 32-bit copy of the weights and applies gradients
        in 32-bit precision even if actual weights used in the model have lower precision.
        Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, **kwargs):
        super(NAG, self).__init__(**kwargs)
        self.momentum = momentum

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "NAG optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
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
                nag_mom_update(weight, grad, state, out=weight, lr=lr, wd=wd, **kwargs)
            else:
                sgd_update(weight, grad, out=weight, lr=lr, wd=wd, **kwargs)
        else:
            if state[0] is not None:
                mp_nag_mom_update(weight, grad, state[0], state[1], out=weight,
                                  lr=lr, wd=wd, **kwargs)
            else:
                mp_sgd_update(weight, grad, state[1], out=weight,
                              lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        use_multi_precision = self.multi_precision and weight.dtype == numpy.float16 \
                                and isinstance(state, (tuple, list))
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)


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
        weight[:] += - lr/2 * (grad + wd * weight)
        weight[:] += normal(0, math.sqrt(lr), shape=weight.shape,
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

    If the storage types of grad is ``row_sparse``, and ``lazy_update`` is True, \
    **lazy updates** at step t are applied by::

        for row in grad.indices:
            rescaled_grad[row] = clip(grad[row] * rescale_grad + wd * weight[row], clip_gradient)
            m[row] = beta1 * m[row] + (1 - beta1) * rescaled_grad[row]
            v[row] = beta2 * v[row] + (1 - beta2) * (rescaled_grad[row]**2)
            lr = learning_rate * sqrt(1 - beta1**t) / (1 - beta2**t)
            w[row] = w[row] - lr * m[row] / (sqrt(v[row]) + epsilon)

    The lazy update only updates the mean and var for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all indices.
    Compared with the original update, it can provide large improvements in model training
    throughput for some applications. However, it provides slightly different semantics than
    the original update, and may lead to different empirical results.

    Otherwise, **standard updates** at step t are applied by::

        rescaled_grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        lr = learning_rate * sqrt(1 - beta1**t) / (1 - beta2**t)
        w = w - lr * m / (sqrt(v) + epsilon)

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
                    lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)

@register
class AdaGrad(Optimizer):
    """AdaGrad optimizer.

    This class implements the AdaGrad optimizer described in *Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization*, and available at
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad, clip_gradient)
        history += square(grad)
        div = grad / sqrt(history + float_stable_eps)
        weight += (div + weight * wd) * -lr

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    See Also
    ----------
    :meth:`mxnet.ndarray.sparse.adagrad_update`.

    Parameters
    ----------
    eps: float, optional
        Initial value of the history accumulator. Avoids division by 0.

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

        is_sparse = grad.stype == 'row_sparse'
        history = state

        if is_sparse:
            kwargs = {'epsilon': self.float_stable_eps,
                      'rescale_grad': self.rescale_grad}
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            sparse.adagrad_update(weight, grad, history, out=weight, lr=lr, wd=wd, **kwargs)
        else:
            grad = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
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
        Flag to control which version of RMSProp to use.::

            True: will use Graves's version of `RMSProp`,
            False: will use Tieleman & Hinton's version of `RMSProp`.

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

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        acc_grad = rho * acc_grad + (1. - rho) * grad * grad
        delta = sqrt(acc_delta + epsilon) / sqrt(acc_grad + epsilon) * grad
        acc_delta = rho * acc_delta + (1. - rho) * delta * delta
        weight -= (delta + wd * weight)

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
            grad = clip(grad, - self.clip_gradient, self.clip_gradient)

        # accumulated g and delta initlization
        acc_g, acc_delta = state

        # update g, delta
        acc_g[:] *= self.rho
        acc_g[:] += (1. - self.rho) * grad * grad
        current_delta = sqrt(acc_delta + self.epsilon) / sqrt(acc_g + self.epsilon) * grad
        acc_delta[:] *= self.rho
        acc_delta[:] += (1. - self.rho) * current_delta * current_delta

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

    The optimizer updates the weight by::

        grad = clip(grad * rescale_grad + wd * weight, clip_gradient)
        m = beta1 * m_t + (1 - beta1) * grad
        u = maximum(beta2 * u, abs(grad))
        weight -= lr / (1 - beta1**t) * m / u

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
        m_t[:] *= self.beta1
        m_t[:] += (1. - self.beta1) * grad
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
        m_t[:] *= self.beta1
        m_t[:] += (1. - self.beta1) * grad
        v_t[:] *= self.beta2
        v_t[:] += (1. - self.beta2) * grad * grad

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


def _as_classic(a, allow_np):
    from ..numpy import ndarray as np_ndarray
    if isinstance(a, (tuple, list)):
        if any(isinstance(x, np_ndarray) for x in a):
            if allow_np:
                return [x.as_classic_ndarray() for x in a]
            else:
                raise ValueError('Converting np.ndarray to mx.nd.NDArray is not allowed')
    else:
        if isinstance(a, np_ndarray):
            if allow_np:
                return a.as_classic_ndarray()
            else:
                raise ValueError('Converting np.ndarray to mx.nd.NDArray is not allowed')
    return a



class Updater(object):
    """Updater for kvstore."""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.states = {}
        self.states_synced = {}
        self.aggregate_updates = optimizer.aggregate_num > 0

    def __call__(self, index, grad, weight):
        """Updates weight given gradient and index."""
        allow_np = self.optimizer.allow_np
        if not isinstance(index, (list, tuple)):
            indices = [index]
            grads = [_as_classic(grad, allow_np)]
            weights = [_as_classic(weight, allow_np)]
        else:
            indices = index
            grads = _as_classic(grad, allow_np)
            weights = _as_classic(weight, allow_np)
        if weights:
            self.optimizer._set_current_context(weights[0].context.device_id)
        for i, idx in enumerate(indices):
            # convert ctypes.char_p.value back to python str if needed
            if isinstance(idx, bytes):
                indices[i] = py_str(idx)
                idx = indices[i]
            if idx not in self.states:
                self.states[idx] = self.optimizer.create_state_multi_precision(idx, weights[i])
                self.states_synced[idx] = True
            elif not self.states_synced[idx]:
                self.states[idx] = \
                    self.sync_state_context(self.states[idx], weights[i].context)
                self.states_synced[idx] = True
        if self.aggregate_updates:
            # segregate values based on type
            type_map = {}
            for i, w, g in zip(indices, weights, grads):
                if w.dtype in type_map:
                    type_map[w.dtype].append((i, w, g))
                else:
                    type_map[w.dtype] = [(i, w, g)]
            for idx in type_map:
                current_index = 0
                indices, weights, grads = zip(*type_map[idx])
                while current_index < len(indices):
                    states = []
                    step = min(self.optimizer.aggregate_num, len(indices) - current_index)
                    for j in range(step):
                        states.append(self.states[indices[current_index + j]])
                    self.optimizer.update_multi_precision(
                        indices[current_index:current_index + self.optimizer.aggregate_num],
                        weights[current_index:current_index + self.optimizer.aggregate_num],
                        grads[current_index:current_index + self.optimizer.aggregate_num],
                        states)
                    current_index += self.optimizer.aggregate_num
        else:
            for i, w, g in zip(indices, weights, grads):
                self.optimizer.update_multi_precision(i, w, g, self.states[i])

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
