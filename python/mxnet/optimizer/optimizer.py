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
"""Base Optimizer class."""
import warnings
import numpy
from ..ndarray import (NDArray, zeros, cast)
from ..util import is_np_array

__all__ = ['Optimizer', 'Test', 'create', 'register']


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

    learning_rate : float, optional, default None
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.

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

    aggregate_num : int, optional, default None
        Number of weights to be aggregated in a list.
        They are passed to the optimizer for a single optimization step.
        In default, only one weight is aggregated.
        When `aggregate_num` is set to numpy.inf, all the weights are aggregated.

    use_fused_step : bool, optional, default None
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.

    Properties
    ----------
    learning_rate : float
        The current learning rate of the optimizer. Given an Optimizer object
        optimizer, its learning rate can be accessed as optimizer.learning_rate.
    """
    def __init__(self, rescale_grad=1., param_idx2name=None, wd=0.,
                 clip_gradient=None, learning_rate=None,
                 lr_scheduler=None, sym=None, begin_num_update=0,
                 multi_precision=False, param_dict=None, aggregate_num=None,
                 use_fused_step=None, **kwargs):
        super(Optimizer, self).__init__(**kwargs)
        self.rescale_grad = rescale_grad
        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is None and learning_rate is None:
            learning_rate = 0.01
        self.lr = learning_rate
        if self.lr_scheduler is not None and learning_rate is not None:
            if self.lr_scheduler.base_lr != learning_rate:
                print(UserWarning("learning rate from ``lr_scheduler`` has been "
                                  "overwritten by ``learning_rate`` in optimizer."))
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

        if aggregate_num is None:
            self.aggregate_num = 1
        else:
            self.aggregate_num = aggregate_num

        if param_idx2name is None:
            param_idx2name = {}
        assert isinstance(param_idx2name, dict), \
            'param_idx2name should be a dict of param indexes to names.'
        self.idx2name = param_idx2name.copy()
        self.sym_info = (sym.attr_dict(), sym.list_arguments()) if sym is not None else ()
        self.param_dict = param_dict if param_dict else {}
        self.allow_np_array = is_np_array()
        self.use_fused_step = use_fused_step \
            if use_fused_step is not None else False

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
            warnings.warn(f'WARNING: New optimizer {klass.__module__}.{klass.__name__} is overriding '
                          f'existing optimizer {Optimizer.opt_registry[name].__module__}.{Optimizer.opt_registry[name].__name__}')
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
            raise ValueError(f'Cannot find optimizer {name}')

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
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (weight_master_copy,) + (self.create_state(index, weight_master_copy),)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "optimizer")
        return self.create_state(index, weight)

    def step(self, indices, weights, grads, states):
        """Perform an optimization step using gradients and states.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        raise NotImplementedError

    def fused_step(self, indices, weights, grads, states):
        """Perform a fused optimization step using gradients and states.
        New operators that fuses optimizer's update should be put in this function.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        raise NotImplementedError

    def update(self, indices, weights, grads, states):
        """Call step to perform a single optimization update if use_fused_step is False,
         otherwise fused_step is called.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for weight, grad in zip(weights, grads):
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
        if not self.use_fused_step:
            self.step(indices, weights, grads, states)
        else:
            self.fused_step(indices, weights, grads, states)

    def update_multi_precision(self, indices, weights, grads, states):
        """Call step to perform a single optimization update if use_fused_step is False,
         otherwise fused_step is called. Mixed precision version.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        weights_master_copy = []
        original_states = []
        grads32 = []
        for weight, grad, state in zip(weights, grads, states):
            if self.multi_precision and weight.dtype == numpy.float16:
                weights_master_copy.append(state[0])
                original_states.append(state[1])
                grads32.append(grad.astype(numpy.float32))
            else:
                weights_master_copy.append(weight)
                original_states.append(state)
                grads32.append(grad)
        self.update(indices, weights_master_copy, grads32, original_states)
        for weight_master_copy, weight in zip(weights_master_copy, weights):
            if self.multi_precision and weight.dtype == numpy.float16:
                cast(weight_master_copy, dtype=weight.dtype, out=weight)

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
        if self.sym_info:
            attr, arg_names = self.sym_info
            for name in arg_names:
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def _set_current_context(self, device_id):
        """This function has been deprecated. Please refer to ``Optimizer._set_current_context``."""
        warnings.warn('Optimizer._set_current_context has been renamed to'
                      ' Optimizer._set_current_device', DeprecationWarning)
        return self._set_current_device(device_id)

    def _set_current_device(self, device_id):
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

# pylint: disable=W0223
@register
class Test(Optimizer):
    """The Test optimizer"""
    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)

    def create_state(self, index, weight):
        """Creates a state to duplicate weight."""
        return zeros(weight.shape, weight.context)

    def step(self, indices, weights, grads, states):
        """Performs w += rescale_grad * grad."""
        for index, weight, grad in zip(indices, weights, grads):
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)
            grad = self.rescale_grad * grad
            weight[:] -= lr * (grad + wd * weight)


create = Optimizer.create_optimizer  # pylint: disable=invalid-name
