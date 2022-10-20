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
# pylint: disable=line-too-long
"""Parameter optimizer."""
__all__ = ['Trainer']

import warnings
from collections import OrderedDict

from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import Parameter
from ..kvstore import KVStore


class Trainer(object):
    """Applies an `Optimizer` on a set of Parameters. Trainer should
    be used together with `autograd`.

    .. note::

        For the following cases, updates will always happen on kvstore,
        i.e., you cannot set update_on_kvstore=False.

        - dist kvstore with sparse weights or sparse gradients
        - dist async kvstore
        - `optimizer.lr_scheduler` is not None

    Parameters
    ----------
    params : Dict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <https://mxnet.apache.org/api/python/docs/api/optimizer/index.html>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    kvstore : str or KVStore
        kvstore type for multi-gpu and distributed training. See help on
        :func:`mxnet.kvstore.create` for more information.
    compression_params : dict
        Specifies type of gradient compression and additional arguments depending
        on the type of compression being used. For example, 2bit compression requires a threshold.
        Arguments would then be {'type':'2bit', 'threshold':0.5}
        See mxnet.KVStore.set_gradient_compression method for more details on gradient compression.
    update_on_kvstore : bool, default None
        Whether to perform parameter updates on kvstore. If None and optimizer.aggregate_num <= 1,
        then trainer will choose the more suitable option depending on the type of kvstore.
        If None and optimizer.aggregate_num > 1, `update_on_kvstore` is set to False.
        If the `update_on_kvstore` argument is provided,
        environment variable `MXNET_UPDATE_ON_KVSTORE` will be ignored.

    Properties
    ----------
    learning_rate : float
        The current learning rate of the optimizer. Given an Optimizer object
        optimizer, its learning rate can be accessed as optimizer.learning_rate.
    """
    def __init__(self, params, optimizer, optimizer_params=None, kvstore='device',
                 compression_params=None, update_on_kvstore=None):
        param_list = []
        if isinstance(params, (dict, OrderedDict)):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])
            params = param_list
        if not isinstance(params, (list, tuple)):
            raise ValueError(
                "First argument must be a list or dict of Parameters, " \
                f"got {type(params)}.")
        self._params = []
        # parameters to initialize on the kvstore
        self._contains_sparse_weight = False
        self._contains_sparse_grad = False
        self._param2idx = {}
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise ValueError(
                    "First argument must be a list or dict of Parameters, " \
                    f"got list of {type(param)}.")
            if param._uuid in self._param2idx:
                # Shared parameters have same uuid; only need to store one of the shared versions
                continue
            self._param2idx[param._uuid] = i
            self._params.append(param)
            param._set_trainer(self)
            if param._stype != 'default':
                self._contains_sparse_weight = True
            if param._grad_stype != 'default':
                self._contains_sparse_grad = True
        self._compression_params = compression_params
        self._devices = self._check_devices()
        optimizer_params = optimizer_params if optimizer_params else {}
        self._init_optimizer(optimizer, optimizer_params)
        self._scale = self._optimizer.rescale_grad
        if self._optimizer.aggregate_num > 1 and update_on_kvstore is not None:
            if update_on_kvstore:
                raise ValueError("Cannot set update_on_kvstore=True "
                                 "when optimizer.aggregate_num > 1.")
        if update_on_kvstore is None and self._optimizer.aggregate_num > 1:
            update_on_kvstore = False
        self._kvstore_params = {'kvstore': kvstore, 'update_on_kvstore': update_on_kvstore}
        self._kv_initialized = False
        self._kvstore = None
        self._update_on_kvstore = None
        self._distributed = None
        self._params_to_init = []
        self._reset_kvstore()

    def _check_contexts(self):
        """This function has been deprecated. Please refer to ``Trainer._check_devices``."""
        warnings.warn('Trainer._check_contexts has been renamed to'
                      ' Trainer._check_devices', DeprecationWarning)
        return self._check_devices()

    def _check_devices(self):
        devices = None
        for param in self._params:
            device = param.list_device()
            assert devices is None or devices == device, \
                "All Parameters must be initialized on the same set of devices, " \
                f"but Parameter {param.name} is initialized on {str(device)} while previous Parameters " \
                f"are initialized on {str(devices)}."
            devices = device
        return devices

    def _init_optimizer(self, optimizer, optimizer_params):
        param_dict = {i: param for i, param in enumerate(self._params)}
        if isinstance(optimizer, opt.Optimizer):
            assert not optimizer_params, \
                "optimizer_params must be None if optimizer is an instance of " \
                "Optimizer instead of str"
            self._optimizer = optimizer
            # param_dict must not be deep copied, so that if user mutate the lr_mult
            # or wd_mult of some parameters, it takes effect.
            self._optimizer.param_dict = param_dict
        else:
            self._optimizer = opt.create(optimizer, param_dict=param_dict,
                                         **optimizer_params)
        self._updaters = [opt.get_updater(self._optimizer) \
                            for _ in self._devices]

    def _init_params(self):
        """Initialize parameters in the KVStore.

        Parameters with incomplete initialization are ignored.

        """
        assert self._kv_initialized, "Cannot initialize parameters in KVStore " \
                                     "when KVStore is not initialized."
        params_to_init = []
        if self._kvstore:
            for param in self._params_to_init:
                if param._deferred_init:
                    params_to_init.append(param)
                else:
                    param_arrays = param._check_and_get(param._data, list)
                    idx = self._param2idx[param._uuid]
                    if param._stype != 'default':
                        self._kvstore.init(idx, param_arrays[0])
                    else:
                        self._kvstore.broadcast(idx, param_arrays[0], param_arrays)

        self._params_to_init = params_to_init

    def _reset_kvstore(self):
        """Reset kvstore."""
        if self._kvstore and 'dist' in self._kvstore.type:
            raise RuntimeError("Cannot reset distributed KVStore.")
        self._kv_initialized = False
        self._kvstore = None
        self._distributed = None
        self._update_on_kvstore = None
        self._params_to_init = [param for param in self._params]

    def _init_kvstore(self):
        """Create kvstore."""
        config = self._kvstore_params
        # configure kvstore, update_on_kvstore and self._distributed on three cases:
        if self._contains_sparse_weight:
            # If weight is sparse, kvstore must be present and the weight must be updated on kvstore.
            # The training loop is the following:
            #    - row_sparse_pull(sparse_weight)
            #    - forward()
            #    - backward()
            #    - push_and_update(grad)
            #    - pull(weight)
            kvstore, update_on_kvstore = _create_sparse_kvstore(config['kvstore'])
            self._distributed = 'dist' in kvstore.type
            # raise err if user provides unsupported configs
            if config['update_on_kvstore'] is False:
                raise ValueError("Cannot set update_on_kvstore=False when sparse weights "
                                 "are present.")

        elif self._contains_sparse_grad:
            # For single node training with dense weight and sparse grad,
            # we prefer update_on_kvstore=False because this is usually faster.
            # This means we push and pull sparse gradients, and we do not store weight in kvstore.
            # The training loop is the following:
            #    - forward()
            #    - backward()
            #    - push(grad)
            #    - pull(grad)
            #    - update(grad, weight)
            #
            # For multi-node training with dense weight and sparse grad,
            # only update_on_kvstore=True is supported, due to the fact that
            # kv.row_sparse_pull(grad) is not implemented.
            # Therefore, we push sparse gradients and pull dense weights.
            # The training loop contains:
            #    - forward()
            #    - backward()
            #    - push_and_update(grad)
            #    - pull(weight)
            arg_arrays = {param._uuid: param.data(self._devices[0]) for param in self._params}
            kvstore, _ = _create_kvstore(config['kvstore'], len(self._devices), arg_arrays)
            self._distributed = 'dist' in kvstore.type if kvstore else False
            update_on_kvstore = self._distributed
            # raise err if user provides unsupported configs
            if config['update_on_kvstore'] is not None:
                if config['update_on_kvstore'] is False and self._distributed:
                    raise ValueError("Cannot set update_on_kvstore=False on dist kvstore "
                                     "when sparse gradients are present.")
                update_on_kvstore = config['update_on_kvstore']
            # raise err if a custom kvstore is used for sparse training
            if kvstore is not None and not isinstance(kvstore, KVStore):
                raise ValueError("Cannot use {} for multi-device training with sparse gradients"
                                 .format(type(kvstore)))

        else:
            # Training with dense weight and dense gradients.
            # The only unsupported mode is async with update_on_kvstore=False
            arg_arrays = {param._uuid: param.data(self._devices[0]) for param in self._params}
            kvstore, update_on_kvstore = _create_kvstore(config['kvstore'], len(self._devices),
                                                         arg_arrays)
            self._distributed = 'dist' in kvstore.type if kvstore else False
            if self._distributed and 'async' in kvstore.type:
                update_on_kvstore = True
                # raise err if user provides unsupported configs
                if config['update_on_kvstore'] is False:
                    raise ValueError("Please set update_on_kvstore=True "
                                     "when training in async mode.")
            if config['update_on_kvstore'] is not None:
                update_on_kvstore = config['update_on_kvstore']
            # raise err if update_on_kvstore is set to True with kvstores that do not support optimizers
            if update_on_kvstore and not kvstore.is_capable('optimizer'):
                if config['update_on_kvstore']:
                    raise ValueError("Please set update_on_kvstore=False "
                                     "when training with {}".format(type(kvstore)))
                update_on_kvstore = False

        # set grad compression and optimizers
        if kvstore:
            if self._compression_params:
                kvstore.set_gradient_compression(self._compression_params)
            if update_on_kvstore:
                # optimizer preferably needs to be set before init for multiprecision
                kvstore.set_optimizer(self._optimizer)
            self._kvstore = kvstore
            self._update_on_kvstore = update_on_kvstore
        else:
            self._kvstore = None
            self._update_on_kvstore = None

        self._kv_initialized = True

    @property
    def learning_rate(self):
        if not isinstance(self._optimizer, opt.Optimizer):
            raise UserWarning("Optimizer has to be defined before its learning "
                              "rate can be accessed.")

        return self._optimizer.learning_rate

    @property
    def optimizer(self):
        if isinstance(self._optimizer, opt.Optimizer):
            return self._optimizer
        else:
            raise UserWarning("Optimizer has not been initialized yet")

    def set_learning_rate(self, lr):
        """Sets a new learning rate of the optimizer.

        Parameters
        ----------
        lr : float
            The new learning rate of the optimizer.
        """
        if not isinstance(self._optimizer, opt.Optimizer):
            raise UserWarning("Optimizer has to be defined before its learning "
                              "rate is mutated.")

        self._optimizer.set_learning_rate(lr)

    def _row_sparse_pull(self, parameter, out, row_id, full_idx=False):
        """Internal method to invoke pull operations on KVStore. If `full_idx` is set to True,
        `kv.pull` is preferred instead of `kv.row_sparse_pull`.
        """
        # initialize kv and params if not already
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        idx = self._param2idx[parameter._uuid]
        if full_idx and 'dist' not in self._kvstore.type:
            assert row_id.size == out.shape[0]
            self._kvstore.pull(idx, out=out, priority=-idx, ignore_sparse=False)
        else:
            self._kvstore.row_sparse_pull(idx, out=out, row_ids=row_id, priority=-idx)

    def _check_and_rescale_grad(self, scale):
        if self._update_on_kvstore and self._distributed and self._kv_initialized:
            if self._optimizer.rescale_grad != scale:
                raise UserWarning('Possible change in the `batch_size` from previous '
                                  '`step` detected. Optimizer gradient normalizing '
                                  'factor will not change w.r.t new batch_size when '
                                  'update_on_kvstore=True and when distributed kvstore '
                                  'is used.')
        self._optimizer.rescale_grad = scale

    def step(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        self._allreduce_grads()
        self._update(ignore_stale_grad)

    def allreduce_grads(self):
        """For each parameter, reduce the gradients from different devices.

        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        """
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        assert not (self._kvstore and self._update_on_kvstore), \
                'allreduce_grads() when parameters are updated on kvstore ' \
                'is not supported. Try setting `update_on_kvstore` ' \
                'to False when creating trainer.'

        self._allreduce_grads()

    def _allreduce_grads(self):
        # nothing to reduce
        if not self._kvstore:
            return
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                idx = self._param2idx[param._uuid]
                grad_list = param.list_grad()
                # sparse gradients, call push and pull separately
                if grad_list[0].stype != 'default':
                    self._kvstore.push(idx, grad_list, priority=-i)
                    if param._stype == 'default':
                        if self._update_on_kvstore:
                            pull_list = param.list_data()
                        else:
                            pull_list = param.list_grad()
                        self._kvstore.pull(idx, pull_list, priority=-i,
                                           ignore_sparse=self._distributed)
                else:
                    # allreduce dense gradients if not update_on_kvstore,
                    # otherwise push dense gradients, pull dense weights
                    if self._update_on_kvstore:
                        self._kvstore.pushpull(idx, grad_list, out=param.list_data(), priority=-i)
                    else:
                        self._kvstore.pushpull(idx, grad_list, priority=-i)

    def update(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update.

        Should be called after `autograd.backward()` and outside of `record()` scope,
        and after `trainer.update()`.


        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        assert not (self._kvstore and self._update_on_kvstore), \
                'update() when parameters are updated on kvstore ' \
                'is not supported. Try setting `update_on_kvstore` ' \
                'to False when creating trainer.'

        self._check_and_rescale_grad(self._scale / batch_size)
        self._update(ignore_stale_grad)

    def _update(self, ignore_stale_grad=False):
        loss_scaler = getattr(self, '_amp_loss_scaler', None)
        if loss_scaler is not None:
            if loss_scaler.has_overflow(self._params):
                return  # skip on overflow

        updates = [[] for _ in self._updaters]

        for i, param in enumerate(self._params):
            if param.grad_req == 'null':
                continue

            if not ignore_stale_grad:
                for data in param._check_and_get(param._data, list):
                    if not data._fresh_grad:
                        raise UserWarning(
                            f"Gradient of Parameter `{param.name}` on device {str(data.device)} has not been updated "
                            "by backward since last `step`. This could mean a bug in your "
                            "model that made it only use a subset of the Parameters (Blocks) "
                            "for this iteration. If you are intentionally only using a subset, "
                            "call step with ignore_stale_grad=True to suppress this "
                            "warning and skip updating of Parameters with stale gradient")

            if self._kvstore and self._update_on_kvstore:
                continue

            for upd, arr, grad in zip(updates, param.list_data(), param.list_grad()):
                if not ignore_stale_grad or arr._fresh_grad:
                    upd.append((i, grad, arr))
                    arr._fresh_grad = False

        if not (self._kvstore and self._update_on_kvstore):
            for updater, upd in zip(self._updaters, updates):
                if upd:
                    i, g, w = zip(*upd)
                    updater(i, g, w)

    def save_states(self, fname):
        """Saves trainer states (e.g. optimizer, momentum) to a file.


        Parameters
        ----------
        fname : str
            Path to output states file.

        Note
        ----
        `optimizer.param_dict`, which contains Parameter information (such as
        `lr_mult` and `wd_mult`) will not be saved.
        """
        assert self._optimizer is not None

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        if self._update_on_kvstore:
            assert not self._params_to_init, "Cannot save trainer states when some " \
                                             "parameters are not yet initialized in kvstore."
            self._kvstore.save_optimizer_states(fname, dump_optimizer=True)
        else:
            with open(fname, 'wb') as fout:
                fout.write(self._updaters[0].get_states(dump_optimizer=True))

    def load_states(self, fname):
        """Loads trainer states (e.g. optimizer, momentum) from a file.

        Parameters
        ----------
        fname : str
            Path to input states file.

        Note
        ----
        `optimizer.param_dict`, which contains Parameter information (such as
        `lr_mult` and `wd_mult`) will not be loaded from the file, but rather set
        based on current Trainer's parameters.
        """
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        if self._update_on_kvstore:
            self._kvstore.load_optimizer_states(fname)
            self._optimizer = self._kvstore._updater.optimizer
        else:
            with open(fname, 'rb') as f:
                states = f.read()
            for updater in self._updaters:
                updater.set_states(states)
                updater.optimizer = self._updaters[0].optimizer
            self._optimizer = self._updaters[0].optimizer
        param_dict = {i: param for i, param in enumerate(self._params)}
        self._optimizer.param_dict = param_dict
