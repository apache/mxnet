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

from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter

class Trainer(object):
    """Applies an `Optimizer` on a set of Parameters. Trainer should
    be used together with `autograd`.

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    kvstore : str or KVStore
        kvstore type for multi-gpu and distributed training. See help on
        :any:`mxnet.kvstore.create` for more information.
    compression_params : dict
        Specifies type of gradient compression and additional arguments depending
        on the type of compression being used. For example, 2bit compression requires a threshold.
        Arguments would then be {'type':'2bit', 'threshold':0.5}
        See mxnet.KVStore.set_gradient_compression method for more details on gradient compression.
    update_on_kvstore : bool, default None
        Whether to perform parameter updates on kvstore. If None, then trainer will choose the more
        suitable option depending on the type of kvstore.

    Properties
    ----------
    learning_rate : float
        The current learning rate of the optimizer. Given an Optimizer object
        optimizer, its learning rate can be accessed as optimizer.learning_rate.
    """
    def __init__(self, params, optimizer, optimizer_params=None, kvstore='device',
                 compression_params=None, update_on_kvstore=None):
        if isinstance(params, (dict, ParameterDict)):
            params = list(params.values())
        if not isinstance(params, (list, tuple)):
            raise ValueError(
                "First argument must be a list or dict of Parameters, " \
                "got %s."%(type(params)))
        self._params = []
        # parameters to initialize on the kvstore
        self._contains_sparse_weight = False
        self._contains_sparse_grad = False
        self._param2idx = {}
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise ValueError(
                    "First argument must be a list or dict of Parameters, " \
                    "got list of %s."%(type(param)))
            self._param2idx[param.name] = i
            self._params.append(param)
            param._set_trainer(self)
            if param._stype != 'default':
                self._contains_sparse_weight = True
            if param._grad_stype != 'default':
                self._contains_sparse_grad = True
        self._compression_params = compression_params
        optimizer_params = optimizer_params if optimizer_params else {}
        self._scale = float(optimizer_params.get('rescale_grad', 1.0))
        self._contexts = self._check_contexts()
        self._init_optimizer(optimizer, optimizer_params)
        self._kvstore_params = {'kvstore': kvstore, 'update_on_kvstore': update_on_kvstore}
        self._kv_initialized = False
        self._kvstore = None
        self._update_on_kvstore = None
        self._distributed = None
        self._params_to_init = []
        self._reset_kvstore()

    def _check_contexts(self):
        contexts = None
        for param in self._params:
            ctx = param.list_ctx()
            assert contexts is None or contexts == ctx, \
                "All Parameters must be initialized on the same set of contexts, " \
                "but Parameter %s is initialized on %s while previous Parameters " \
                "are initialized on %s."%(param.name, str(ctx), str(contexts))
            contexts = ctx
        return contexts

    def _init_optimizer(self, optimizer, optimizer_params):
        param_dict = {i: param for i, param in enumerate(self._params)}
        if isinstance(optimizer, opt.Optimizer):
            assert not optimizer_params, \
                "optimizer_params must be None if optimizer is an instance of " \
                "Optimizer instead of str"
            self._optimizer = optimizer
            self._optimizer.param_dict = param_dict
        else:
            self._optimizer = opt.create(optimizer, param_dict=param_dict,
                                         **optimizer_params)

        self._updaters = [opt.get_updater(self._optimizer) \
                            for _ in self._contexts]

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
                    idx = self._param2idx[param.name]
                    self._kvstore.init(idx, param_arrays[0])
                    if param._stype == 'default':
                        self._kvstore.pull(idx, param_arrays, priority=-idx)

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
        # if weight is sparse, the weight must be updated on KVStore.
        # training loop contains:
        #    - row_sparse_pull(sparse_weight)
        #    - forward()
        #    - backward()
        #    - push(sparse_grad), push(dense_grad)
        #    - pull(dense_weight)
        if self._contains_sparse_weight:
            kvstore, update_on_kvstore = _create_sparse_kvstore(config['kvstore'])
            # raise Error if update_on_kvstore is set to False by the user
            if config['update_on_kvstore'] is False:
                raise RuntimeError("Cannot set update_on_kvstore to False when sparse weights "
                                   "are present.")
        # if weight is dense and grad is sparse, the weight better not be updated on KVStore.
        # training loop contains:
        #    - forward()
        #    - backward()
        #    - push(grad)
        #    - pull(grad)
        #    - update(grad, weight)
        elif self._contains_sparse_grad:
            arg_arrays = {param.name: param.data(self._contexts[0]) for param in self._params}
            kvstore, _ = _create_kvstore(config['kvstore'], len(self._contexts), arg_arrays)
            update_on_kvstore = False
        # normal case
        else:
            arg_arrays = {param.name: param.data(self._contexts[0]) for param in self._params}
            kvstore, update_on_kvstore = _create_kvstore(config['kvstore'], len(self._contexts),
                                                         arg_arrays)
            if kvstore and 'async' in kvstore.type and config['update_on_kvstore'] is not None\
                    and not config['update_on_kvstore']:
                raise ValueError("Please set update_on_kvstore to true "
                                 "when training in async mode.")

            if config['update_on_kvstore'] is not None:
                update_on_kvstore = config['update_on_kvstore']

        if kvstore:
            if self._compression_params:
                kvstore.set_gradient_compression(self._compression_params)
            self._distributed = 'dist' in kvstore.type
            if self._distributed:
                # kv.pull(row_sparse_grad) is not supported for dist kvstore
                # Captures condition for dist_async, dist_device_sync or based on config for
                # update_on_kvstore
                update_on_kvstore = self._contains_sparse_weight or self._contains_sparse_grad \
                                    or 'device' in kvstore.type or 'async' in kvstore.type \
                                    or config['update_on_kvstore']
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
        else:
            return self._optimizer.learning_rate

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
        else:
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
        idx = self._param2idx[parameter.name]
        if full_idx and 'dist' not in self._kvstore.type:
            assert row_id.size == out.shape[0]
            self._kvstore.pull(idx, out=out, priority=-idx, ignore_sparse=False)
        else:
            self._kvstore.row_sparse_pull(idx, out=out, row_ids=row_id, priority=-idx)

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
        if self._update_on_kvstore and self._distributed and \
           self._optimizer.rescale_grad != rescale_grad:
            raise UserWarning('Possible change in the `batch_size` from previous `step` detected.' \
                            'Optimizer gradient normalizing factor will not change w.r.t new batch_size when ' \
                            'update_on_kvstore=True and when distributed `kvstore` is used.')

        self._optimizer.rescale_grad = rescale_grad

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        self._allreduce_grads()
        self._update(ignore_stale_grad)

    def allreduce_grads(self):
        """For each parameter, reduce the gradients from different contexts.

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
        if self._kvstore:
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':

                    self._kvstore.push(i, param.list_grad(), priority=-i)
                    if not self._update_on_kvstore:
                        self._kvstore.pull(i, param.list_grad(), priority=-i,
                                           ignore_sparse=self._distributed)

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

        self._optimizer.rescale_grad = self._scale / batch_size
        self._update(ignore_stale_grad)

    def _update(self, ignore_stale_grad=False):
        for i, param in enumerate(self._params):
            if param.grad_req == 'null':
                continue

            if not ignore_stale_grad:
                for data in param._check_and_get(param._data, list):
                    if not data._fresh_grad:
                        raise UserWarning(
                            "Gradient of Parameter `%s` on context %s has not been updated "
                            "by backward since last `step`. This could mean a bug in your "
                            "model that made it only use a subset of the Parameters (Blocks) "
                            "for this iteration. If you are intentionally only using a subset, "
                            "call step with ignore_stale_grad=True to suppress this "
                            "warning and skip updating of Parameters with stale gradient" \
                            %(param.name, str(data.context)))

            if self._kvstore and self._update_on_kvstore:
                if param._stype == 'default':
                    # 'row_sparse' parameters are not pulled immediately - they're pulled
                    # in `Block.forward`
                    self._kvstore.pull(i, param.list_data(), priority=-i)
                continue

            for upd, arr, grad in zip(self._updaters, param.list_data(), param.list_grad()):
                if not ignore_stale_grad or arr._fresh_grad:
                    upd(i, grad, arr)
                    arr._fresh_grad = False

    def save_states(self, fname):
        """Saves trainer states (e.g. optimizer, momentum) to a file.

        Parameters
        ----------
        fname : str
            Path to output states file.
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
        """
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        if self._update_on_kvstore:
            self._kvstore.load_optimizer_states(fname)
            self._optimizer = self._kvstore._updater.optimizer
            param_dict = {i: param for i, param in enumerate(self._params)}
            self._optimizer.param_dict = param_dict
        else:
            with open(fname, 'rb') as f:
                states = f.read()
            for updater in self._updaters:
                updater.set_states(states)
                updater.optimizer = self._updaters[0].optimizer
            self._optimizer = self._updaters[0].optimizer
