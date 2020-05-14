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
"""A `SVRGModule` implements the `Module` API by wrapping an auxiliary module to perform
SVRG optimization logic.
"""

import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer


class SVRGModule(Module):
    """SVRGModule is a module that encapsulates two Modules to accommodate the SVRG optimization technique.
    It is functionally the same as Module API, except it is implemented using SVRG optimization logic.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
        Defaults to `('data')` for a typical model used in image classification.
    label_names : list of str
        Defaults to `('softmax_label')` for a typical model used in image classification.
    logger : Logger
        Defaults to `logging`.
    context : Context or list of Context
        Defaults to ``mx.cpu()``.
    work_load_list : list of number
        Default ``None``, indicating uniform workload.
    fixed_param_names: list of str
        Default ``None``, indicating no network parameters are fixed.
    state_names : list of str
        states are similar to data and label, but not provided by data iterator. \
        Instead they are initialized to 0 and can be set by `set_states()`.
    group2ctxs : dict of str to context or list of context, or list of dict of str to context
        Default is `None`. Mapping the `ctx_group` attribute to the context assignment.
    compression_params : dict
        Specifies type of gradient compression and additional arguments depending \
        on the type of compression being used. For example, 2bit compression requires a threshold. \
        Arguments would then be {'type':'2bit', 'threshold':0.5} \
        See mxnet.KVStore.set_gradient_compression method for more details on gradient compression. \
    update_freq: int
        Specifies the number of times to update the full gradients to be used in the SVRG optimization. For instance, \
        update_freq = 2 will calculates the gradients over all data every two epochs

    Examples
    --------
    >>> # An example of declaring and using SVRGModule.
    >>> mod = SVRGModule(symbol=lro, data_names=['data'], label_names=['lin_reg_label'], update_freq=2)
    >>> mod.fit(di, eval_metric='mse', optimizer='sgd', optimizer_params=(('learning_rate', 0.025),),
    >>>         num_epoch=num_epoch, kvstore='local')
    """

    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=mx.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None, group2ctxs=None,
                 compression_params=None, update_freq=None):
        super(SVRGModule, self).__init__(symbol, data_names=data_names, label_names=label_names, logger=logger,
                                         context=context, work_load_list=work_load_list,
                                         fixed_param_names=fixed_param_names, state_names=state_names,
                                         group2ctxs=group2ctxs, compression_params=compression_params)

        # Type check update_frequency
        if isinstance(update_freq, int):
            if update_freq <= 0:
                raise ValueError("update_freq in SVRGModule must be a positive integer to represent the frequency for "
                                 "calculating full gradients")
            self.update_freq = update_freq
        else:
            raise TypeError("update_freq in SVRGModule must be an integer to represent the frequency for "
                            "calculating full gradients")

        self._mod_aux = mx.mod.Module(symbol, data_names, label_names, logger, context, work_load_list,
                                      fixed_param_names, state_names, group2ctxs, compression_params)

        self._param_dict = None
        self._ctx_len = len(self._context)

    def _reset_bind(self):
        """Internal function to reset binded state for both modules."""
        super(SVRGModule, self)._reset_bind()
        self._mod_aux._reset_bind()

    def reshape(self, data_shapes, label_shapes=None):
        """Reshapes both modules for new input shapes.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_data``.
        label_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_label``.
        """
        super(SVRGModule, self).reshape(data_shapes, label_shapes=label_shapes)
        self._mod_aux.reshape(data_shapes, label_shapes=label_shapes)

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Installs and initializes SVRGOptimizer. The SVRGOptimizer is a wrapper class for a regular optimizer that is
        passed in and a special AssignmentOptimizer to accumulate the full gradients.  If KVStore is 'local' or None,
        the full gradients will be accumulated locally without pushing to the KVStore. Otherwise, additional keys will
        be pushed to accumulate the full gradients in the KVStore.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default ``False``, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """

        # Init dict for storing average of full gradients for each device
        self._param_dict = [{key: mx.nd.zeros(shape=value.shape, ctx=self._context[i])
                             for key, value in self.get_params()[0].items()} for i in range(self._ctx_len)]

        svrg_optimizer = self._create_optimizer(_SVRGOptimizer.__name__, default_opt=optimizer,
                                                kvstore=kvstore, optimizer_params=optimizer_params)

        super(SVRGModule, self).init_optimizer(kvstore=kvstore, optimizer=svrg_optimizer,
                                               optimizer_params=optimizer_params, force_init=force_init)

        # Init additional keys for accumulating full grads in KVStore
        if self._kvstore:
            for idx, param_on_devs in enumerate(self._exec_group.param_arrays):
                name = self._exec_group.param_names[idx]
                self._kvstore.init(name + "_full", mx.nd.zeros(shape=self._arg_params[name].shape))
                if self._update_on_kvstore:
                    self._kvstore.pull(name + "_full", param_on_devs, priority=-idx)

    def _create_optimizer(self, optimizer, default_opt, kvstore, optimizer_params):
        """Helper function to create a svrg optimizer. SVRG optimizer encapsulates two optimizers and
        will redirect update() to the correct optimizer based on the key.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer: str
            Name for SVRGOptimizer
        default_opt : str or Optimizer that was passed in.
        optimizer_params : dict
           optimizer params that was passed in.
        """

        # code partially copied from mxnet module.init_optimizer() to accomodate svrg_optimizer
        batch_size = self._exec_group.batch_size

        (kv_store, update_on_kvstore) = mx.model._create_kvstore(kvstore, self._ctx_len, self._arg_params)
        if kv_store and 'dist' in kv_store.type and '_sync' in kv_store.type:
            batch_size *= kv_store.num_workers
        rescale_grad = 1.0 / batch_size

        idx2name = {}
        if update_on_kvstore:
            idx2name.update(enumerate(self._exec_group.param_names))
        else:
            for k in range(self._ctx_len):
                idx2name.update({i * self._ctx_len + k: n
                                 for i, n in enumerate(self._exec_group.param_names)})

        # update idx2name to include new keys
        for key in self._param_dict[0].keys():
            max_key = max(list(idx2name.keys())) + 1
            idx2name[max_key] = key + "_full"

        optimizer_params = dict(optimizer_params)
        if 'rescale_grad' not in optimizer_params:
            optimizer_params['rescale_grad'] = rescale_grad
        optimizer_params["default_optimizer"] = default_opt
        optimizer_params["param_idx2name"] = idx2name
        optimizer = mx.optimizer.create(optimizer, **optimizer_params)

        return optimizer

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None, grad_req='write'):
        """Binds the symbols to construct executors for both two modules. This is necessary before one
        can perform computation with the SVRGModule.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_data``.
        label_shapes : list of (str, tuple)
            Typically is ``data_iter.provide_label``.
        for_training : bool
            Default is ``True``. Whether the executors should be bound for training.
        inputs_need_grad : bool
            Default is ``False``. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is ``False``. This function does nothing if the executors are already
            bound. But with this ``True``, the executors will be forced to rebind.
        shared_module : Module
            Default is ``None``. This is used in bucketing. When not ``None``, the shared module
            essentially corresponds to a different bucket -- a module with different symbol
            but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        super(SVRGModule, self).bind(data_shapes, label_shapes, for_training, inputs_need_grad, force_rebind,
                                     shared_module, grad_req)

        if for_training:
            self._mod_aux.bind(data_shapes, label_shapes, for_training, inputs_need_grad, force_rebind, shared_module,
                               grad_req)

    def forward(self, data_batch, is_train=None):
        """Forward computation for both two modules. It supports data batches with different shapes, such as
        different batch sizes or different image sizes.
        If reshaping of data batch relates to modification of symbol or module, such as
        changing image layout ordering or switching from training to predicting, module
        rebinding is required.

        See Also
        ----------
        :meth:`BaseModule.forward`.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is ``None``, which means ``is_train`` takes the value of ``self.for_training``.
        """
        super(SVRGModule, self).forward(data_batch, is_train)

        if is_train:
            self._mod_aux.forward(data_batch, is_train)

    def backward(self, out_grads=None):
        """Backward computation.

        See Also
        ----------
        :meth:`BaseModule.backward`.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        super(SVRGModule, self).backward(out_grads)

        if self._mod_aux.binded:
            self._mod_aux.backward(out_grads)

    def update(self):
        """Updates parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch. The gradients in the _exec_group will be overwritten
        using the gradients calculated by the SVRG update rule.

        When KVStore is used to update parameters for multi-device or multi-machine training,
        a copy of the parameters is stored in KVStore. Note that for `row_sparse` parameters,
        this function does update the copy of parameters in KVStore, but doesn't broadcast the
        updated parameters to all devices / machines. Please call `prepare` to broadcast
        `row_sparse` parameters with the next batch of data.

        See Also
        ----------
        :meth:`BaseModule.update`.
        """
        self._update_svrg_gradients()
        super(SVRGModule, self).update()

    def update_full_grads(self, train_data):
        """Computes the gradients over all data w.r.t weights of past
        m epochs. For distributed env, it will accumulate full grads in the kvstore.

        Parameters
        ----------
        train_data: DataIter
            Train data iterator
        """
        param_names = self._exec_group.param_names
        arg, aux = self.get_params()
        self._mod_aux.set_params(arg_params=arg, aux_params=aux)
        train_data.reset()
        nbatch = 0
        padding = 0
        for batch in train_data:
            self._mod_aux.forward(batch, is_train=True)
            self._mod_aux.backward()
            nbatch += 1
            for ctx in range(self._ctx_len):
                for index, name in enumerate(param_names):
                    grads = self._mod_aux._exec_group.grad_arrays[index][ctx]
                    self._param_dict[ctx][name] = mx.nd.broadcast_add(self._param_dict[ctx][name], grads, axis=0)
            padding = batch.pad

        true_num_batch = nbatch - padding / train_data.batch_size
        for name in param_names:
            grad_list = []
            for i in range(self._ctx_len):
                self._param_dict[i][name] /= true_num_batch
                grad_list.append(self._param_dict[i][name])
            if self._kvstore:
                # If in distributed mode, push a list of gradients from each worker/device to the KVStore
                self._accumulate_kvstore(name, grad_list)

    def _accumulate_kvstore(self, key, value):
        """Accumulate gradients over all data in the KVStore. In distributed setting, each worker sees a portion of
        data. The full gradients will be aggregated from each worker in the KVStore.

        Parameters
        ----------

        key: int or str
            Key in the KVStore.
        value: NDArray, RowSparseNDArray
            Average of the full gradients.
        """
        # Accumulate full gradients for current epochs
        self._kvstore.push(key + "_full", value)
        self._kvstore._barrier()
        self._kvstore.pull(key + "_full", value)

        self._allocate_gradients(key, value)

    def _allocate_gradients(self, key, value):
        """Allocate average of full gradients accumulated in the KVStore to each device.

        Parameters
        ----------

        key: int or str
            Key in the kvstore.
        value: List of NDArray, List of RowSparseNDArray
            A list of average of the full gradients in the KVStore.
        """
        for i in range(self._ctx_len):
            self._param_dict[i][key] = value[i] / self._ctx_len

    def _svrg_grads_update_rule(self, g_curr_batch_curr_weight, g_curr_batch_special_weight,
                                g_special_weight_all_batch):
        """Calculates the gradient based on the SVRG update rule.
        Parameters
        ----------
        g_curr_batch_curr_weight : NDArray
            gradients of current weight of self.mod w.r.t current batch of data
        g_curr_batch_special_weight: NDArray
            gradients of the weight of past m epochs of self._mod_special w.r.t current batch of data
        g_special_weight_all_batch: NDArray
            average of full gradients over full pass of data

        Returns
        ----------
        Gradients calculated using SVRG update rule:
        grads = g_curr_batch_curr_weight - g_curr_batch_special_weight + g_special_weight_all_batch
        """
        for index, grad in enumerate(g_curr_batch_curr_weight):
            grad -= g_curr_batch_special_weight[index]
            grad += g_special_weight_all_batch[index]
        return g_curr_batch_curr_weight

    def _update_svrg_gradients(self):
        """Calculates gradients based on the SVRG update rule.
        """
        param_names = self._exec_group.param_names
        for ctx in range(self._ctx_len):
            for index, name in enumerate(param_names):
                g_curr_batch_reg = self._exec_group.grad_arrays[index][ctx]
                g_curr_batch_special = self._mod_aux._exec_group.grad_arrays[index][ctx]
                g_special_weight_all_batch = self._param_dict[ctx][name]
                g_svrg = self._svrg_grads_update_rule(g_curr_batch_reg, g_curr_batch_special,
                                                      g_special_weight_all_batch)
                self._exec_group.grad_arrays[index][ctx] = g_svrg

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=mx.init.Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, sparse_row_id_fn=None):
        """Trains the module parameters.

        Parameters
        ----------
        train_data : DataIter
            Train DataIter.
        eval_data : DataIter
            If not ``None``, will be used as validation set and the performance
            after each epoch will be evaluated.
        eval_metric : str or EvalMetric
            Defaults to 'accuracy'. The performance measure used to display during training.
            Other possible predefined metrics are:
            'ce' (CrossEntropy), 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'.
        epoch_end_callback : function or list of functions
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Defaults to 'local'.
        optimizer : str or Optimizer
            Defaults to 'sgd'.
        optimizer_params : dict
            Defaults to ``(('learning_rate', 0.01),)``. The parameters for
            the optimizer constructor.
            The default value is not a dict, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each mini-batch during evaluation.
        initializer : Initializer
            The initializer is called to initialize the module parameters when they are
            not already initialized.
        arg_params : dict
            Defaults to ``None``, if not ``None``, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has a higher priority than `initializer`.
        aux_params : dict
            Defaults to ``None``. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Defaults to ``False``. Indicates whether to allow missing parameters when `arg_params`
            and `aux_params` are not ``None``. If this is ``True``, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Defaults to ``False``. Whether to force rebinding the executors if already bound.
        force_init : bool
            Defaults to ``False``. Indicates whether to force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Defaults to 0. Indicates the starting epoch. Usually, if resumed from a
            checkpoint saved at a previous training phase at epoch N, then this value should be
            N+1.
        num_epoch : int
            Number of epochs for training.
        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.
        validation_metric: str or EvalMetric
            The performance measure used to display during validation.
        """
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer, optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            eval_metric.reset()
            tic = time.time()
            if epoch % self.update_freq == 0:
                self.update_full_grads(train_data)

            train_data.reset()
            data_iter = iter(train_data)
            end_of_batch = False
            nbatch = 0
            next_data_batch = next(data_iter)

            while not end_of_batch:
                data_batch = next_data_batch
                if monitor is not None:
                    monitor.tic()

                self.forward_backward(data_batch)
                self.update()

                if isinstance(data_batch, list):
                    self.update_metric(eval_metric, [db.label for db in data_batch], pre_sliced=True)
                else:
                    self.update_metric(eval_metric, data_batch.label)

                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
                except StopIteration:
                    end_of_batch = True

                if monitor is not None:
                    monitor.toc_print()

                if end_of_batch:
                    eval_name_vals = eval_metric.get_name_value()

                if batch_end_callback is not None:
                    batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                              eval_metric=eval_metric, locals=locals())
                    for callback in mx.base._as_list(batch_end_callback):
                        callback(batch_end_params)

                nbatch += 1
            for name, val in eval_name_vals:
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in mx.base._as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            # ----------------------------------------
            # evaluation on validation set
            if eval_data:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

    def prepare(self, data_batch, sparse_row_id_fn=None):
        """Prepares two modules for processing a data batch.

        Usually involves switching bucket and reshaping.
        For modules that contain `row_sparse` parameters in KVStore,
        it prepares the `row_sparse` parameters based on the sparse_row_id_fn.

        When KVStore is used to update parameters for multi-device or multi-machine training,
        a copy of the parameters are stored in KVStore. Note that for `row_sparse` parameters,
        the `update()` updates the copy of parameters in KVStore, but doesn't broadcast
        the updated parameters to all devices / machines. The `prepare` function is used to
        broadcast `row_sparse` parameters with the next batch of data.

        Parameters
        ----------
        data_batch : DataBatch
            The current batch of data for forward computation.

        sparse_row_id_fn : A callback function
            The function  takes `data_batch` as an input and returns a dict of
            str -> NDArray. The resulting dict is used for pulling row_sparse
            parameters from the kvstore, where the str key is the name of the param,
            and the value is the row id of the param to pull.
        """
        super(SVRGModule, self).prepare(data_batch, sparse_row_id_fn=sparse_row_id_fn)
        self._mod_aux.prepare(data_batch, sparse_row_id_fn=sparse_row_id_fn)
