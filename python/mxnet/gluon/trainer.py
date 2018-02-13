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
from ..model import _create_kvstore
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

    Properties
    ----------
    learning_rate : float
        The current learning rate of the optimizer. Given an Optimizer object
        optimizer, its learning rate can be accessed as optimizer.learning_rate.
    """
    def __init__(self, params, optimizer, optimizer_params=None, kvstore='device',
                 compression_params=None):
        if isinstance(params, (dict, ParameterDict)):
            params = list(params.values())
        if not isinstance(params, (list, tuple)):
            raise ValueError(
                "First argument must be a list or dict of Parameters, " \
                "got %s."%(type(params)))
        self._params = []
        for param in params:
            if not isinstance(param, Parameter):
                raise ValueError(
                    "First argument must be a list or dict of Parameters, " \
                    "got list of %s."%(type(param)))
            self._params.append(param)
        self._compression_params = compression_params
        optimizer_params = optimizer_params if optimizer_params else {}
        self._scale = optimizer_params.get('rescale_grad', 1.0)
        self._contexts = self._check_contexts()
        self._init_optimizer(optimizer, optimizer_params)
        self._kv_initialized = False
        self._kvstore = kvstore

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

    def _init_kvstore(self):
        arg_arrays = {param.name: param.data(self._contexts[0]) for param in self._params}
        kvstore, update_on_kvstore = _create_kvstore(self._kvstore, len(self._contexts),
                                                     arg_arrays)
        if kvstore:
            if self._compression_params:
                kvstore.set_gradient_compression(self._compression_params)
            if 'dist' in kvstore.type:
                update_on_kvstore = False
            for i, param in enumerate(self._params):
                param_arrays = param.list_data()
                kvstore.init(i, param_arrays[0])
                kvstore.pull(i, param_arrays, priority=-i)
            if update_on_kvstore:
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


    def step(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.compute_gradient` and outside of `record()` scope.

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

        self._optimizer.rescale_grad = self._scale / batch_size

        for i, param in enumerate(self._params):
            if param.grad_req == 'null':
                continue
            if not ignore_stale_grad:
                for data in param.list_data():
                    if not data._fresh_grad:
                        raise UserWarning(
                            "Gradient of Parameter `%s` on context %s has not been updated "
                            "by backward since last `step`. This could mean a bug in your "
                            "model that maked it only use a subset of the Parameters (Blocks) "
                            "for this iteration. If you are intentionally only using a subset, "
                            "call step with ignore_stale_grad=True to suppress this "
                            "warning and skip updating of Parameters with stale gradient" \
                            %(param.name, str(data.context)))

            if self._kvstore:
                self._kvstore.push(i, param.list_grad(), priority=-i)
                if self._update_on_kvstore:
                    self._kvstore.pull(i, param.list_data(), priority=-i)
                    continue
                else:
                    self._kvstore.pull(i, param.list_grad(), priority=-i)

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

        if self._update_on_kvstore:
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
