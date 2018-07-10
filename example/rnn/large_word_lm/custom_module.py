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

import logging
import warnings

import mxnet as mx
import numpy as np
from mxnet.module import Module
from mxnet.model import load_checkpoint

class CustomModule(Module):

    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=mx.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None, group2ctxs=None,
                 compression_params=None):

        super(CustomModule, self).__init__(symbol, data_names=data_names, label_names=label_names,
                                           logger=logger, context=context, work_load_list=work_load_list,
                                           fixed_param_names=fixed_param_names, state_names=state_names,
                                           group2ctxs=group2ctxs, compression_params=compression_params)

    def prepare_sparse_params(self, param_rowids):
        '''Prepares the module for processing a data batch by pulling row_sparse
        parameters from kvstore to all devices based on rowids.

        Parameters
        ----------
        param_rowids : dict of str to NDArray of list of NDArrays
        '''
        if not self._kvstore:
            return
        assert(isinstance(param_rowids, dict))
        for param_name, rowids in param_rowids.items():
            if isinstance(rowids, (tuple, list)):
                rowids_1d = []
                for r in rowids:
                    rowids_1d.append(r.reshape((-1,)).astype(np.int64))
                rowid = mx.nd.concat(*rowids_1d, dim=0)
            else:
                rowid = rowids
            param_idx = self._exec_group.param_names.index(param_name)
            param_val = self._exec_group.param_arrays[param_idx]
            self._kvstore.row_sparse_pull(param_name, param_val, row_ids=rowid,
                                          priority=-param_idx)

    @staticmethod
    def load(prefix, epoch, load_optimizer_states=False, symbol=None, **kwargs):
        """Creates a model from previously saved checkpoint.

        Parameters
        ----------
        prefix : str
            path prefix of saved model files. You should have
            "prefix-symbol.json", "prefix-xxxx.params", and
            optionally "prefix-xxxx.states", where xxxx is the
            epoch number.
        epoch : int
            epoch to load.
        load_optimizer_states : bool
            whether to load optimizer states. Checkpoint needs
            to have been made with save_optimizer_states=True.
        data_names : list of str
            Default is `('data')` for a typical model used in image classification.
        label_names : list of str
            Default is `('softmax_label')` for a typical model used in image
            classification.
        logger : Logger
            Default is `logging`.
        context : Context or list of Context
            Default is ``cpu()``.
        work_load_list : list of number
            Default ``None``, indicating uniform workload.
        fixed_param_names: list of str
            Default ``None``, indicating no network parameters are fixed.
        """
        sym, args, auxs = load_checkpoint(prefix, epoch)
        sym = sym if symbol is None else symbol
        mod = CustomModule(symbol=sym, **kwargs)
        mod._arg_params = args
        mod._aux_params = auxs
        mod.params_initialized = True
        if load_optimizer_states:
            mod._preload_opt_states = '%s-%04d.states'%(prefix, epoch)
        return mod

    def save_params(self, fname):
        """Saves model parameters to file.
        Parameters
        ----------
        fname : str
            Path to output param file.
        Examples
        --------
        >>> # An example of saving module parameters.
        >>> mod.save_params('myfile')
        """
        arg_params, aux_params = self.get_params_from_kv(self._arg_params, self._aux_params)
        save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
        save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
        mx.nd.save(fname, save_dict)

    def get_params_from_kv(self, arg_params, aux_params):
        """ Copy data from kvstore to `arg_params` and `aux_params`.
        Parameters
        ----------
        arg_params : list of NDArray
            Target parameter arrays.
        aux_params : list of NDArray
            Target aux arrays.
        Notes
        -----
        - This function will inplace update the NDArrays in arg_params and aux_params.
        """
        assert(self._kvstore is not None)
        for name, block in zip(self._exec_group.param_names, self._exec_group.param_arrays):
            assert(isinstance(block, list))
            if block[0].stype == 'row_sparse':
                row_ids = mx.nd.arange(start=0, stop=block[0].shape[0], dtype='int64')
                self._kvstore.row_sparse_pull(name, arg_params[name], row_ids=row_ids)
            else:
                assert(block[0].stype == 'default')
                self._kvstore.pull(name, out=arg_params[name])
        if len(aux_params) > 0:
            raise NotImplementedError()
        return arg_params, aux_params

    def clip_by_global_norm_per_ctx(self, max_norm=1.0, param_names=None):
        """Clips gradient norm.

        The norm is computed over all gradients together, as if they were
         concatenated into a single vector. Gradients are modified in-place.

        The method is first used in
         `[ICML2013] On the difficulty of training recurrent neural networks`

        Note that the gradients are concatenated per context in this implementation.

        Examples
        --------
        An example of using clip_grad_norm to clip the gradient before updating the parameters::
            >>> #Get the gradient via back-propagation
            >>> net.forward_backward(data_batch=data_batch)
            >>> norm_val = net.clip_by_global_norm(max_norm=2.0, param_names='w0')
            >>> net.update()
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized
        num_ctx = len(self._exec_group.grad_arrays[0])
        grad_array_per_ctx = [[] for i in range(num_ctx)]
        assert(param_names is not None)
        for param_name in param_names:
            param_idx = self._exec_group.param_names.index(param_name)
            grad_val = self._exec_group.grad_arrays[param_idx]
            assert(len(grad_val) == num_ctx)
            for i in range(num_ctx):
                grad_array_per_ctx[i].append(grad_val[i])
        norm_vals = []
        for i in range(num_ctx):
            mx.gluon.utils.clip_global_norm(grad_array_per_ctx[i], max_norm)

    def rescale_grad(self, scale=None, param_name=None):
        """ Rescale the gradient of provided parameters by a certain scale """
        if scale is None or param_name is None:
            return
        param_idx = self._exec_group.param_names.index(param_name)
        grad_vals = self._exec_group.grad_arrays[param_idx]
        for grad in grad_vals:
            grad[:] *= scale
