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

from __future__ import absolute_import, division, print_function

import mxnet as mx
import mxnet.ndarray as nd
import numpy
import os
import pickle
from collections import OrderedDict
from utils import (get_bucket_key, save_params,
                   save_misc, load_params)
import logging

logger = logging.getLogger(__name__)


class Base(object):
    """Basic wrapper for the symbols

    Parameters
    ----------
    data_shapes : dict
        The shapes of tensor variables
    sym_gen : mx.sym.Symbol
        Symbol of the network
    params : None or dict, optional
    params_grad : None or dict, optional
    aux_states:
    initializer:
    ctx:
    name:

    """

    def __init__(self, data_shapes, sym_gen, params=None, aux_states=None,
                 default_bucket_kwargs=None, learn_init_keys=None,
                 initializer=mx.init.Xavier(factor_type="in", rnd_type="gaussian", magnitude=2),
                 ctx=mx.gpu(), name='Net'):
        self.sym_gen = sym_gen
        bucket_kwargs = default_bucket_kwargs.copy() if \
            default_bucket_kwargs is not None else dict()
        self.curr_bucket_key = None
        self.ctx = ctx
        self.name = name
        self.initializer = initializer
        if params is None:
            self.params = None
            self.params_grad = None
        else:
            self.params = OrderedDict([(k, v.copyto(ctx)) for k, v in params.items()])
            self.params_grad = OrderedDict([(n, nd.empty(v.shape, ctx=ctx))
                                            for n, v in self.params.items()])
        if aux_states is not None:
            self.aux_states = OrderedDict([(k, v.copyto(ctx)) for k, v in aux_states.items()])
        else:
            self.aux_states = None
        self._buckets = dict()
        self.learn_init_keys = learn_init_keys if learn_init_keys is not None else []
        self.learn_init_key_shapes = {k: data_shapes[k] for k in self.learn_init_keys}
        self.switch_bucket(bucket_kwargs=bucket_kwargs, data_shapes=data_shapes)
        self.acc_grad = None

    @property
    def exe(self):
        """Get the current executor

        Returns
        -------
        exe : mxnet.executor.Executor
        """
        return self._buckets[self.curr_bucket_key]['exe'][tuple(self.data_shapes.items())]

    @property
    def data_shapes(self):
        return self._buckets[self.curr_bucket_key]['data_shapes']

    @property
    def sym(self):
        return self._buckets[self.curr_bucket_key]['sym']

    def switch_bucket(self, bucket_kwargs=None, data_shapes=None):
        if bucket_kwargs is not None:
            self.curr_bucket_key = get_bucket_key(bucket_kwargs=bucket_kwargs)
        # 1. Check if bucket key exists
        if self.curr_bucket_key in self._buckets:
            if data_shapes is not None:
                if tuple(data_shapes.items()) not in self._buckets[self.curr_bucket_key]['exe']:
                    #TODO Optimize the reshaping functionality!
                    self._buckets[self.curr_bucket_key]['exe'][tuple(data_shapes.items())] = \
                        self.exe.reshape(partial_shaping=True, allow_up_sizing=True, **data_shapes)
                    self._buckets[self.curr_bucket_key]['data_shapes'] = data_shapes
                else:
                    self._buckets[self.curr_bucket_key]['data_shapes'] = data_shapes
            return
        # 2. If the bucket key does not exist, create new symbol + executor
        assert data_shapes is not None, "Must set data_shapes for new bucket!"
        if isinstance(self.sym_gen, mx.symbol.Symbol):
            sym = self.sym_gen
        else:
            sym = self.sym_gen(**dict(self.curr_bucket_key))
        arg_names = sym.list_arguments()
        aux_names = sym.list_auxiliary_states()
        param_names = [n for n in arg_names
                       if n in self.learn_init_keys or (n not in data_shapes.keys())]
        for k, v in data_shapes.items():
            assert isinstance(v, tuple), "Data_shapes must be tuple! Find k=%s, v=%s, " \
                                         "data_shapes=%s" % (k, str(v), str(data_shapes))
        arg_shapes, _, aux_shapes = sym.infer_shape(**data_shapes)
        arg_name_shape = OrderedDict([(k, s) for k, s in zip(arg_names, arg_shapes)])
        if self.params is None:
            self.params = OrderedDict([(n, nd.empty(arg_name_shape[n], ctx=self.ctx))
                                       for n in param_names])
            self.params_grad = OrderedDict([(n, nd.empty(arg_name_shape[n], ctx=self.ctx))
                                            for n in param_names])
            if len(self.params) > 0:
                assert self.initializer is not None, \
                    'We must set the initializer if we donnot initialize' \
                    'manually the free parameters of the network!!'
            for k, v in self.params.items():
                self.initializer(k, v)
        else:
            assert set(arg_name_shape.items()) == \
                   set(list(data_shapes.items()) + list([(k, v.shape) for k, v in self.params.items()]))
        if self.aux_states is None:
            self.aux_states = OrderedDict([(k, nd.empty(s, ctx=self.ctx))
                                           for k, s in zip(aux_names, aux_shapes)])
        data_inputs = {k: mx.nd.empty(data_shapes[k], ctx=self.ctx)
                       for k in set(data_shapes.keys()) - set(self.learn_init_keys)}
        if len(self._buckets) > 0:
            shared_exe = list(list(self._buckets.values())[0]['exe'].values())[0]
        else:
            shared_exe = None
        self._buckets[self.curr_bucket_key] = {
            'exe': {tuple(data_shapes.items()):
                    sym.bind(ctx=self.ctx,
                             args=dict(self.params, **data_inputs),
                             args_grad=dict(self.params_grad.items()),
                             aux_states=self.aux_states,
                             shared_exec=shared_exe)
                    },
            'data_shapes': data_shapes,
            'sym': sym
        }

    def save_params(self, dir_path="", epoch=None):
        param_saving_path = save_params(dir_path=dir_path, name=self.name, epoch=epoch,
                                        params=self.params,
                                        aux_states=self.aux_states)
        misc_saving_path = save_misc(dir_path=dir_path, epoch=epoch, name=self.name,
                                     content={'data_shapes': {k: list(map(int, v)) for k, v in self.data_shapes.items()}})
        logging.info('Saving %s, params: \"%s\", misc: \"%s\"',
                     self.name, param_saving_path, misc_saving_path)

    def load_params(self, name="", dir_path="", epoch=None):
        params, aux_states, param_loading_path = load_params(dir_path=dir_path, epoch=epoch, name=name)
        logging.info('Loading params from \"%s\" to %s' % (param_loading_path, self.name))
        for k, v in params.items():
            if k in self.params:
                logging.debug('   Loading %s %s' %(k, str(v.shape)))
                self.params[k][:] = v
            else:
                logging.warn("Found unused param in the saved model file: %s" % k)
        for k, v in aux_states.items():
            self.aux_states[k][:] = v

    @property
    def internal_sym_names(self):
        return self.sym.get_internals().list_outputs()

    @property
    def output_keys(self):
        return self.sym.list_outputs()

    def compute_internal(self, sym_name, bucket_kwargs=None, **arg_dict):
        """
        View the internal symbols using the forward function.

        :param sym_name:
        :param bucket_kwargs:
        :param input_dict:
        :return:
        """
        data_shapes = {k: v.shape for k, v in arg_dict.items()}
        self.switch_bucket(bucket_kwargs=bucket_kwargs,
                           data_shapes=data_shapes)
        internal_sym = self.sym.get_internals()[sym_name]
        data_inputs = {k: mx.nd.empty(v, ctx=self.ctx)
                       for k, v in self.data_shapes.items()
                       if k in internal_sym.list_arguments()}
        params = {k: v for k, v in self.params.items() if
                  k in internal_sym.list_arguments()}
        aux_states = {k: v for k, v in self.aux_states.items()
                      if k in internal_sym.list_auxiliary_states()}
        exe = internal_sym.bind(ctx=self.ctx,
                                args=dict(params, **data_inputs),
                                args_grad=None,
                                grad_req='null',
                                aux_states=aux_states,
                                shared_exec=self.exe)
        for k, v in arg_dict.items():
            exe.arg_dict[k][:] = v
        exe.forward(is_train=False)
        assert 1 == len(exe.outputs)
        for output in exe.outputs:
            output.wait_to_read()
        return exe.outputs[0]

    def forward(self, is_train=False, bucket_kwargs=None, **arg_dict):
        #import time
        #start = time.time()
        data_shapes = {k: v.shape for k, v in arg_dict.items()}
        for name in self.learn_init_keys:
            data_shapes[name] = self.learn_init_key_shapes[name]
        self.switch_bucket(bucket_kwargs=bucket_kwargs,
                           data_shapes=data_shapes)
        #end = time.time()
        #print 'Swith Bucket:', end - start
        #start = time.time()
        for k, v in arg_dict.items():
            assert self.exe.arg_dict[k].shape == v.shape,\
                "Shape not match: key %s, need %s, received %s" \
                %(k, str(self.exe.arg_dict[k].shape), str(v.shape))
            self.exe.arg_dict[k][:] = v
        self.exe.forward(is_train=is_train)
        for output in self.exe.outputs:
            output.wait_to_read()
        #end = time.time()
        #print 'Forward:', end - start
        return self.exe.outputs

    def backward(self, out_grads=None, **arg_dict):
        for k, v in arg_dict.items():
            assert self.exe.arg_dict[k].shape == v.shape, \
                "Shape not match: key %s, need %s, received %s" \
                % (k, str(self.exe.arg_dict[k].shape), str(v.shape))
            self.exe.arg_dict[k][:] = v
        self.exe.backward(out_grads=out_grads)

    def forward_backward(self, bucket_kwargs=None, out_grads=None, **arg_dict):
        data_shapes = {k: v.shape for k, v in arg_dict.items()}
        for name in self.learn_init_keys:
            data_shapes[name] = self.learn_init_key_shapes[name]
        self.switch_bucket(bucket_kwargs=bucket_kwargs,
                           data_shapes=data_shapes)
        for k, v in arg_dict.items():
            self.exe.arg_dict[k][:] = v
        self.exe.forward(is_train=True)
        self.exe.backward(out_grads=out_grads)
        for output in self.exe.outputs:
            output.wait_to_read()
        return self.exe.outputs

    def update(self, updater, params_grad=None):
        if params_grad is None:
            params_grad = self.params_grad
        assert type(params_grad) is OrderedDict
        for ind, k in enumerate(self.params.keys()):
            updater(index=ind, grad=params_grad[k], weight=self.params[k])

    def update_acc_grad(self):
        if self.acc_grad is None:
            self.acc_grad = OrderedDict([(n, nd.zeros(v.shape, ctx=self.ctx))
                                         for n, v in self.params_grad.items()])
        for k, v in self.acc_grad.items():
            v[:] = v + self.params_grad[k]

    def reset_acc_grad(self):
        for v in self.acc_grad.values():
            v[:] = 0

    def copy(self, name=None, ctx=None):
        if ctx is None:
            ctx = self.ctx
        if name is None:
            name = self.name + '-copy-' + str(ctx)
        return Base(data_shapes=self.data_shapes,
                    sym_gen=self.sym_gen,
                    default_bucket_kwargs=dict(self.curr_bucket_key),
                    params=self.params,
                    aux_states=self.aux_states, ctx=ctx, name=name)

    def copy_params_to(self, dst):
        for k, v in self.params.items():
            dst.params[k][:] = v
            # TODO `wait_to_read()` here seems unnecessary, remove it in the future!
            dst.params[k].wait_to_read()

    @property
    def total_param_num(self):
        return sum(v.size for v in self.params.values())

    def print_stat(self):
        logging.info("Name: %s" % self.name)
        assert self.params is not None, "Fatal Error!"
        logging.info("Params: ")
        for k, v in self.params.items():
            logging.info("   %s: %s" % (k, v.shape))
        if self.aux_states is None or 0 == len(self.aux_states):
            logging.info("Aux States: None")
        else:
            logging.info("Aux States: " + ' '.join(
                ["%s:%s" % (str(k), str(v.shape)) for k, v in self.aux_states.items()]))
        logging.info("Total Parameter Num: " + str(self.total_param_num))
