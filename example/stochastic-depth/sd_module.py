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
import mxnet as mx
import numpy as np


class RandomNumberQueue(object):
    def __init__(self, pool_size=1000):
        self._pool = np.random.rand(pool_size)
        self._index = 0

    def get_sample(self):
        if self._index >= len(self._pool):
            self._pool = np.random.rand(len(self._pool))
            self._index = 0
        self._index += 1
        return self._pool[self._index-1]


class StochasticDepthModule(mx.module.BaseModule):
    """Stochastic depth module is a two branch computation: one is actual computing and the
    other is the skip computing (usually an identity map). This is similar to a Residual block,
    except that a random variable is used to randomly turn off the computing branch, in order
    to save computation during training.

    Parameters
    ----------
    symbol_compute: Symbol
        The computation branch.
    symbol_skip: Symbol
        The skip branch. Could be None, in which case an identity map will be automatically
        used. Note the two branch should produce exactly the same output shapes.
    data_names: list of str
        Default is `['data']`. Indicating the input names. Note if `symbol_skip` is not None,
        it should have the same input names as `symbol_compute`.
    label_names: list of str
        Default is None, indicating that this module does not take labels.
    death_rate: float
        Default 0. The probability of turning off the computing branch.
    """
    def __init__(self, symbol_compute, symbol_skip=None,
                 data_names=('data',), label_names=None,
                 logger=logging, context=mx.context.cpu(),
                 work_load_list=None, fixed_param_names=None,
                 death_rate=0):
        super(StochasticDepthModule, self).__init__(logger=logger)

        self._module_compute = mx.module.Module(
            symbol_compute, data_names=data_names,
            label_names=label_names, logger=logger,
            context=context, work_load_list=work_load_list,
            fixed_param_names=fixed_param_names)

        if symbol_skip is not None:
            self._module_skip = mx.module.Module(
                symbol_skip, data_names=data_names,
                label_names=label_names, logger=logger,
                context=context, work_load_list=work_load_list,
                fixed_param_names=fixed_param_names)
        else:
            self._module_skip = None

        self._open_rate = 1 - death_rate
        self._gate_open = True
        self._outputs = None
        self._input_grads = None
        self._rnd_queue = RandomNumberQueue()

    @property
    def data_names(self):
        return self._module_compute.data_names

    @property
    def output_names(self):
        return self._module_compute.output_names

    @property
    def data_shapes(self):
        return self._module_compute.data_shapes

    @property
    def label_shapes(self):
        return self._module_compute.label_shapes

    @property
    def output_shapes(self):
        return self._module_compute.output_shapes

    def get_params(self):
        params = self._module_compute.get_params()
        if self._module_skip:
            params = [x.copy() for x in params]
            skip_params = self._module_skip.get_params()
            for a, b in zip(params, skip_params):
                # make sure they do not contain duplicated param names
                assert len(set(a.keys()) & set(b.keys())) == 0
                a.update(b)
        return params

    def init_params(self, *args, **kwargs):
        self._module_compute.init_params(*args, **kwargs)
        if self._module_skip:
            self._module_skip.init_params(*args, **kwargs)

    def bind(self, *args, **kwargs):
        self._module_compute.bind(*args, **kwargs)
        if self._module_skip:
            self._module_skip.bind(*args, **kwargs)

    def init_optimizer(self, *args, **kwargs):
        self._module_compute.init_optimizer(*args, **kwargs)
        if self._module_skip:
            self._module_skip.init_optimizer(*args, **kwargs)

    def borrow_optimizer(self, shared_module):
        self._module_compute.borrow_optimizer(shared_module._module_compute)
        if self._module_skip:
            self._module_skip.borrow_optimizer(shared_module._module_skip)

    def forward(self, data_batch, is_train=None):
        if is_train is None:
            is_train = self._module_compute.for_training

        if self._module_skip:
            self._module_skip.forward(data_batch, is_train=True)
            self._outputs = self._module_skip.get_outputs()
        else:
            self._outputs = data_batch.data

        if is_train:
            self._gate_open = self._rnd_queue.get_sample() < self._open_rate
            if self._gate_open:
                self._module_compute.forward(data_batch, is_train=True)
                computed_outputs = self._module_compute.get_outputs()
                for i in range(len(self._outputs)):
                    self._outputs[i] += computed_outputs[i]

        else:  # do expectation for prediction
            self._module_compute.forward(data_batch, is_train=False)
            computed_outputs = self._module_compute.get_outputs()
            for i in range(len(self._outputs)):
                self._outputs[i] += self._open_rate * computed_outputs[i]

    def backward(self, out_grads=None):
        if self._module_skip:
            self._module_skip.backward(out_grads=out_grads)
            self._input_grads = self._module_skip.get_input_grads()
        else:
            self._input_grads = out_grads

        if self._gate_open:
            self._module_compute.backward(out_grads=out_grads)
            computed_input_grads = self._module_compute.get_input_grads()
            for i in range(len(self._input_grads)):
                self._input_grads[i] += computed_input_grads[i]

    def update(self):
        self._module_compute.update()
        if self._module_skip:
            self._module_skip.update()

    def update_metric(self, eval_metric, labels):
        self._module_compute.update_metric(eval_metric, labels)
        if self._module_skip:
            self._module_skip.update_metric(eval_metric, labels)

    def get_outputs(self, merge_multi_context=True):
        assert merge_multi_context, "Force merging for now"
        return self._outputs

    def get_input_grads(self, merge_multi_context=True):
        assert merge_multi_context, "Force merging for now"
        return self._input_grads
