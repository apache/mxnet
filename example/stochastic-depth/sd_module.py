import logging
import mxnet as mx
import numpy as np

class StochasticDepthModule(mx.module.BaseModule):
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
            self._gate_open = np.random.rand() < self._open_rate
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

    def get_outputs(self, merge_multi_context=True):
        assert merge_multi_context, "Force merging for now"
        return self._outputs

    def get_input_grads(self, merge_multi_context=True):
        assert merge_multi_context, "Force merging for now"
        return self._input_grads