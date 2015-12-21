# pylint: skip-file
import numpy as np
import mxnet as mx
import time
import logging
from collections import namedtuple
from mxnet import optimizer as opt
from mxnet.optimizer import get_updater
from mxnet import metric

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])
class Solver(object):
    def __init__(self, symbol, ctx=None,
                 begin_epoch=0, num_epoch=None,
                 arg_params=None, aux_params=None,
                 optimizer='sgd', **kwargs):
        self.symbol = symbol
        if ctx is None:
            ctx = mx.cpu()
        self.ctx = ctx
        self.begin_epoch = begin_epoch
        self.num_epoch = num_epoch
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.optimizer = optimizer
        self.kwargs = kwargs.copy()

    def fit(self, train_data, eval_data=None,
            eval_metric='acc',
            grad_req='write',
            epoch_end_callback=None,
            batch_end_callback=None,
            kvstore='local',
            logger=None):
        if logger is None:
            logger = logging
        logging.info('Start training with %s', str(self.ctx))
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=train_data.provide_data[0][1])
        arg_names = self.symbol.list_arguments()
        if grad_req != 'null':
            self.grad_params = {}
            for name, shape in zip(arg_names, arg_shapes):
                if not (name.endswith('data') or name.endswith('label')):
                    self.grad_params[name] = mx.nd.zeros(shape, self.ctx)
        else:
            self.grad_params = None
        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k : nd.zeros(s) for k, s in zip(aux_names, aux_shapes)}
        data_name = train_data.data_name
        label_name = train_data.label_name
        input_names = [data_name, label_name]
        self.optimizer = opt.create(self.optimizer, rescale_grad=(1.0/train_data.get_batch_size()), **(self.kwargs))
        self.updater = get_updater(self.optimizer)
        eval_metric = metric.create(eval_metric)
        # begin training
        for epoch in range(self.begin_epoch, self.num_epoch):
            nbatch = 0
            train_data.reset()
            eval_metric.reset()
            for data in train_data:
                nbatch += 1
                label_shape = data[label_name].shape
                self.arg_params[data_name] = mx.nd.array(data[data_name], self.ctx)
                self.arg_params[label_name] = mx.nd.array(data[label_name].reshape(label_shape[0], \
                    label_shape[1]*label_shape[2]), self.ctx)
                output_names = self.symbol.list_outputs()
                self.exector = self.symbol.bind(self.ctx, self.arg_params,
                                args_grad=self.grad_params,
                                grad_req=grad_req,
                                aux_states=self.aux_params)
                assert len(self.symbol.list_arguments()) == len(self.exector.grad_arrays)
                update_dict = {name: nd for name, nd in zip(self.symbol.list_arguments(), \
                    self.exector.grad_arrays) if nd}
                output_dict = {}
                output_buff = {}
                for key, arr in zip(self.symbol.list_outputs(), self.exector.outputs):
                    output_dict[key] = arr
                    output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
                self.exector.forward(is_train=True)
                for key in output_dict:
                    output_dict[key].copyto(output_buff[key])
                self.exector.backward()
                for key, arr in update_dict.items():
                    if key != "bigscore_weight":
                        self.updater(key, arr, self.arg_params[key])
                pred_shape = self.exector.outputs[0].shape
                label = mx.nd.array(data[label_name].reshape(label_shape[0], label_shape[1]*label_shape[2]))
                pred = mx.nd.array(output_buff["softmax_output"].asnumpy().reshape(pred_shape[0], \
                    pred_shape[1], pred_shape[2]*pred_shape[3]))
                eval_metric.update([label], [pred])
                self.exector.outputs[0].wait_to_read()
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=eval_metric)
                batch_end_callback(batch_end_params)
            if epoch_end_callback != None:
                epoch_end_callback(epoch, self.symbol, self.arg_params, self.aux_params)
            name, value = eval_metric.get()
            logger.info("                     --->Epoch[%d] Train-%s=%f", epoch, name, value)
            # evaluation
            if eval_data:
                logger.info(" in eval process...")
                nbatch = 0
                eval_data.reset()
                eval_metric.reset()
                for data in eval_data:
                    nbatch += 1
                    label_shape = data[label_name].shape
                    self.arg_params[data_name] = mx.nd.array(data[data_name], self.ctx)
                    self.arg_params[label_name] = mx.nd.array(data[label_name].reshape(label_shape[0], \
                        label_shape[1]*label_shape[2]), self.ctx)
                    exector = self.symbol.bind(self.ctx, self.arg_params,
                                    args_grad=self.grad_params,
                                    grad_req=grad_req,
                                    aux_states=self.aux_params)
                    cpu_output_array = mx.nd.zeros(exector.outputs[0].shape)
                    exector.forward(is_train=False)
                    exector.outputs[0].copyto(cpu_output_array)
                    pred_shape = cpu_output_array.shape
                    label = mx.nd.array(data[label_name].reshape(label_shape[0], \
                        label_shape[1]*label_shape[2]))
                    pred = mx.nd.array(cpu_output_array.asnumpy().reshape(pred_shape[0], \
                        pred_shape[1], pred_shape[2]*pred_shape[3]))
                    eval_metric.update([label], [pred])
                    exector.outputs[0].wait_to_read()
            name, value = eval_metric.get()
            logger.info('batch[%d] Validation-%s=%f', nbatch, name, value)
