import mxnet as mx
import logging
import metric

from collections import namedtuple
from callback import Speedometer
from config import config


class Solver(object):
    def __init__(self, prefix,
                 symbol, ctx=None,
                 begin_epoch=0, num_epoch=None,
                 arg_params=None, aux_params=None,
                 optimizer='sgd', **kwargs):
        self.prefix = prefix
        self.symbol = symbol
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.begin_epoch = begin_epoch
        self.num_epoch = num_epoch
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.grad_params = None
        self.executor = None
        self.optimizer = optimizer
        self.updater = None
        self.kwargs = kwargs.copy()

    def get_params(self, grad_req):
        arg_names = self.symbol.list_arguments()
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=(1, 3, 224, 224), rois=(1, 5))
        if grad_req != 'null':
            self.grad_params = {}
            for name, shape in zip(arg_names, arg_shapes):
                if not (name.endswith('data') or name.endswith('rois') or
                        name.endswith('inside_weight') or name.endswith('outside_weight') or
                        name.endswith('label') or name.endswith('target') or
                        name.startswith('conv1') or name.startswith('conv2')):
                    self.grad_params[name] = mx.nd.zeros(shape, self.ctx)
        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k: mx.nd.zeros(s, self.ctx) for k, s in zip(aux_names, aux_shapes)}

    def fit(self, train_data,
            grad_req='write',
            frequent=20,
            logger=None):
        if logger is None:
            logger = logging
        logger.info('Start training with %s', str(self.ctx))
        speedometer_param = namedtuple('BatchEndParams',
                                       ['epoch', 'nbatch', 'eval_metric', 'cls_metric', 'bbox_metric'])
        batch_end_callback = Speedometer(train_data.batch_size, frequent=frequent)
        epoch_end_callback = mx.callback.do_checkpoint(self.prefix)

        self.get_params(grad_req)
        self.optimizer = mx.optimizer.create(self.optimizer, rescale_grad=(1.0 / config.TRAIN.BATCH_SIZE), **self.kwargs)
        self.updater = mx.optimizer.get_updater(self.optimizer)

        eval_metric = mx.metric.create("accuracy")
        cls_metric = metric.LogLossMetric()
        bbox_metric = metric.SmoothL1LossMetric()

        # begin training
        for epoch in range(self.begin_epoch, self.num_epoch):
            nbatch = 0
            train_data.reset()
            eval_metric.reset()
            cls_metric.reset()
            bbox_metric.reset()
            for databatch in train_data:
                nbatch += 1
                for k, v in databatch.data.items():
                    self.arg_params[k] = mx.nd.array(v, self.ctx)
                for k, v in databatch.label.items():
                    self.arg_params[k] = mx.nd.array(v, self.ctx)
                self.executor = self.symbol.bind(self.ctx, self.arg_params, args_grad=self.grad_params,
                                                 grad_req=grad_req, aux_states=self.aux_params)
                assert len(self.symbol.list_arguments()) == len(self.executor.grad_arrays)
                update_dict = {name: nd for name, nd
                               in zip(self.symbol.list_arguments(), self.executor.grad_arrays) if nd}
                output_dict = {name: nd for name, nd
                               in zip(self.symbol.list_outputs(), self.executor.outputs)}
                self.executor.forward(is_train=True)
                self.executor.backward()

                for key, arr in update_dict.items():
                    self.updater(key, arr, self.arg_params[key])

                label = self.arg_params['cls_prob_label']
                pred = output_dict['cls_prob_output']
                bb_target = self.arg_params['bbox_loss_target']
                bb_loss = output_dict['bbox_loss_output']
                eval_metric.update([label], [pred])
                cls_metric.update([label], [pred])
                bbox_metric.update([bb_target], [bb_loss])

                # print speed and accuracy metric
                batch_end_params = speedometer_param(epoch=epoch, nbatch=nbatch, eval_metric=eval_metric,
                                                     cls_metric=cls_metric, bbox_metric=bbox_metric)
                batch_end_callback(batch_end_params)

            if epoch_end_callback:
                epoch_end_callback(epoch, self.symbol, self.arg_params, self.aux_params)
            name, value = eval_metric.get()
            logger.info("                     --->Epoch[%d] Train-%s=%f", epoch, name, value)
