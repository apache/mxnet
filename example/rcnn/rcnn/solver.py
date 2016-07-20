import mxnet as mx
import logging
import metric

from callback import Speedometer
from config import config


class Solver(object):
    def __init__(self, prefix,
                 symbol, ctx=None,
                 begin_epoch=0, num_epoch=None,
                 kv_store='local',
                 arg_params=None, aux_params=None,
                 optimizer='sgd',
                 mutable_data_shape=False, max_data_shape=None, max_label_shape=None, **kwargs):
        self.prefix = prefix
        self.symbol = symbol
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.begin_epoch = begin_epoch
        self.num_epoch = num_epoch
        self.kv_store = kv_store
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.optimizer = optimizer
        self.updater = None
        self.mutable_data_shape = mutable_data_shape
        self.max_data_shape = max_data_shape
        self.max_label_shape = max_label_shape
        self.kwargs = kwargs.copy()

        self.check_params()
        self.arg_names = None
        self.param_names = None
        self.aux_names = None

    def get_params(self, grad_req, data_shapes):
        arg_names = self.symbol.list_arguments()
        self.arg_names = arg_names

        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(**dict(data_shapes))
        if grad_req != 'null':
            param_names = []
            for name, shape in zip(arg_names, arg_shapes):
                if not (name.endswith('data') or name.endswith('rois') or
                        name.endswith('im_info') or name.endswith('gt_boxes') or
                        name.endswith('inside_weight') or name.endswith('outside_weight') or
                        name.endswith('label') or name.endswith('target') or
                        name.startswith('conv1') or name.startswith('conv2')):
                    if not (config.TRAIN.FINETUNE and name.startswith('conv')):
                        param_names.append(name)
            self.param_names = param_names

        aux_names = self.symbol.list_auxiliary_states()
        self.aux_names = aux_names

    def check_params(self):
        arg_names = set(self.symbol.list_arguments())
        self.arg_params = {k: v for k, v in self.arg_params.items() if k in arg_names}
        aux_names = set(self.symbol.list_arguments())
        self.aux_params = {k: v for k, v in self.aux_params.items() if k in aux_names}

    def fit(self, train_data,
            grad_req='write',
            frequent=20,
            logger=None):
        (kvstore, update_on_kvstore) = mx.model._create_kvstore(self.kv_store, len(self.ctx), self.arg_params)
        if logger is None:
            logger = logging

        batch_end_callback = Speedometer(train_data.batch_size, frequent=frequent)
        epoch_end_callback = mx.callback.do_checkpoint(self.prefix)

        self.get_params(grad_req, train_data.provide_data + train_data.provide_label)

        if config.TRAIN.HAS_RPN is True:
            eval_metric = metric.AccuracyMetric(use_ignore=True, ignore=-1)
            cls_metric = metric.LogLossMetric(use_ignore=True, ignore=-1)
        else:
            eval_metric = metric.AccuracyMetric()
            cls_metric = metric.LogLossMetric()
        bbox_metric = metric.SmoothL1LossMetric()
        eval_metrics = mx.metric.CompositeEvalMetric()
        for child_metric in [eval_metric, cls_metric, bbox_metric]:
            eval_metrics.add(child_metric)
        mutable_data_shape = self.mutable_data_shape
        max_data_shape = self.max_data_shape
        max_label_shape = self.max_label_shape

        self.optimizer = mx.optimizer.create(self.optimizer,
                                             rescale_grad=(1.0 / config.TRAIN.BATCH_SIZE), **self.kwargs)
        mx.model._train_multi_device(self.symbol, self.ctx, self.arg_names, self.param_names, self.aux_names,
                                     self.arg_params, self.aux_params, self.begin_epoch, self.num_epoch,
                                     epoch_size=None, optimizer=self.optimizer,
                                     kvstore=kvstore, update_on_kvstore=update_on_kvstore,
                                     train_data=train_data, eval_data=None, eval_metric=eval_metrics,
                                     epoch_end_callback=epoch_end_callback, batch_end_callback=batch_end_callback,
                                     logger=logger, work_load_list=None, monitor=None,
                                     mutable_data_shape=mutable_data_shape, max_data_shape=max_data_shape,
                                     max_label_shape=max_label_shape)
