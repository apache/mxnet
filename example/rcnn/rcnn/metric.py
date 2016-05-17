import mxnet as mx
import numpy as np

from rcnn.config import config


class LogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(LogLossMetric, self).__init__('LogLoss')

    def update(self, labels, preds):
        pred_cls = preds[0].asnumpy()
        label = labels[0].asnumpy().astype('int32')
        cls = pred_cls[np.arange(label.shape[0]), label]
        cls += config.EPS
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class SmoothL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(SmoothL1LossMetric, self).__init__('SmoothL1Loss')

    def update(self, labels, preds):
        bbox_loss = preds[0].asnumpy()
        label = labels[0].asnumpy()
        bbox_loss = np.sum(bbox_loss)
        self.sum_metric += bbox_loss
        self.num_inst += label.shape[0]
