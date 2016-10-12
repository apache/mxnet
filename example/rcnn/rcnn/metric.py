"""
if config.END2END = 1, then preds =
[cls_label, rpn_cls_loss, rpn_bbox_loss, cls_loss, bbox_loss]
"""

import mxnet as mx
import numpy as np

from rcnn.config import config


class AccuracyMetric(mx.metric.EvalMetric):
    def __init__(self, use_ignore=False, ignore=None, ex_rpn=False):
        if ex_rpn:
            super(AccuracyMetric, self).__init__('RPN-Accuracy')
        else:
            super(AccuracyMetric, self).__init__('Accuracy')
        self.use_ignore = use_ignore
        self.ignore = ignore
        self.ex_rpn = ex_rpn  # used in end2end joint training, export rpn loss
        self.has_rpn = config.TRAIN.HAS_RPN and config.END2END != 1
        if self.has_rpn:
            assert self.use_ignore and self.ignore is not None

    def update(self, labels, preds):
        if self.has_rpn:
            pred_label = mx.ndarray.argmax_channel(preds[0]).asnumpy().astype('int32')
            label = labels[0].asnumpy().astype('int32')
            non_ignore_inds = np.where(label != self.ignore)
            pred_label = pred_label[non_ignore_inds]
            label = label[non_ignore_inds]
        else:
            if config.END2END != 1:
                last_dim = preds[0].shape[-1]
                pred_label = preds[0].asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
                label = labels[0].asnumpy().reshape(-1,).astype('int32')
            else:
                if self.ex_rpn:
                    pred_label = mx.ndarray.argmax_channel(preds[1]).asnumpy().astype('int32').reshape(1,-1)
                    label = labels[0].asnumpy().astype('int32')
                    # import pdb;pdb.set_trace()
                    non_ignore_inds = np.where(label != self.ignore)
                    pred_label = pred_label[non_ignore_inds]
                    label = label[non_ignore_inds]
                else:
                    last_dim = preds[3].shape[-1]
                    pred_label = preds[3].asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
                    label = preds[0].asnumpy().reshape(-1,).astype('int32')

        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)


class LogLossMetric(mx.metric.EvalMetric):
    def __init__(self, use_ignore=False, ignore=None, ex_rpn=False):
        if ex_rpn:
            super(LogLossMetric, self).__init__('RPN-LogLoss')
        else:
            super(LogLossMetric, self).__init__('LogLoss')
        self.use_ignore = use_ignore
        self.ignore = ignore
        self.ex_rpn = ex_rpn
        self.has_rpn = config.TRAIN.HAS_RPN and config.END2END != 1
        if self.has_rpn:
            assert self.use_ignore and self.ignore is not None

    def update(self, labels, preds):
        if self.has_rpn:
            pred_cls = preds[0].asnumpy()[0]
            label = labels[0].asnumpy().astype('int32')[0]
            non_ignore_inds = np.where(label != self.ignore)[0]
            label = label[non_ignore_inds]
            cls = pred_cls[label, non_ignore_inds]
        else:
            if config.END2END != 1:
                last_dim = preds[0].shape[-1]
                pred_cls = preds[0].asnumpy().reshape(-1, last_dim)
                label = labels[0].asnumpy().reshape(-1,).astype('int32')
                cls = pred_cls[np.arange(label.shape[0]), label]
            else:
                if self.ex_rpn:
                    pred_cls = preds[1].asnumpy()[0].reshape(2, -1)
                    label = labels[0].asnumpy().astype('int32')[0]
                    non_ignore_inds = np.where(label != self.ignore)[0]
                    label = label[non_ignore_inds]
                    cls = pred_cls[label, non_ignore_inds]
                else:
                    last_dim = preds[3].shape[-1]
                    pred_cls = preds[3].asnumpy().reshape(-1, last_dim)
                    label = preds[0].asnumpy().reshape(-1,).astype('int32')
                    cls = pred_cls[np.arange(label.shape[0]), label]
        cls += config.EPS
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class SmoothL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, ex_rpn=False):
        if ex_rpn:
            super(SmoothL1LossMetric, self).__init__('RPN-SmoothL1Loss')
        else:
            super(SmoothL1LossMetric, self).__init__('SmoothL1Loss')
        self.ex_rpn = ex_rpn
        self.has_rpn = config.TRAIN.HAS_RPN and config.END2END != 1

    def update(self, labels, preds):
        bbox_loss = preds[1].asnumpy()
        if self.has_rpn:
            bbox_loss = bbox_loss.reshape((bbox_loss.shape[0], -1))
        else:
            if config.END2END != 1:
                first_dim = bbox_loss.shape[0] * bbox_loss.shape[1]
                bbox_loss = bbox_loss.reshape(first_dim, -1)
            else:
                if self.ex_rpn:
                    bbox_loss = preds[2].asnumpy()
                    bbox_loss = bbox_loss.reshape((bbox_loss.shape[0], -1))
                else:
                    bbox_loss = preds[-1].asnumpy()
                    first_dim = bbox_loss.shape[0] * bbox_loss.shape[1]
                    bbox_loss = bbox_loss.reshape(first_dim, -1)
        self.num_inst += bbox_loss.shape[0]
        bbox_loss = np.sum(bbox_loss)
        self.sum_metric += bbox_loss
