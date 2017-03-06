# coding: utf-8
# pylint: disable=
""" losses for training neural networks """
from __future__ import absolute_import

from .base import numeric_types
from . import symbol

class BaseLoss(object):
    def __init__(self, loss, outputs, label_names, weight=None,
                 sample_weight=None, metrics=()):
        assert len(loss.list_outputs()) == 1, "loss symbol must have a single output"
        label_names = list(label_names) if label_names else []

        if sample_weight is not None:
            assert isinstance(sample_weight, symbol.Symbol), "sample_weight must be a Symbol"
            label_names += [name for name in sample_weight.list_arguments()
                            if x not in loss.list_arguments()]
            loss = symbol.broadcast_mul(loss, sample_weight)

        if weight is not None:
            assert isinstance(weight, numeric_types), "weight must be a number"
            loss *= weight

        loss = symbol.make_loss(loss)
        outputs = symbol.Group([symbol.stop_gradient(x) for x in outputs])

        self._loss_symbol = loss
        self._output_symbol = outputs
        self._label_names = label_names
        self._metrics = metrics

    @property
    def label_names(self):
        return self._label_names

    @property
    def loss_symbol(self):
        return self._loss_symbol

    @property
    def output_symbol(self):
        return self._output_symbol

    def get_metrics(self):
        return self._metrics


class GenericLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super(GenericLoss, self).__init__(*args, **kwargs)


class MultiLoss(BaseLoss):
    """docstring for MultiLoss"""
    def __init__(self, losses):
        super(MultiLoss, self).__init__()
        

class L2Loss(BaseLoss):
    def __init__(self, output, label, extra_outputs=(), **kwargs):
        loss = symbol.square(output - label)
        label_names = [x for x in label.list_arguments()
                       if x not in output.list_arguments()]
        outputs = [output] + list(extra_outputs)
        super(L2Loss, self).__init__(loss, outputs, label_names, **kwargs)


class L1Loss(BaseLoss):
    def __init__(self, output, label, extra_outputs=(), **kwargs):
        loss = symbol.abs(output - label)
        label_names = [x for x in label.list_arguments()
                       if x not in output.list_arguments()]
        outputs = [output] + list(extra_outputs)
        super(L1Loss, self).__init__(loss, outputs, label_names, **kwargs)


class CrossEntropyLoss(BaseLoss):
    def __init__(self, output, label, sparse_label=True, axis=1,
                 extra_outputs=(), **kwargs):
        # TODO(Eric): make a log_softmax op
        basis = symbol.max(output, axis=axis, keepdims=True)
        basis = symbol.stop_gradient(basis)
        output = symbol.broadcast_sub(output, basis)
        norm = symbol.log(symbol.sum(symbol.exp(output), axis=axis, keepdims=True))
        output = symbol.broadcast_sub(output, norm)

        if sparse_label:
            loss = -symbol.pick(output, label, axis=axis, keepdims=False)
        else:
            loss = -symbol.sum(output*label, axis=axis, keepdims=False)

        label_names = [x for x in label.list_arguments()
                       if x not in output.list_arguments()]
        outputs = [output] + list(extra_outputs)
        super(CrossEntropyLoss, self).__init__(loss, outputs, label_names, **kwargs)


