# coding: utf-8
# pylint: disable=too-many-arguments, no-member
""" losses for training neural networks """
from __future__ import absolute_import

from .base import numeric_types
from . import symbol


def _apply_weight(loss, weight=None, sample_weight=None):
    assert len(loss.list_outputs()) == 1, "loss symbol must have a single output"

    if sample_weight is not None:
        assert isinstance(sample_weight, symbol.Symbol), "sample_weight must be a Symbol"
        loss = symbol.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss *= weight

    return loss


class Loss(object):
    """Base class for all loss layers.

    """
    def __init__(self, loss, output, label_names, name,
                 output_head_grad=False, loss_head_grad=False):
        if not loss_head_grad:
            self._loss_symbol = symbol.Group([symbol.make_loss(x, name=x.name+'_loss')
                                              for x in loss])
        if not output_head_grad:
            self._output_symbol = symbol.Group([symbol.stop_gradient(x, name=x.name+'_out')
                                                for x in output])
        self._label_names = list(label_names) if label_names else []
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def label_names(self):
        return self._label_names

    @property
    def loss_symbol(self):
        return self._loss_symbol

    @property
    def output_symbol(self):
        return self._output_symbol


def custom_loss(loss, output, label_names, weight=None,
                sample_weight=None, name='custom', **kwargs):
    label_names = list(label_names)
    if sample_weight is not None:
        label_names += [i for i in sample_weight.list_arguments()
                        if i not in loss.list_arguments()]
    loss = _apply_weight(loss, weight=weight, sample_weight=sample_weight)
    loss._set_attr(name=name)
    return Loss(loss, output, label_names, name, **kwargs)


def multi_loss(losses, name='multi'):
    loss = sum([list(l.loss_symbol) for l in losses], [])
    output = sum([list(l.output_symbol) for l in losses], [])
    label_names = []
    for l in losses:
        for name in l.label_names:
            if name not in label_names:
                label_names.append(name)
    return Loss(loss, output, label_names, name,
                output_head_grad=False, loss_head_grad=False)


def l2_loss(output, label, extra_outputs=(), name='l2',
            weight=1., sample_weight=None, **kwargs):
    loss = symbol.square(output - label)
    loss = _apply_weight(loss, weight/2., sample_weight)
    loss._set_attr(name=name)
    label_names = [x for x in loss.list_arguments()
                   if x not in output.list_arguments()]
    outputs = [output] + list(extra_outputs)
    return Loss(loss, outputs, label_names, name, **kwargs)


def l1_loss(output, label, extra_outputs=(), name='l1',
            weight=None, sample_weight=None, **kwargs):
    loss = symbol.abs(output - label)
    loss = _apply_weight(loss, weight, sample_weight)
    loss._set_attr(name=name)
    label_names = [x for x in loss.list_arguments()
                   if x not in output.list_arguments()]
    outputs = [output] + list(extra_outputs)
    return Loss(loss, outputs, label_names, name, **kwargs)


def cross_entropy_loss(output, label, sparse_label=True, axis=1,
                       extra_outputs=(), name='ce',
                       weight=None, sample_weight=None, **kwargs):
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

    loss = _apply_weight(loss, weight, sample_weight)
    loss._set_attr(name=name)
    label_names = [x for x in loss.list_arguments()
                   if x not in output.list_arguments()]
    outputs = [output] + list(extra_outputs)
    return Loss(loss, outputs, label_names, name, **kwargs)


