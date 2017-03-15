# coding: utf-8
# pylint: disable=too-many-arguments, no-member, protected-access, too-many-locals
""" losses for training neural networks """
from __future__ import absolute_import

from .base import numeric_types, string_types, __version__
from . import symbol
from . import registry
from . import metric as _metric


def _apply_weight(loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        the loss to be weighted.
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch sample_weight should have shape (64, 1)

    Returns
    -------
    loss : Symbol
        weighted loss
    """
    assert len(loss.list_outputs()) == 1, "loss symbol must have a single output"

    if sample_weight is not None:
        assert isinstance(sample_weight, symbol.Symbol), "sample_weight must be a Symbol"
        loss = symbol.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss *= weight

    return loss


def _parse_metric(metric, output, label):
    """Create metric and set output/label names"""
    if metric is None:
        return None
    if isinstance(metric, string_types):
        metric = _metric.create(metric)
    if isinstance(label, symbol.Symbol):
        metric.label_names = [i for i in label.list_arguments()
                              if i not in output.list_arguments()]
    else:
        metric.label_names = list(label)
    assert len(metric.label_names) == 1, "too many labels %s"%str(metric.label_names)
    assert len(output.list_outputs()) == 1, "too many outputs %s"%str(output.list_outputs())
    metric.output_names = [output.name + '_out_output']
    return metric


class BaseLoss(object):
    """Base class for all loss layers.

    Parameters
    ----------
    losses : Symbol
        a symbol whose output is the loss. Can be a scalar value
        or an array. If loss is an array, the sum of its elements
        will be the final loss.
    outputs : Symbol
        output of the model when predicting.
    label_names : list of str
        names of label variables. labels are used for training
        and scoring but not for predicting output.
    name : str
        name of this loss
    metric : EvalMetric or None
        metric for training and scoring. If None, only the loss
        values are displayed.
    output_head_grad : bool
        whether output needs head gradient for backward.
    loss_head_grad : bool
        whether loss needs head gradients for backward.

    Returns
    -------
    loss : BaseLoss
        created loss
    """
    def __init__(self, losses, outputs, label_names, name, metric=None,
                 output_head_grad=False, loss_head_grad=False, **kwargs):
        if losses is None:
            sym = list(symbol.load_json(kwargs['__group_sym__']))
            num = kwargs['__num_losses__']
            losses = symbol.Group(sym[:num])
            outputs = symbol.Group(sym[num:])
            if metric is not None:
                metric = _metric.create(**metric)

        losses = symbol.Group(list(losses))
        outputs = symbol.Group(list(outputs))
        self._kwargs = kwargs
        self._kwargs.update({
            'loss': self.__class__.__name__,
            'losses': None,
            'outputs': None,
            'label_names': label_names,
            'name': name,
            'metric': metric.get_config() if metric is not None else None,
            'output_head_grad': output_head_grad,
            'loss_head_grad': loss_head_grad,
            '__group_sym__': symbol.Group(list(losses)+list(outputs)).tojson(),
            '__num_losses__': len(list(losses)),
            '__type__': 'loss',
            '__version__': __version__})

        if not loss_head_grad:
            self._loss_symbol = symbol.Group([symbol.make_loss(x, name=x.name+'_loss')
                                              for x in losses])
        else:
            self._loss_symbol = losses

        if not output_head_grad:
            self._output_symbol = symbol.Group([symbol.stop_gradient(x, name=x.name+'_out')
                                                for x in outputs])
        else:
            self._output_symbol = outputs

        self._label_names = list(label_names) if label_names else []
        self._name = name

        composite = _metric.CompositeEvalMetric()
        if metric is not None:
            composite.add(metric)
        for i in self._loss_symbol:
            composite.add(_metric.Loss(name=i.name, output_names=i.list_outputs(), label_names=[]))
        self._metric = composite

    @property
    def name(self):
        """Name of loss"""
        return self._name

    @property
    def label_names(self):
        """names of label variables used to compute loss"""
        return self._label_names

    @property
    def loss_symbol(self):
        """loss symbol for training and scoring"""
        return self._loss_symbol

    @property
    def output_symbol(self):
        """output symbol for prediction"""
        return self._output_symbol

    @property
    def metric(self):
        """Metric for evaluation"""
        return self._metric

    def get_config(self):
        """get configs for serialization"""
        return self._kwargs.copy()


# pylint: disable=invalid-name
register = registry.get_register_func(BaseLoss, 'loss')
create = registry.get_create_func(BaseLoss, 'loss')
register(BaseLoss)
# pylint: enable=invalid-name


def custom_loss(loss, output, label_names, extra_outputs=(),
                weight=None, sample_weight=None, name='loss',
                metric=None, **kwargs):
    """User defined custom loss. For example, the following
    custom loss has the same effect as l2_loss::
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')
        output = mx.sym.FullyConnected(data, num_hidden=1)
        L = mx.sym.square(output-label)/2
        loss = mx.sym.custom_loss(L, output, label_names=['label'])

    Parameters
    ----------
    loss : Symbol
        a symbol whose output is the loss. Can be a scalar value
        or an array. If loss is an array, the sum of its elements
        will be the final loss.
    output : Symbol
        output of the model when predicting.
    label_names : list of str
        names of label variables. labels are used for training
        and scoring but not for predicting output.
    extra_outputs : list of Symbol
        extra outputs for predition but not used for evaluating
        metric. Normally used when you want to analyze internal
        feature maps.
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    name : str
        name of this loss
    metric : EvalMetric or None
        metric for training and scoring. If None, only the loss
        values are displayed.
    output_head_grad : bool
        whether output needs head gradient for backward.
    loss_head_grad : bool
        whether loss needs head gradients for backward.

    Returns
    -------
    loss : BaseLoss
        created loss
    """
    label_names = list(label_names)
    metric = _parse_metric(metric, output, label_names)
    output = list(output) + list(extra_outputs)
    if sample_weight is not None:
        label_names += [i for i in sample_weight.list_arguments()
                        if i not in loss.list_arguments()]
    loss = _apply_weight(loss, weight=weight, sample_weight=sample_weight)
    loss._set_attr(name=name)
    return BaseLoss(loss, output, label_names, name, **kwargs)


def multi_loss(losses, extra_outputs=(), name='multi'):
    """Combine multiple losses. The final loss is the sum
    of all losses.

    Parameters
    ----------
    losses : list of BaseLoss
        a list of individual losses with no extra outputs.
    extra_outputs : list of Symbol
        extra outputs for predition but not used for evaluating
        metric. Normally used when you want to analyze internal
        feature maps.
    name : str
        name of combined loss

    Returns
    -------
    loss : BaseLoss
        created loss
    """
    loss = symbol.Group(sum([list(i.loss_symbol) for i in losses], []))
    output = symbol.Group(sum([list(i.output_symbol) for i in losses], []) + list(extra_outputs))
    label_names = []
    for i in losses:
        for name in i.label_names:
            if name not in label_names:
                label_names.append(name)
    ret = BaseLoss(loss, output, label_names, name,
                   output_head_grad=True, loss_head_grad=True)
    del ret.metric.metrics[:]
    for i in losses:
        ret.metric.add(i.metric)
    return ret



def l2_loss(output, label, extra_outputs=(), weight=1.,
            sample_weight=None, metric='mse', name='l2',
            **kwargs):
    """Calculate the mean squared error between output and label:

    .. math::
    L = \\frac{1}{2}\\sum_i \\Vert {output}_i - {label}_i \\Vert^2.

    output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    output : Symbol
        output of the network
    label : Symbol
        target to compare output against
    extra_outputs : list of Symbol
        extra outputs for predition but not used for evaluating
        metric. Normally used when you want to analyze internal
        feature maps.
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    metric : EvalMetric or None
        metric for training and scoring. If None, only the loss
        values are displayed.
    name : str
        name of this loss
    output_head_grad : bool
        whether output needs head gradient for backward.
    loss_head_grad : bool
        whether loss needs head gradients for backward.

    Returns
    -------
    loss : BaseLoss
        created loss
    """
    metric = _parse_metric(metric, output, label)
    outputs = [output] + list(extra_outputs)

    output = symbol.reshape(output, shape=(-1,))
    label = symbol.reshape(label, shape=(-1,))
    loss = symbol.square(output - label)
    loss = _apply_weight(loss, weight/2., sample_weight)
    loss._set_attr(name=name)

    label_names = [x for x in loss.list_arguments()
                   if x not in output.list_arguments()]
    return BaseLoss(loss, outputs, label_names, name, metric=metric, **kwargs)


def l1_loss(output, label, extra_outputs=(), name='l1',
            weight=None, sample_weight=None, metric='mae',
            **kwargs):
    """Calculate the mean absolute error between output and label:

    .. math::
    L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert.

    output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    output : Symbol
        output of the network
    label : Symbol
        target to compare output against
    extra_outputs : list of Symbol
        extra outputs for predition but not used for evaluating
        metric. Normally used when you want to analyze internal
        feature maps.
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    metric : EvalMetric or None
        metric for training and scoring. If None, only the loss
        values are displayed.
    name : str
        name of this loss
    output_head_grad : bool
        whether output needs head gradient for backward.
    loss_head_grad : bool
        whether loss needs head gradients for backward.

    Returns
    -------
    loss : BaseLoss
        created loss
    """
    metric = _parse_metric(metric, output, label)
    outputs = [output] + list(extra_outputs)

    output = symbol.reshape(output, shape=(-1,))
    label = symbol.reshape(label, shape=(-1,))
    loss = symbol.abs(output - label)
    loss = _apply_weight(loss, weight, sample_weight)
    loss._set_attr(name=name)

    label_names = [x for x in loss.list_arguments()
                   if x not in output.list_arguments()]
    return BaseLoss(loss, outputs, label_names, name, metric=metric, **kwargs)


def softmax_cross_entropy_loss(output, label, sparse_label=True, axis=1,
                               extra_outputs=(), name='ce', weight=None,
                               sample_weight=None, metric='acc', **kwargs):
    """Compute the softmax cross entropy loss.

    If sparse_label is True, label should contain integer category indicators:
    .. math::
    p = {softmax}({output})
    L = -\\sum_i {log}(p_{i,{label}_i})

    If sparse_label is False, label should cantain probability distribution
    with the same shape as output:
    .. math::
    p = {softmax}({output})
    L = -\\sum_i \\sum_j {label}_j {log}(p_{ij})

    Parameters
    ----------
    output : Symbol
        output of the network
    label : Symbol
        target to compare output against
    sparse_label : bool
        where label is sparse integer or probability distribution
    axis : int
        The axis to sum over when computing softmax and entropy
    extra_outputs : list of Symbol
        extra outputs for predition but not used for evaluating
        metric. Normally used when you want to analyze internal
        feature maps.
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    metric : EvalMetric or None
        metric for training and scoring. If None, only the loss
        values are displayed.
    name : str
        name of this loss
    output_head_grad : bool
        whether output needs head gradient for backward.
    loss_head_grad : bool
        whether loss needs head gradients for backward.

    Returns
    -------
    loss : BaseLoss
        created loss
    """
    metric = _parse_metric(metric, output, label)
    outputs = [output] + list(extra_outputs)

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
    return BaseLoss(loss, outputs, label_names, name, metric=metric, **kwargs)
