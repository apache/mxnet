# coding: utf-8
# pylint: disable=too-many-arguments, no-member, protected-access, too-many-locals
# pylint: disable=unused-argument
""" losses for training neural networks """
from __future__ import absolute_import

import json

from ... import symbol, ndarray, metric
from ...base import numeric_types


def _get_F(x):
    """Get function domain from tensor"""
    return symbol if isinstance(x, symbol.Symbol) else ndarray


def _apply_weighting(F, loss, weight=None, sample_weight=None):
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
        in the batch separately, sample_weight should have
        shape (64, 1)

    Returns
    -------
    loss : Symbol
        weighted loss
    """
    if sample_weight is not None:
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss


def _unpack_symbol(loss):
    """unpack a loss symbol into outputs, extra_outputs and losses"""
    assert isinstance(loss, symbol.Symbol)
    outputs = symbol.Group([i for i in loss if i.attr('__output__') == 'pred'])
    extra_outputs = symbol.Group([i for i in loss if i.attr('__output__') == 'extra'])
    losses = symbol.Group([i for i in loss if i.attr('__output__') == 'loss'])
    return outputs, extra_outputs, losses


def custom_loss(loss, output, label, weight=None, sample_weight=None, batch_axis=0,
                extra_outputs=(), metrics=None, name='custom'):
    """Construct user defined loss symbol.

    Parameters
    ----------
    loss : Symbol
        loss value computed from output and label.
    output : Symbol
        output of the network
    label : Symbol
        target to compare output against
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    batch_axis : int, default 0
        The axis that represents mini-batch.

    Returns
    -------
    loss : BaseLoss
        created loss

    Example
    -------
    The following code defines a least square loss (same as `nn.l2_loss`)::
        data = mx.sym.var('data')
        output = mx.sym.FullyConnected(data, num_hidden=1)
        label = mx.sym.var('label')
        loss = mx.sym.square(output - label.reshape((-1, 1)))/2
        loss = nn.custom_loss(loss, output, label, name='l2')
    """
    F = _get_F(loss)
    loss = _apply_weighting(F, loss, weight, sample_weight)
    loss = F.mean(loss, axis=batch_axis, exclude=True)
    if F is ndarray:
        return loss
    outputs = symbol.Group([F.stop_gradient(i, name=i.name+'_out', __output__='pred')
                            for i in output])
    extra_outputs = symbol.Group([F.stop_gradient(i, name=i.name+'_out', __output__='extra')
                                  for i in extra_outputs])

    loss = F.make_loss(loss, name=name, __output__='loss')

    if metrics:
        metrics = metric.create(metrics)
        metrics.output_names = outputs.list_outputs()
        metrics.label_names = label.list_outputs()
        loss._set_attr(__metric__=json.dumps(metrics.get_config()))

    return symbol.Group([outputs, extra_outputs, loss])


def multitask_loss(losses):
    """Combine multiple losses together for multitask learning.

    Parameters
    ----------
    losses : list of Symbol
        list of losses to be combined.
    """
    F = _get_F(losses[0])
    if F is ndarray:
        return losses
    out, extra, loss = zip(*[_unpack_symbol(i) for i in losses])
    return symbol.Group(out+extra+loss)


def l2_loss(output, label, weight=1., sample_weight=None, batch_axis=0,
            extra_outputs=(), metrics=None, name='l2'):
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
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    batch_axis : int, default 0
        The axis that represents mini-batch.

    Returns
    -------
    loss : Symbol
        created loss
    """
    if isinstance(output, ndarray.NDArray):
        loss = ndarray.square(output - label.reshape(output.shape))
    else:
        loss = symbol.square(output - label.reshape(()))
    return custom_loss(loss, output, label, weight/2, sample_weight, batch_axis,
                       extra_outputs, metrics, name)


def l1_loss(output, label, weight=None, sample_weight=None, batch_axis=0,
            extra_outputs=(), metrics=None, name='l1'):
    """Calculate the mean absolute error between output and label:

    .. math::
    L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert.

    output and label must have the same shape.

    Parameters
    ----------
    output : Symbol
        output of the network
    label : Symbol
        target to compare output against
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    batch_axis : int, default 0
        The axis that represents mini-batch.

    Returns
    -------
    loss : Symbol
        created loss
    """
    if isinstance(output, ndarray.NDArray):
        loss = ndarray.abs(output - label.reshape(output.shape))
    else:
        loss = symbol.abs(output - label.reshape(()))
    return custom_loss(loss, output, label, weight, sample_weight, batch_axis,
                       extra_outputs, metrics, name)


def softmax_cross_entropy_loss(output, label, sparse_label=True, axis=-1,
                               weight=None, sample_weight=None, batch_axis=0,
                               extra_outputs=(), metrics='acc', name='ce'):
    """Compute the softmax cross entropy loss.

    If sparse_label is True, label should contain integer category indicators:
    .. math::
    p = {softmax}({output})
    L = -\\sum_i {log}(p_{i,{label}_i})
    label's shape should be output's shape without the `axis` dimension. i.e. for
    output.shape = (1,2,3,4) and axis = 2, label.shape should be (1,2,4)

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
    sparse_label : bool, default True
        where label is sparse integer or probability distribution
    axis : int, default -1
        The axis to sum over when computing softmax and entropy
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    batch_axis : int, default 0
        The axis that represents mini-batch.

    Returns
    -------
    loss : Symbol
        created loss
    """
    F = _get_F(output)
    prob = F.log_softmax(output)
    if sparse_label:
        loss = -F.pick(prob, label, axis=axis, keepdims=True)
    else:
        loss = -F.sum(prob*label, axis=axis, keepdims=True)
    return custom_loss(loss, prob, label, weight, sample_weight, batch_axis,
                       extra_outputs, metrics, name)
