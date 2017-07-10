# coding: utf-8
# pylint: disable=arguments-differ
""" losses for training neural networks """
from __future__ import absolute_import

from .. import symbol, ndarray
from ..base import numeric_types
from .block import HybridBlock

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


class L2Loss(HybridBlock):
    """Calculate the mean squared error between output and label:

    .. math::
        L = \\frac{1}{2}\\sum_i \\Vert {output}_i - {label}_i \\Vert^2.

    output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if F is ndarray:
            loss = ndarray.square(output - label.reshape(output.shape))
        else:
            # for symbolic output.shape is not available so we reshape
            # to empty shape and let it be inferred from output's shape
            # via the '-' operator later.
            loss = symbol.square(output - label.reshape(()))
        loss = _apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class L1Loss(HybridBlock):
    """Calculate the mean absolute error between output and label:

    .. math::
        L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert.

    output and label must have the same shape.

    Parameters
    ----------
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if F is ndarray:
            loss = ndarray.abs(output - label.reshape(output.shape))
        else:
            # for symbolic output.shape is not available so we reshape
            # to empty shape and let it be inferred from output's shape
            # via the '-' operator later.
            loss = symbol.abs(output - label.reshape(()))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class SoftmaxCrossEntropyLoss(HybridBlock):
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
    axis : int, default -1
        The axis to sum over when computing softmax and entropy
    sparse_label : bool, default True
        whether label is a integer array instead of probability distribution
    from_logits : bool, default False
        whether input is log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        global scalar weight for loss
    sample_weight : Symbol or None
        per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, sample_weight should have shape (64, 1)
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(**kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self._weight = weight
        self._batch_axis = batch_axis

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if not self._from_logits:
            output = F.log_softmax(output)
        if self._sparse_label:
            loss = -F.pick(output, label, axis=self._axis, keepdims=True)
        else:
            loss = -F.sum(output*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
