# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=arguments-differ
""" losses for training neural networks """
__all__ = ['Loss', 'L2Loss', 'L1Loss',
           'SigmoidBinaryCrossEntropyLoss', 'SigmoidBCELoss',
           'SoftmaxCrossEntropyLoss', 'SoftmaxCELoss',
           'KLDivLoss', 'CTCLoss', 'HuberLoss', 'HingeLoss',
           'SquaredHingeLoss', 'LogisticLoss', 'TripletLoss', 'PoissonNLLLoss', 'CosineEmbeddingLoss', 'SDMLLoss']

import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array


def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        if is_np_array():
            loss = loss * sample_weight
        else:
            loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss


def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    if F is ndarray:
        return x.reshape(y.shape)
    elif is_np_array():
        F = F.npx
    return F.reshape_like(x, y)


def _batch_mean(F, loss, batch_axis):
    """Return mean on the specified batch axis, not keeping the axis"""
    if is_np_array():
        if F is ndarray:
            axes = list(range(loss.ndim))
            del axes[batch_axis]
            return F.np.mean(loss, axis=axes)
        else:
            assert batch_axis == 0, 'Currently, we have not supported the "exclude" ' \
                                    'flag in mean. So we only support batch_axis=0.'
            return F.npx.batch_flatten(loss).mean(axis=1)
    else:
        return F.mean(loss, axis=batch_axis, exclude=True)

def _batch_sum(F, loss, batch_axis):
    """Return sum on the specified batch axis, not keeping the axis"""
    if is_np_array():
        if F is ndarray:
            axes = list(range(loss.ndim))
            del axes[batch_axis]
            return F.np.sum(loss, axis=axes)
        else:
            assert batch_axis == 0, 'Currently, we have not supported the "exclude" ' \
                                    'flag in mean. So we only support batch_axis=0.'
            return F.npx.batch_flatten(loss).sum(axis=1)
    else:
        return F.sum(loss, axis=batch_axis, exclude=True)



class Loss(HybridBlock):
    """Base class for loss.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """

    def __init__(self, weight, batch_axis, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis

    def __repr__(self):
        s = '{name}(batch_axis={_batch_axis}, w={_weight})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """Overrides to construct symbolic graph for this `Block`.

        Parameters
        ----------
        x : Symbol or NDArray
            The first input tensor.
        *args : list of Symbol or list of NDArray
            Additional input tensors.

        """
        # pylint: disable= invalid-name
        raise NotImplementedError


class L2Loss(Loss):
    r"""Calculates the mean squared error between `label` and `pred`.

    .. math:: L = \frac{1}{2} \sum_i \vert {label}_i - {pred}_i \vert^2.

    `label` and `pred` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        square_fn = F.np.square if is_np_array() else F.square
        label = _reshape_like(F, label, pred)
        loss = square_fn(label - pred)
        loss = _apply_weighting(F, loss, self._weight / 2, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


class L1Loss(Loss):
    r"""Calculates the mean absolute error between `label` and `pred`.

    .. math:: L = \sum_i \vert {label}_i - {pred}_i \vert.

    `label` and `pred` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        abs_fn = F.np.abs if is_np_array() else F.abs
        label = _reshape_like(F, label, pred)
        loss = abs_fn(label - pred)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


class SigmoidBinaryCrossEntropyLoss(Loss):
    r"""The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)

    BCE loss is useful when training logistic regression. If `from_sigmoid`
    is False (default), this loss computes:

    .. math::

        prob = \frac{1}{1 + \exp(-{pred})}

        L = - \sum_i {label}_i * \log({prob}_i) * pos\_weight +
            (1 - {label}_i) * \log(1 - {prob}_i)

    If `from_sigmoid` is True, this loss computes:

    .. math::

        L = - \sum_i {label}_i * \log({pred}_i) * pos\_weight +
            (1 - {label}_i) * \log(1 - {pred}_i)

    A tensor `pos_weight > 1` decreases the false negative count, hence increasing
    the recall.
    Conversely setting `pos_weight < 1` decreases the false positive count and
    increases the precision.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    from_sigmoid : bool, default is `False`
        Whether the input is from the output of sigmoid. Set this to false will make
        the loss calculate sigmoid and BCE together, which is more numerically
        stable through log-sum-exp trick.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with values in range `[0, 1]`. Must have the
          same size as `pred`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).
        - **pos_weight**: a weighting tensor of positive examples. Must be a vector with length
          equal to the number of classes.For example, if pred has shape (64, 10),
          pos_weight should have shape (1, 10).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, pred, label, sample_weight=None, pos_weight=None):
        if is_np_array():
            relu_fn = F.npx.relu
            act_fn = F.npx.activation
            abs_fn = F.np.abs
            mul_fn = F.np.multiply
            log_fn = F.np.log
        else:
            relu_fn = F.relu
            act_fn = F.Activation
            abs_fn = F.abs
            mul_fn = F.broadcast_mul
            log_fn = F.log
        label = _reshape_like(F, label, pred)
        if not self._from_sigmoid:
            if pos_weight is None:
                # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
                loss = relu_fn(pred) - pred * label + \
                    act_fn(-abs_fn(pred), act_type='softrelu')
            else:
                # We use the stable formula: x - x * z + (1 + z * pos_weight - z) * \
                #    (log(1 + exp(-abs(x))) + max(-x, 0))
                log_weight = 1 + mul_fn(pos_weight - 1, label)
                loss = pred - pred * label + log_weight * \
                       (act_fn(-abs_fn(pred), act_type='softrelu') + relu_fn(-pred))
        else:
            eps = 1e-12
            if pos_weight is None:
                loss = -(log_fn(pred + eps) * label
                         + log_fn(1. - pred + eps) * (1. - label))
            else:
                loss = -(mul_fn(log_fn(pred + eps) * label, pos_weight)
                         + log_fn(1. - pred + eps) * (1. - label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


SigmoidBCELoss = SigmoidBinaryCrossEntropyLoss


class SoftmaxCrossEntropyLoss(Loss):
    r"""Computes the softmax cross entropy loss. (alias: SoftmaxCELoss)

    If `sparse_label` is `True` (default), label should contain integer
    category indicators:

    .. math::

        \DeclareMathOperator{softmax}{softmax}

        p = \softmax({pred})

        L = -\sum_i \log p_{i,{label}_i}

    `label`'s shape should be `pred`'s shape with the `axis` dimension removed.
    i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape should
    be (1,2,4).

    If `sparse_label` is `False`, `label` should contain probability distribution
    and `label`'s shape should be the same with `pred`:

    .. math::

        p = \softmax({pred})

        L = -\sum_i \sum_j {label}_j \log p_{ij}

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the `axis` dimension removed.
          i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape
          should be (1,2,4) and values should be integers between 0 and 2. If
          `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if is_np_array():
            log_softmax_fn = F.npx.log_softmax
            pick_fn = F.npx.pick
        else:
            log_softmax_fn = F.log_softmax
            pick_fn = F.pick
        if not self._from_logits:
            pred = log_softmax_fn(pred, self._axis)
        if self._sparse_label:
            loss = -pick_fn(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -(pred * label).sum(axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


SoftmaxCELoss = SoftmaxCrossEntropyLoss


class KLDivLoss(Loss):
    r"""The Kullback-Leibler divergence loss.

    KL divergence measures the distance between contiguous distributions. It
    can be used to minimize information loss when approximating a distribution.
    If `from_logits` is True (default), loss is defined as:

    .. math::

        L = \sum_i {label}_i * \big[\log({label}_i) - {pred}_i\big]

    If `from_logits` is False, loss is defined as:

    .. math::

        \DeclareMathOperator{softmax}{softmax}

        prob = \softmax({pred})

        L = \sum_i {label}_i * \big[\log({label}_i) - \log({prob}_i)\big]


    `label` and `pred` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    from_logits : bool, default is `True`
        Whether the input is log probability (usually from log_softmax) instead
        of unnormalized numbers.
    axis : int, default -1
        The dimension along with to compute softmax. Only used when `from_logits`
        is False.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape. If `from_logits` is
          True, `pred` should be log probabilities. Otherwise, it should be
          unnormalized predictions, i.e. from a dense layer.
        - **label**: truth tensor with values in range `(0, 1)`. Must have
          the same size as `pred`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.


    References
    ----------
        `Kullback-Leibler divergence
        <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`_
    """

    def __init__(self, from_logits=True, axis=-1, weight=None, batch_axis=0,
                 **kwargs):
        super(KLDivLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_logits = from_logits
        self._axis = axis

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if is_np_array():
            log_softmax_fn = F.npx.log_softmax
            log_fn = F.np.log
        else:
            log_softmax_fn = F.log_softmax
            log_fn = F.log
        if not self._from_logits:
            pred = log_softmax_fn(pred, self._axis)
        loss = label * (log_fn(label + 1e-12) - pred)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


class CTCLoss(Loss):
    r"""Connectionist Temporal Classification Loss.


    Parameters
    ----------
    layout : str, default 'NTC'
        Layout of prediction tensor. 'N', 'T', 'C' stands for batch size,
        sequence length, and alphabet_size respectively.
    label_layout : str, default 'NT'
        Layout of the labels. 'N', 'T' stands for batch size, and sequence
        length respectively.
    weight : float or None
        Global scalar weight for loss.


    Inputs:
        - **pred**: unnormalized prediction tensor (before softmax).
          Its shape depends on `layout`. If `layout` is 'TNC', pred
          should have shape `(sequence_length, batch_size, alphabet_size)`.
          Note that in the last dimension, index `alphabet_size-1` is reserved
          for internal use as blank label. So `alphabet_size` is one plus the
          actual alphabet size.

        - **label**: zero-based label tensor. Its shape depends on `label_layout`.
          If `label_layout` is 'TN', `label` should have shape
          `(label_sequence_length, batch_size)`.

        - **pred_lengths**: optional (default None), used for specifying the
          length of each entry when different `pred` entries in the same batch
          have different lengths. `pred_lengths` should have shape `(batch_size,)`.

        - **label_lengths**: optional (default None), used for specifying the
          length of each entry when different `label` entries in the same batch
          have different lengths. `label_lengths` should have shape `(batch_size,)`.

    Outputs:
        - **loss**: output loss has shape `(batch_size,)`.


    **Example**: suppose the vocabulary is `[a, b, c]`, and in one batch we
    have three sequences 'ba', 'cbb', and 'abac'. We can index the labels as
    `{'a': 0, 'b': 1, 'c': 2, blank: 3}`. Then `alphabet_size` should be 4,
    where label 3 is reserved for internal use by `CTCLoss`. We then need to
    pad each sequence with `-1` to make a rectangular `label` tensor::

        [[1, 0, -1, -1],
         [2, 1,  1, -1],
         [0, 1,  0,  2]]


    References
    ----------
        `Connectionist Temporal Classification: Labelling Unsegmented
        Sequence Data with Recurrent Neural Networks
        <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_
    """

    def __init__(self, layout='NTC', label_layout='NT', weight=None, **kwargs):
        assert layout in ['NTC', 'TNC'],\
            "Only 'NTC' and 'TNC' layouts for pred are supported. Got: %s" % layout
        assert label_layout in ['NT', 'TN'],\
            "Only 'NT' and 'TN' layouts for label are supported. Got: %s" % label_layout
        self._layout = layout
        self._label_layout = label_layout
        batch_axis = label_layout.find('N')
        super(CTCLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label,
                       pred_lengths=None, label_lengths=None, sample_weight=None):
        if is_np_array():
            swapaxes_fn = F.np.swapaxes
            ctc_fn = F.npx.ctc_loss
        else:
            swapaxes_fn = F.swapaxes
            ctc_fn = F.ctc_loss
        if self._layout == 'NTC':
            pred = swapaxes_fn(pred, 0, 1)
        if self._batch_axis == 1:
            label = swapaxes_fn(label, 0, 1)
        loss = ctc_fn(pred, label, pred_lengths, label_lengths,
                      use_data_lengths=pred_lengths is not None,
                      use_label_lengths=label_lengths is not None,
                      blank_label='last')
        return _apply_weighting(F, loss, self._weight, sample_weight)


class HuberLoss(Loss):
    r"""Calculates smoothed L1 loss that is equal to L1 loss if absolute error
    exceeds rho but is equal to L2 loss otherwise. Also called SmoothedL1 loss.

    .. math::
        L = \sum_i \begin{cases} \frac{1}{2 {rho}} ({label}_i - {pred}_i)^2 &
                           \text{ if } |{label}_i - {pred}_i| < {rho} \\
                           |{label}_i - {pred}_i| - \frac{{rho}}{2} &
                           \text{ otherwise }
            \end{cases}

    `label` and `pred` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    rho : float, default 1
        Threshold for trimmed mean estimator.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, rho=1, weight=None, batch_axis=0, **kwargs):
        super(HuberLoss, self).__init__(weight, batch_axis, **kwargs)
        self._rho = rho

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if is_np_array():
            abs_fn = F.np.abs
            where_fn = F.np.where
            square_fn = F.np.square
        else:
            abs_fn = F.abs
            where_fn = F.where
            square_fn = F.square
        label = _reshape_like(F, label, pred)
        loss = abs_fn(label - pred)
        loss = where_fn(loss > self._rho, loss - 0.5 * self._rho,
                        (0.5 / self._rho) * square_fn(loss))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


class HingeLoss(Loss):
    r"""Calculates the hinge loss function often used in SVMs:

    .. math::
        L = \sum_i max(0, {margin} - {pred}_i \cdot {label}_i)

    where `pred` is the classifier prediction and `label` is the target tensor
    containing values -1 or 1. `label` and `pred` must have the same number of
    elements.

    Parameters
    ----------
    margin : float
        The margin in hinge loss. Defaults to 1.0
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape.
        - **label**: truth tensor with values -1 or 1. Must have the same size
          as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, margin=1, weight=None, batch_axis=0, **kwargs):
        super(HingeLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        relu_fn = F.npx.relu if is_np_array() else F.relu
        label = _reshape_like(F, label, pred)
        loss = relu_fn(self._margin - pred * label)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


class SquaredHingeLoss(Loss):
    r"""Calculates the soft-margin loss function used in SVMs:

    .. math::
        L = \sum_i max(0, {margin} - {pred}_i \cdot {label}_i)^2

    where `pred` is the classifier prediction and `label` is the target tensor
    containing values -1 or 1. `label` and `pred` can have arbitrary shape as
    long as they have the same number of elements.

    Parameters
    ----------
    margin : float
        The margin in hinge loss. Defaults to 1.0
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: truth tensor with values -1 or 1. Must have the same size
          as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, margin=1, weight=None, batch_axis=0, **kwargs):
        super(SquaredHingeLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if is_np_array():
            relu_fn = F.npx.relu
            square_fn = F.np.square
        else:
            relu_fn = F.relu
            square_fn = F.square
        label = _reshape_like(F, label, pred)
        loss = square_fn(relu_fn(self._margin - pred * label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


class LogisticLoss(Loss):
    r"""Calculates the logistic loss (for binary losses only):

    .. math::
        L = \sum_i \log(1 + \exp(- {pred}_i \cdot {label}_i))

    where `pred` is the classifier prediction and `label` is the target tensor
    containing values -1 or 1 (0 or 1 if `label_format` is binary).
    `label` and `pred` can have arbitrary shape as long as they have the same number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    label_format : str, default 'signed'
        Can be either 'signed' or 'binary'. If the label_format is 'signed', all label values should
        be either -1 or 1. If the label_format is 'binary', all label values should be either
        0 or 1.

    Inputs:
        - **pred**: prediction tensor with arbitrary shape.
        - **label**: truth tensor with values -1/1 (label_format is 'signed')
          or 0/1 (label_format is 'binary'). Must have the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, label_format='signed', **kwargs):
        super(LogisticLoss, self).__init__(weight, batch_axis, **kwargs)
        self._label_format = label_format
        if self._label_format not in ["signed", "binary"]:
            raise ValueError("label_format can only be signed or binary, recieved %s."
                             % label_format)

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if is_np_array():
            relu_fn = F.npx.relu
            act_fn = F.npx.activation
            abs_fn = F.np.abs
        else:
            relu_fn = F.relu
            act_fn = F.Activation
            abs_fn = F.abs
        label = _reshape_like(F, label, pred)
        if self._label_format == 'signed':
            label = (label + 1.0) / 2.0  # Transform label to be either 0 or 1
        # Use a stable formula in computation
        loss = relu_fn(pred) - pred * label + \
            act_fn(-abs_fn(pred), act_type='softrelu')
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


class TripletLoss(Loss):
    r"""Calculates triplet loss given three input tensors and a positive margin.
    Triplet loss measures the relative similarity between a positive
    example, a negative example, and prediction:

    .. math::
        L = \sum_i \max(\Vert {pos_i}_i - {pred} \Vert_2^2 -
                        \Vert {neg_i}_i - {pred} \Vert_2^2 + {margin}, 0)

    `positive`, `negative`, and 'pred' can have arbitrary shape as long as they
    have the same number of elements.

    Parameters
    ----------
    margin : float
        Margin of separation between correct and incorrect pair.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **positive**: positive example tensor with arbitrary shape. Must have
          the same size as pred.
        - **negative**: negative example tensor with arbitrary shape Must have
          the same size as pred.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """

    def __init__(self, margin=1, weight=None, batch_axis=0, **kwargs):
        super(TripletLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, pred, positive, negative, sample_weight=None):
        if is_np_array():
            relu_fn = F.npx.relu
            square_fn = F.np.square
        else:
            relu_fn = F.relu
            square_fn = F.square
        positive = _reshape_like(F, positive, pred)
        negative = _reshape_like(F, negative, pred)
        loss = _batch_sum(F, square_fn(positive - pred) - square_fn(negative - pred), self._batch_axis)
        loss = relu_fn(loss + self._margin)
        return _apply_weighting(F, loss, self._weight, sample_weight)


class PoissonNLLLoss(Loss):
    r"""For a target (Random Variable) in a Poisson distribution, the function calculates the Negative
    Log likelihood loss.
    PoissonNLLLoss measures the loss accrued from a poisson regression prediction made by the model.

    .. math::
        L = \text{pred} - \text{target} * \log(\text{pred}) +\log(\text{target!})

    `target`, 'pred' can have arbitrary shape as long as they have the same number of elements.

    Parameters
    ----------
    from_logits : boolean, default True
        indicating whether log(predicted) value has already been computed. If True, the loss is computed as
        :math:`\exp(\text{pred}) - \text{target} * \text{pred}`, and if False, then loss is computed as
        :math:`\text{pred} - \text{target} * \log(\text{pred}+\text{epsilon})`.The default value
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    compute_full: boolean, default False
        Indicates whether to add an approximation(Stirling factor) for the Factorial term in the formula for the loss.
        The Stirling factor is:
        :math:`\text{target} * \log(\text{target}) - \text{target} + 0.5 * \log(2 * \pi * \text{target})`
    epsilon: float, default 1e-08
        This is to avoid calculating log(0) which is not defined.


    Inputs:
        - **pred**:   Predicted value
        - **target**: Random variable(count or number) which belongs to a Poisson distribution.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: Average loss (shape=(1,1)) of the loss tensor with shape (batch_size,).
    """

    def __init__(self, weight=None, from_logits=True, batch_axis=0, compute_full=False, **kwargs):
        super(PoissonNLLLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_logits = from_logits
        self._compute_full = compute_full

    def hybrid_forward(self, F, pred, target, sample_weight=None, epsilon=1e-08):
        if is_np_array():
            exp_fn = F.np.exp
            log_fn = F.np.log
        else:
            exp_fn = F.exp
            log_fn = F.log
        target = _reshape_like(F, target, pred)
        if self._from_logits:
            loss = exp_fn(pred) - target * pred
        else:
            loss = pred - target * log_fn(pred + epsilon)
        if self._compute_full:
            # Using numpy's pi value
            stirling_factor = target * \
                log_fn(target) - target + 0.5 * log_fn(2 * target * np.pi)
            target_gt_1 = target > 1
            stirling_factor = stirling_factor * target_gt_1
            loss = loss + stirling_factor
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)


class CosineEmbeddingLoss(Loss):
    r"""For a target label 1 or -1, vectors input1 and input2, the function computes the cosine distance
    between the vectors. This can be interpreted as how similar/dissimilar two input vectors are.

    .. math::

        L = \sum_i \begin{cases} 1 - {cos\_sim({input1}_i, {input2}_i)} & \text{ if } {label}_i = 1\\
                         {cos\_sim({input1}_i, {input2}_i)} & \text{ if } {label}_i = -1 \end{cases}\\
        cos\_sim(input1, input2) = \frac{{input1}_i.{input2}_i}{||{input1}_i||.||{input2}_i||}

    `input1`, `input2` can have arbitrary shape as long as they have the same number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    margin : float
        Margin of separation between correct and incorrect pair.


    Inputs:
        - **input1**: a tensor with arbitrary shape
        - **input2**: another tensor with same shape as pred to which input1 is
          compared for similarity and loss calculation
        - **label**: A 1-D tensor indicating for each pair input1 and input2, target label is 1 or -1
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as input1. For example, if input1 has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: The loss tensor with shape (batch_size,).
    """

    def __init__(self, weight=None, batch_axis=0, margin=0, **kwargs):
        super(CosineEmbeddingLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, input1, input2, label, sample_weight=None):
        if is_np_array():
            where_fn = F.np.where
            clip_fn = F.np.clip
        else:
            where_fn = F.where
            clip_fn = F.clip

        input1 = _reshape_like(F, input1, input2)
        cos_sim = self._cosine_similarity(F, input1, input2)
        label = _reshape_like(F, label, cos_sim)
        loss = where_fn(label == 1,
                        1 - cos_sim,
                        clip_fn(cos_sim - self._margin, 0, 1 - self._margin))

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return _batch_mean(F, loss, self._batch_axis)

    def _cosine_similarity(self, F, x, y, axis=-1):
        if is_np_array():
            reshape_fn = F.npx.reshape
            norm_fn = F.npx.norm
            sum_fn = F.np.sum
            full_fn = F.np.full
            max_fn = F.np.maximum
        else:
            reshape_fn = F.reshape
            norm_fn = F.norm
            sum_fn = F.sum
            full_fn = F.full
            max_fn = F.broadcast_maximum
        # Calculates the cosine similarity between 2 vectors
        x_norm = reshape_fn(norm_fn(x, axis=axis), (-1, 1))
        y_norm = reshape_fn(norm_fn(y, axis=axis), (-1, 1))
        x_dot_y = reshape_fn(sum_fn(x * y, axis=axis), (-1, 1))
        eps_arr = full_fn((1, 1), 1e-12)
        return (x_dot_y / max_fn(x_norm * y_norm, eps_arr))


class SDMLLoss(Loss):
    r"""Calculates Batchwise Smoothed Deep Metric Learning (SDML) Loss given two input tensors and a smoothing weight
    SDM Loss learns similarity between paired samples by using unpaired samples in the minibatch
    as potential negative examples.

    The loss is described in greater detail in
    "Large Scale Question Paraphrase Retrieval with Smoothed Deep Metric Learning."
    - by Bonadiman, Daniele, Anjishnu Kumar, and Arpit Mittal.  arXiv preprint arXiv:1905.12786 (2019).
    URL: https://arxiv.org/pdf/1905.12786.pdf

    According to the authors, this loss formulation achieves comparable or higher accuracy to
    Triplet Loss but converges much faster.
    The loss assumes that the items in both tensors in each minibatch
    are aligned such that x1[0] corresponds to x2[0] and all other datapoints in the minibatch are unrelated.
    `x1` and  `x2` are minibatches of vectors.

    Parameters
    ----------
    smoothing_parameter : float
        Probability mass to be distributed over the minibatch. Must be < 1.0.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.

    Inputs:
        - **x1**: Minibatch of data points with shape (batch_size, vector_dim)
        - **x2**: Minibatch of data points with shape (batch_size, vector_dim)
          Each item in x2 is a positive sample for the same index in x1.
          That is, x1[0] and x2[0] form a positive pair, x1[1] and x2[1] form a positive pair - and so on.
          All data points in different rows should be decorrelated

    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """

    def __init__(self, smoothing_parameter=0.3, weight=1., batch_axis=0, **kwargs):
        super(SDMLLoss, self).__init__(weight, batch_axis, **kwargs)
        self.kl_loss = KLDivLoss(from_logits=True)
        # Smoothing probability mass
        self.smoothing_parameter = smoothing_parameter

    def _compute_distances(self, F, x1, x2):
        """
        This function computes the euclidean distance between every vector
        in the two batches in input.
        """
        if is_np_array():
            expand_dims_fn = F.np.expand_dims
        else:
            expand_dims_fn = F.expand_dims

        # expanding x1 form [batch_size, dim] to [batch_size, 1, dim]
        # and x2 to [1, batch_size, dim]
        x1_ = expand_dims_fn(x1, 1)
        x2_ = expand_dims_fn(x2, 0)
        # pointwise squared differences
        squared_diffs = (x1_ - x2_)**2
        # sum of squared differences distance
        return squared_diffs.sum(axis=2)


    def _compute_labels(self, F, batch_size):
        """
        The function creates the label matrix for the loss.
        It is an identity matrix of size [BATCH_SIZE x BATCH_SIZE]
        labels:
            [[1, 0]
             [0, 1]]

        after the proces the labels are smoothed by a small amount to
        account for errors.

        labels:
            [[0.9, 0.1]
             [0.1, 0.9]]


        Pereyra, Gabriel, et al. "Regularizing neural networks by penalizing
        confident output distributions." arXiv preprint arXiv:1701.06548 (2017).
        """

        gold = F.eye(batch_size)
        labels = gold * (1 - self.smoothing_parameter) + (1 - gold) * self.smoothing_parameter / (batch_size - 1)
        return labels

    def hybrid_forward(self, F, x1, x2):
        """
        the function computes the kl divergence between the negative distances
        (internally it compute a softmax casting into probabilities) and the
        identity matrix.

        This assumes that the two batches are aligned therefore the more similar
        vector should be the one having the same id.

        Batch1                                Batch2

        President of France                   French President
        President of US                       American President

        Given the question president of France in batch 1 the model will
        learn to predict french president comparing it with all the other
        vectors in batch 2
        """
        assert F is ndarray, 'SDMLLoss does not support symbolic '
        if is_np_array():
            log_softmax_fn = F.npx.log_softmax
        else:
            log_softmax_fn = F.log_softmax
        batch_size = x1.shape[0]
        labels = self._compute_labels(F, batch_size)
        distances = self._compute_distances(F, x1, x2)
        log_probabilities = log_softmax_fn(-distances, axis=1)
        # multiply for the number of labels to obtain the correct loss (gluon kl_loss averages instead of sum)
        # PR#18423:multiply for the number of labels should multiply x1.shape[1] rather than x1.shape[0])
        # After PR#18423, it is no need to multiply it anymore.
        return self.kl_loss(log_probabilities, labels.as_in_context(distances.context))
