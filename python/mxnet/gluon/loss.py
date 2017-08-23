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
from __future__ import absolute_import

from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock

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
        loss = F.broadcast_mul(loss, sample_weight)

    if weight is not None:
        assert isinstance(weight, numeric_types), "weight must be a number"
        loss = loss * weight

    return loss

def _reshape_label_as_output(F, output, label):
    # for symbolic output.shape is not available so we reshape
    # to empty shape and let it be inferred from output's shape
    # via the '-' operator later.
    return label.reshape(output.shape) if F is ndarray else label.reshape(())

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
    """Calculates the mean squared error between output and label:

    .. math::
        L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert^2.

    Output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(L2Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.square(output - label)
        loss = _apply_weighting(F, loss, self._weight/2, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class L1Loss(Loss):
    """Calculates the mean absolute error between output and label:

    .. math::
        L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert.

    Output and label must have the same shape.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(L1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.abs(output - label)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class SigmoidBinaryCrossEntropyLoss(Loss):
    r"""The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)

    BCE loss is useful when training logistic regression.

    .. math::
        loss(o, t) = - 1/n \\sum_i (t[i] * \\log(o[i]) + (1 - t[i]) * \\log(1 - o[i]))


    Parameters
    ----------
    from_sigmoid : bool, default is `False`
        Whether the input is from the output of sigmoid. Set this to false will make
        the loss calculate sigmoid and then BCE, which is more numerically stable through
        log-sum-exp trick.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        if not self._from_sigmoid:
            max_val = F.maximum(-output, 0)
            loss = output - output*label + max_val + F.log(F.exp(-max_val)+F.exp(-output-max_val))
        else:
            loss = -(F.log(output+1e-8)*label + F.log(1.-output+1e-8)*(1.-label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

SigmoidBCELoss = SigmoidBinaryCrossEntropyLoss


class SoftmaxCrossEntropyLoss(Loss):
    """Computes the softmax cross entropy loss. (alias: SoftmaxCELoss)

    If `sparse_label` is `True`, label should contain integer category indicators:

    .. math::
        p = {softmax}({output})

        L = -\\sum_i {log}(p_{i,{label}_i})

    Label's shape should be output's shape without the `axis` dimension. i.e. for
    `output.shape` = (1,2,3,4) and axis = 2, `label.shape` should be (1,2,4).

    If `sparse_label` is `False`, label should contain probability distribution
    with the same shape as output:

    .. math::
        p = {softmax}({output})

        L = -\\sum_i \\sum_j {label}_j {log}(p_{ij})

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
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if not self._from_logits:
            output = F.log_softmax(output)
        if self._sparse_label:
            loss = -F.pick(output, label, axis=self._axis, keepdims=True)
        else:
            loss = -F.sum(output*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

SoftmaxCELoss = SoftmaxCrossEntropyLoss


class KLDivLoss(Loss):
    """The Kullback-Leibler divergence loss.

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
    .. math::
        L = 1/n \\sum_i (label_i * (log(label_i) - output_i))

    Label's shape should be the same as output's.

    Parameters
    ----------
    from_logits : bool, default is `True`
        Whether the input is log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, from_logits=True, weight=None, batch_axis=0, **kwargs):
        super(KLDivLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_logits = from_logits

    def hybrid_forward(self, F, output, label, sample_weight=None):
        if not self._from_logits:
            output = F.log_softmax(output)
        loss = label * (F.log(label+1e-8) - output)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class Huber(Loss):
    """Calculates Huber's robust loss function yielding a trimmed mean estimator, i.e. 
       L2 loss in the center and L1 loss for deviations beyond rho:

    .. math::
        L = \\begin{cases} \\frac{1}{2 \\rho} ({output}_i - {label}_i)^2 & 
                           \\text{ if } |{output}_i - {label}_i| < \\rho \\\
                           |{output}_i - {label}_i| - \\frac{\\rho}{2} & 
                           \\text{ otherwise }
            \\end{cases}

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    rho : float
        Threshold for trimmed mean estimator. By default set to 1
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, rho=1, weight=None, batch_axis=0, **kwargs):
        super(Huber, self).__init__(weight, batch_axis, **kwargs)
        self._rho = rho

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.abs(output - label)
        loss = (loss > self._rho) * (loss - 0.5 * self._rho) + (0.5/self._rho) * (loss <= self._rho) * loss**2
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class SoftMargin(Loss):
    """Calculates the soft-margin loss function used in SVMs:

    .. math::
        L = max(0, 1 - {output}_i {label}_i)

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(SoftMargin, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.maximum(1.0 - output * label, F.zeros_like(output))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class SquaredSoftMargin(Loss):
    """Calculates the soft-margin loss function used in SVMs:

    .. math::
        L = max(0, 1 - {output}_i {label}_i)^2

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(SquaredSoftMargin, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.maximum(1.0 - output * label, F.zeros_like(output))**2
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
    
class Exponential(Loss):
    """Calculates the exponential hinge loss (quite obscure):

    .. math::
        L = \\exp(- {output}_i {label}_i)

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(Exponential, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.exp(-output * label)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
    
class Logistic(Loss):
    """Calculates the logistic loss (for binary losses only):

    .. math::
        L = \\log(1 + \\exp(- {output}_i {label}_i))

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(Logistic, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.log(1.0 + F.exp(-output * label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
    
class Quantile(Loss):
    """Calculates Koenker's quantile regression loss function yielding an estimate of the
       appropriately chosen quantile rather than the mean (or median):

    .. math::
        L = {max}(\\tau ({output}_i - {label}_i), (1-\\tau) ({label}_i - {output}_i)

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    tau : float
        Quantile of the estimator. By default set to 0.5, i.e. by
        default identical with L1 loss, up to a scaling factor.
        This must be in the range (0,1).
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, tau=0.5, weight=None, batch_axis=0, **kwargs):
        super(Quantile, self).__init__(weight, batch_axis, **kwargs)
        self._tau = tau

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = output - label
        loss = F.maximum(self._tau * loss, (self._tau - 1.0)* loss)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class Langford(Loss):
    """Calculates the Huberized soft-margin loss that is used in VW (Vowpal Wabbit). 
       It is given by a squared loss for margin values of [-1, 0] and by a linear 
       loss for values larger than that. 

    .. math::
        L = \\begin{cases}
          0 & 
          \\text{ if } {output}_i {label}_i > 1 \\\
          \\frac{1}{2} - {output}_i {label}_i & 
          \\text{ if } {output}_i {label}_i < 0 \\\
          \\frac{1}{2} (1 - {output}_i {label}_i)^2 & 
          \\text{ otherwise }
          \\end{cases}

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(Langford, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.maximum(F.zeros_like(output), 1 - output * label)
        loss = (loss < 1.0) * 0.5 * (loss**2) + (loss >= 1.0) * (loss - 0.5)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class DualKL(Loss):
    """Estimates the Kullback Leibler Divergence between two
       distributions by convex duality. See Nguyen, Wainwright and
       Jordan (NGW), 2008 for a detailed derivation. In a nutshell it
       estimates:

    .. math::
       KL(p\\|q) = E_p[\\log p(x)] - E_p[\\log q(x)]

       Clearly this isn't easy to compute. Hence, NGW use the dual of
       the F-divergence log p(x)/q(x) and pose it as an optimization
       problem. This leads to the following loss function, which is
       different for both distributions (which we treat as a binary
       classification problem). The function that is being estimated
       allows us to get the Radon-Nikodym via dp/dq = exp(f).

    .. math::
        L = \\begin{cases}
            \\exp(f) & \\text{ if } y = -1 \\\
             -f-1 & \text{ if } y = 1
          \\end{cases}

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(DualKL, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = (label == -1) * F.exp(output) - (label == 1) * (output + 1)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
    
class RelativeNovelty(Loss):
    """Estimates a relative novelty detector. See the Song, Teo and
       Smola (STS), 2009 for details. The main point is to estimate
       the ratio dp/dq well via max(0, rho - log dp/dq). As with the
       KL divergence estimator, the Fenchel-Legendre dual is easier to
       deal with. This leads to the following loss function:

    .. math::
        L = \\begin{cases}
            \\exp(f - rho) & \\text{ if } y = -1 \\\
             -f-1 & \\text{ if } y = 1 \\text{ and } f > 0
            \\exp(f) & \\text{ if } y = 1 \\text{ and } f <= 0
          \\end{cases}

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    rho : float
        Relative probability weight for the most prevalent part of the
        probability distribution. It needs to be (0, 1).
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, rho=0.1, weight=None, batch_axis=0, **kwargs):
        super(RelativeNovelty, self).__init__(weight, batch_axis, **kwargs)
        self._rho = rho

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = -(output > 0) * (output + 1) - (output <= 0) * F.exp(output)
        loss = (label == 1) * loss + (label == -1) * F.exp(f - self._rho)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
    
class LogCosh(Loss):
    """Calculates the smoothed L1 loss, aka log cosh loss in a
       numerically stable manner (i.e. without exponentiating large
       values of the cosh function. 

    .. math::
        L = \\log 2 \\cosh ({output}_i - {label}_i)

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(LogCosh, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.abs(label - output)
        loss += F.log(1.0 + F.exp(-loss))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
    
class Poisson(Loss):
    """Calculates the Poisson loss function (up to the normalization
       by a factorial in the label, due to computational efficiency
       reasons). 
       
       NOTE THAT THIS IS DIFFERENT FROM THE POISSON LOSS IN PYTORCH
       AND KERAS INSOFAR AS IT USES THE EPXONENTIAL VERSION. THAT ONE
       DOESN'T SUFFER FROM LOG 0 PROBLEMS.

    .. math::
        L = -\\log p({label}_i|{output}_i) 
          = \\log {label}_i! + \\exp({output}_i) - {output}_i {label}_i

    Output and label must have the same shape. This is a scalar loss function.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(Poisson, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        label = _reshape_label_as_output(F, output, label)
        loss = F.exp(output) - output * label
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class MaxMargin(Loss):
    """Calculates the MaxMargin loss, aka multiclass soft-margin
       loss. This requires access to a multiclass loss matrix delta
       which measures the cost for misclassifying label y as y'. This
       matrix can be specified at construction time. If it does not
       exist, we will susbstitute it with a 0-1 loss with automagic
       size inference. 

    .. math::
       L = {max}_{y} [\\delta({label}, y) + {output}[y]] - {output}_{label}

    Label's shape should be output's shape without the `axis` dimension. i.e. for
    `output.shape` = (1,2,3,4) and axis = 2, `label.shape` should be (1,2,4).

    Parameters
    ----------
    delta : loss matrix, default None. In this case it is presumed to
        be a (0,1) loss, i.e. a constant loss of 1 for all
        misclassifications. Otherwise its dimensionality must match
        that of the number of classes. 
    axis : int, default -1
        The axis to sum over when taking the maximum. 
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, delta=None, axis=-1, weight=None, batch_axis=0, **kwargs):
        super(MaxMargin, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._delta = delta
        super(MaxMargin, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, sample_weight=None):
        # Check if the cost matrix has been defined. If not, use a
        # dumb (0,1) loss. This is only executed once, the first time
        # you invoke the loss. 
        if (self._delta == None):
            classes = output.shape[self._axis]
            self._delta = F.ones(shape=(classes, classes))
            for i in range(classes):
                self._delta[i,i] = 0
        loss = -F.pick(output, label, axis=self._axis, keepdims=True)
        loss += F.max(output + F.take(self._delta, label), axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class TripletLoss(Loss):
    """Calculates the mean squared error between output and label:

    .. math::
        L = \\frac{1}{2}\\sum_i \\vert {output}_i - {label}_i \\vert^2.

    Output and label can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    margin : float
        Margin of separation between correct and incorrect pair. 
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch, `sample_weight` should have shape (64, 1).
    axis : int, default 1
        The axis over which to sum distances. 
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, margin=1, weight=1., axis=1, batch_axis=0, **kwargs):
        super(TripletLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin
        self._axis = axis
        
    def hybrid_forward(self, F, output1, output2, output3, sample_weight=None):
        loss = F.sum((f1-f2)**2 - (f1-f3)**2, axis=self._axis) + self._margin
        loss = nd.maximum(loss, F.zeros_like(loss))
        return F.mean(loss, axis=self._batch_axis, exclude=True)

