"""Samplers for positive/negative/ignore sample selections.
This module is used to select samples during training.
Based on different strategies, we would like to choose different number of
samples as positive, negative or ignore(don't care). The purpose is to alleviate
unbalanced training target in some circumstances.
The output of sampler is an NDArray of the same shape as the matching results.
Note: 1 for positive, -1 for negative, 0 for ignore.
"""
import numpy as np
from mxnet import gluon
from mxnet import nd
from mxnet import autograd
from .registry import register, alias, create


class Sampler(gluon.Block):
    """A Base class for standard samplers when hybrid_forward is not available."""
    def __init__(self):
        super(Sampler, self).__init__()


class HybridSampler(gluon.HybridBlock):
    """A Base class for hybrid implementation of Samplers."""
    def __init__(self):
        super(HybridSampler, self).__init__()


@register
class NaiveSampler(HybridSampler):
    """A naive sampler that take all existing matching results.
    There is no ignored sample in this case.
    """
    def __init__(self):
        super(NaiveSampler, self).__init__()

    def hybrid_forward(self, F, x, *args, **kwargs):
        marker = F.ones_like(x)
        y = F.where(x >= 0, marker, marker * -1)
        return y


@register
class OHEMSampler(Sampler):
    """A sampler implementing Online Hard-negative mining.
    As described in paper https://arxiv.org/abs/1604.03540.

    Parameters
    ----------

    """
    def __init__(self, ratio, min_samples=0, thresh=0.5):
        super(OHEMSampler, self).__init__()
        assert ratio > 0, "OHEMSampler ratio must > 0, {} given".format(ratio)
        self._ratio = ratio
        self._min_samples = min_samples
        self._thresh = thresh

    def forward(self, x, logits, ious, *args):
        """

        """
        F = nd
        num_positive = F.sum(x > -1, axis=1)
        num_negative = self._ratio * num_positive
        num_total = x.shape[1]  # scalar
        num_negative = F.minimum(F.maximum(self._min_samples, num_negative),
                                 num_total - num_positive)
        positive = logits.slice_axis(axis=2, begin=1, end=-1)
        background = logits.slice_axis(axis=2, begin=0, end=1).reshape((0, -1))
        maxval = positive.max(axis=2)
        esum = F.exp(logits - maxval.reshape((0, 0, 1))).sum(axis=2)
        score = -F.log(F.exp(background - maxval) / esum)
        mask = F.ones_like(score) * -1
        score = F.where(x < 0, score, mask)  # mask out positive samples
        if len(ious.shape) == 3:
            ious = F.max(ious, axis=2)
        score = F.where(ious < self._thresh, score, mask)  # mask out if iou is large
        argmaxs = F.argsort(score, axis=1, is_ascend=False)

        # neg number is different in each batch, using dynamic numpy operations.
        y = np.zeros(x.shape)
        y[np.where(x.asnumpy() >= 0)] = 1  # assign positive samples
        argmaxs = argmaxs.asnumpy()
        for i, num_neg in zip(range(x.shape[0]), num_negative.asnumpy().astype(np.int32)):
            indices = argmaxs[i, :num_neg]
            y[i, indices.astype(np.int32)] = -1  # assign negative samples
        return F.array(y, ctx=x.context)
