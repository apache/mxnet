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
    def __init__(self, ratio):
        super(OHEMSampler, self).__init__()
        self._ratio = ratio

    def forward(self, x, logits, *args):
        """

        """
        F = nd
        num_positive = F.sum(x >= 0, axis=1)
        num_negative = self._ratio * num_positive
        num_total = x.shape[1]
        num_negative = num_negative.clip(a_min=0, a_max=num_total)
        positive = logits.slice_axis(axis=2, begin=1, end=None)
        maxval = positive.max(axis=2)
        esum = F.exp(logits - maxval).sum(axis=2)
        score = -F.log(maxval / esum)
        score = F.where(x < 0, scores, -1)  # mask out positive samples
        argmaxs = F.argsort(score, axis=1, is_ascend=False)
        pos = F.ones_like(x)
        ignore = F.zeros_like(x)
        y = F.where(x >= 0, pos, ignore)
