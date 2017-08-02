"""Samplers for positive/negative/ignore sample selections.
This module is used to select samples during training.
Based on different strategies, we would like to choose different number of
samples as positive, negative or ignore(don't care). The purpose is to alleviate
unbalanced training target in some circumstances.
The output of sampler is an NDArray of the same shape as the matching results.
Note: 1 for positive, -1 for negative, 0 for ignore.
"""
from mxnet import gluon
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

    def hybrid_foward(self, x, *args, **kwargs):
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

    def forward(self, x, *args):
        pass
