# coding: utf-8
# pylint: disable=arguments-differ
"""Utility for using Gluon models."""
__all__ = ['ColorNormalize']
from .. import nn

class ColorNormalize(nn.HybridBlock):
    """Module for color normalization for images."""
    def hybrid_forward(self, F, x):
        # rescale to 0, 1
        x = x / 255.0
        # subtract mean values, divide by std
        mean = F.concat(F.full((1,), 0.485), F.full((1,), 0.456), F.full((1,), 0.406), dim=0)\
                .reshape((1, 3, 1, 1))
        std = F.concat(F.full((1,), 0.229), F.full((1,), 0.224), F.full((1,), 0.225), dim=0)\
               .reshape((1, 3, 1, 1))
        return F.broadcast_div(F.broadcast_sub(x, mean), std)
