# coding: utf-8
from __future__ import absolute_import
from .ndarray import zeros

def momentum(learning_rate = .01, weight_decay = 0.0001, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with momentum

    Parameters
    ----------
    """
    momentums = {}
    def momentum_update(key, grad, weight):
        # weight += - learning_rate * (grad + weight_decay * weight)
        if not momentums.has_key(key):
            momentums[key] = zeros(grad.shape)
        mom = momentums[key]
        mom *= momentum
        mom += - learning_rate * (grad + weight_decay * weight)
        weight += mom

    return momentum_update
