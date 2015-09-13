# coding: utf-8


def momentum(learning_rate = .01, weight_decay = 0.0001, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with momentum

    Parameters
    ----------
    """

    def momentum_update(key, grad, weight):
        mom = momentums[key]
        mom *= momentum
        mom += - learning_rate * (grad + weight_decay * weight)
        weight += mom

    return momentum_update
