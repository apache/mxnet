import mxnet as mx
from mxnet import np, npx
import math

__all__ = ['getF', 'prob2logit', 'logit2prob']

def getF(*params):
    """Get running mode from parameters,
    return mx.ndarray if inputs are python scalar.
    
    Returns
    -------
    ndarray or _Symbol
        the running mode inferred from `*params`
    """
    # TODO: Raise exception when params types are not consistent, i.e. mixed ndarray and symbols.
    for param in params:
        if isinstance(param, np.ndarray):
            return mx.ndarray
        elif isinstance(param, mx.symbol.numpy._Symbol):
            return mx.symbol.numpy._Symbol
    return mx.ndarray


def _clip_prob(prob, F):
    import numpy as onp
    eps = onp.finfo('float32').eps
    return F.np.clip(prob, eps, 1 - eps)


def prob2logit(prob, binary=True, F=None):
    r"""Convert probability to logit form.
    For the binary case, the logit stands for log(p / (1 - p)).
    Whereas for the multinomial case, the logit denotes log(p).
    """
    if F is None:
        F = getF(prob)
    _clipped_prob = _clip_prob(prob, F)
    if binary:
        return F.np.log(_clipped_prob) - F.np.log1p(-_clipped_prob)
    return F.np.log(_clipped_prob)

def logit2prob(logit, binary=True, F=None):
    r"""Convert logit into probability form.
    For the binary case, `sigmoid()` is applied on the logit tensor.
    Whereas for the multinomial case, `softmax` is applied along the last
    dimension of the logit tensor.
    """
    if F is None:
        F = getF(logit)
    if binary:
        return F.npx.sigmoid(logit)
    return F.npx.softmax(logit)