import mxnet as mx
from mxnet import np, npx
from .exp_family import ExponentialFamily
import math

def getF(*params):
    r"""
    Get running mode from parameters,
    return mx.ndarray if inputs are python scalar.
    """
    for param in params:
        if isinstance(param, np.ndarray):
            return mx.ndarray
        elif isinstance(param, mx.symbol.numpy._Symbol):
            return mx.symbol.numpy._Symbol
    return mx.ndarray

def prob2logit(prob, F=None):
    if F is None:
        F = getF(prob)
    return F.np.log(prob) - F.np.log1p(-prob)

def logit2prob(logit, F=None):
    if F is None:
        F = getF(logit)
    return F.npx.sigmoid(logit)