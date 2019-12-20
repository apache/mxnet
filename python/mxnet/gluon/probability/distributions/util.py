from mxnet import np, npx
from .exp_family import ExponentialFamily
from .distribution import getF
import math

def prob2logit(prob, F):
    return F.np.log(prob) - F.np.log1p(-y)

def logit2prob(logit, F):
    return F.npx.sigmoid(logit)