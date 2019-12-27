from mxnet import np, npx

def kl_divergence(lhs, rhs):
    return lhs._kl(rhs)
