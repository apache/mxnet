import mxnet as mx
import numpy as np


def make_tensor_3d(eigvals, factors_T):
    order = len(factors_T)
    k = len(eigvals)

    tensor_shape = [factors_T[id_mode].shape[1] for id_mode in range(order)]
    ts = np.zeros(shape=tensor_shape)

    for i in range(k):
        ts += eigvals[i] * np.einsum('i,j,k->ijk', factors_T[0][i, :],
                                     factors_T[1][i, :], factors_T[2][i, :])
    return ts

def tensor_factors_diff(eigvals0, factors_T0, eigvals, factors_T, k):
    def diff_vectors(v1, v2):
        return np.min([np.linalg.norm(v1 - v2), np.linalg.norm(v1 + v2)])

    def diff_matrices(m1, m2):
        d = np.sum([diff_vectors(m1[i, :], m2[i, :]) for i in range(len(m1))])
        return d

    d = diff_vectors(eigvals0[:k], eigvals)
    for id_mode in range(len(factors_T)):
        d += diff_matrices(factors_T0[id_mode][:k, :], factors_T[id_mode])

    return d

def test_cp_decomp_3d():
    eigvals0 = np.array([10, 6, 1])
    factors_T0 = [
        np.array([-0.43467446, -0.5572915, -0.15002647, -0.52690359,
                  -0.44760359, 0.27817005, -0.35825313, -0.55669702,
                  0.60305847, -0.34739751, -0.08818202,  0.63349627,
                  -0.69893409, -0.2969217 , -0.11931069]).reshape((3, 5)),
        np.array([-0.38177064, -0.22495485, -0.67352004, -0.59162256,
                  0.5703576, -0.64595109,  0.26986906, -0.42966275,
                  0.3282279, 0.72630226, 0.09523822,
                  -0.59639011]).reshape((3, 4)),
        np.array([-0.66722764, -0.52088417, -0.53243494,
                  0.63742185, -0.02948216, -0.76995077,
                  0.38535783, -0.8531181, 0.35169427]).reshape((3, 3))
    ]

    ts = make_tensor_3d(eigvals0, factors_T0)
    k = 2

    t = mx.sym.Variable('t')
    r = mx.operator.symbol.CPDecomp3D(data=t, k=k)
    
    r_bound = r.bind(mx.cpu(), {'t': mx.nd.array(ts, dtype='float64')})
    cp_decomp_results = r_bound.forward()

    eigvals = cp_decomp_results[0].asnumpy()
    factors_T = [mat.asnumpy() for mat in cp_decomp_results[1:]]

    print(eigvals)
    for i in range(3):
        print(factors_T[i])

    delta = tensor_factors_diff(eigvals0, factors_T0, eigvals, factors_T, k)
    assert delta <= 1e-6

