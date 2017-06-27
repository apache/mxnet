''' Test CPDecomp3D Op

'''

import mxnet as mx
import numpy as np


def make_tensor_3d(eigvals, factors_t):
    ''' Make a rank-k tensor for CPDecomp '''
    order = len(factors_t)
    k = len(eigvals)

    tensor_shape = [factors_t[id_mode].shape[1] for id_mode in range(order)]
    tensor = np.zeros(shape=tensor_shape)

    for i in range(k):
        tensor += eigvals[i] * np.einsum('i,j,k->ijk', factors_t[0][i, :],
                                         factors_t[1][i, :], factors_t[2][i, :])
    return tensor

def tensor_factors_diff(eigvals0, factors_t0, eigvals, factors_t, k):
    ''' Compute L2-norm of differences '''
    def diff_vectors(vec1, vec2):
        ''' Compute L2-norm of vec1 - vec2 '''
        return np.min([np.linalg.norm(vec1 - vec2),
                       np.linalg.norm(vec1 + vec2)])

    def diff_matrices(mat1, mat2):
        ''' Sum of L2-norm of differences between matrices, row by row '''
        row_diff = [diff_vectors(mat1[i, :], mat2[i, :])
                    for i in range(len(mat1))]
        return np.sum(row_diff)

    delta = diff_vectors(eigvals0[:k], eigvals)
    for id_mode, _ in enumerate(factors_t):
        delta += diff_matrices(factors_t0[id_mode][:k, :],
                               factors_t[id_mode])

    return delta

def test_cp_decomp_3d():
    ''' Test CPDecomp on a 3D tensor '''
    eigvals0 = np.array([10, 6, 1])
    factors_t0 = [
        np.array([-0.43467446, -0.5572915, -0.15002647, -0.52690359,
                  -0.44760359, 0.27817005, -0.35825313, -0.55669702,
                  0.60305847, -0.34739751, -0.08818202, 0.63349627,
                  -0.69893409, -0.2969217, -0.11931069]).reshape((3, 5)),
        np.array([-0.38177064, -0.22495485, -0.67352004, -0.59162256,
                  0.5703576, -0.64595109, 0.26986906, -0.42966275,
                  0.3282279, 0.72630226, 0.09523822,
                  -0.59639011]).reshape((3, 4)),
        np.array([-0.66722764, -0.52088417, -0.53243494,
                  0.63742185, -0.02948216, -0.76995077,
                  0.38535783, -0.8531181, 0.35169427]).reshape((3, 3))
    ]

    tensor = make_tensor_3d(eigvals0, factors_t0)

    for k in [1, 2, 3]:
        sym_t = mx.sym.Variable('t')
        sym_r = mx.operator.symbol.CPDecomp3D(data=sym_t, k=k)

        feed_dict = {'t': mx.nd.array(tensor, dtype='float64')}
        sym_r_bound = sym_r.bind(mx.cpu(), feed_dict)
        cp_decomp_results = sym_r_bound.forward()

        eigvals = cp_decomp_results[0].asnumpy()
        factors_t = [mat.asnumpy() for mat in cp_decomp_results[1:]]

        print(eigvals)
        for i in range(3):
            print(factors_t[i])

        delta = tensor_factors_diff(eigvals0, factors_t0,
                                    eigvals, factors_t, k)
        assert delta <= 1e-6
