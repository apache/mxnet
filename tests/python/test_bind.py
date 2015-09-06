import numpy as np
import mxnet as mx


def reldiff(a, b):
    diff = np.sum(np.abs(a - b))
    norm = np.sum(np.abs(a))
    reldiff = diff  / norm
    return reldiff


def check_bind_with_uniform(uf, gf, dim):
    """check function consistency with uniform random numbers"""
    shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
    lhs = mx.symbol.Variable('lhs')
    rhs = mx.symbol.Variable('rhs')
    ret = uf(lhs, rhs)
    assert ret.list_arguments() == ['lhs', 'rhs']
    lhs_arr = mx.narray.create(shape)
    rhs_arr = mx.narray.create(shape)
    lhs_grad = mx.narray.create(shape)
    rhs_grad = mx.narray.create(shape)
    lhs_arr.numpy[:] = np.random.uniform(-10, 10, shape)
    rhs_arr.numpy[:] = np.random.uniform(-10, 10, shape)

    executor = ret.bind(mx.Context('cpu'),
                        args=[lhs_arr, rhs_arr],
                        args_grad=[lhs_grad, rhs_grad],
                        reqs=['write_to'] * 2)
    executor.forward()
    out2 = executor.heads()[0].numpy
    out1 = uf(lhs_arr.numpy, rhs_arr.numpy)
    assert reldiff(out1, out2) < 1e-6
    # test gradient
    out_grad = mx.narray.create(shape)
    out_grad.numpy[:] = np.ones(shape)
    lhs_grad2, rhs_grad2 = gf(out_grad.numpy,
                              lhs_arr.numpy,
                              rhs_arr.numpy)
    executor.backward([out_grad])
    assert reldiff(lhs_grad.numpy, lhs_grad2) < 1e-6
    assert reldiff(rhs_grad.numpy, rhs_grad2) < 1e-6


def test_bind():
    np.random.seed(0)
    nrepeat = 10
    maxdim = 4
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            check_bind_with_uniform(lambda x, y: x + y,
                                    lambda g, x, y: (g, g),
                                    dim)
            check_bind_with_uniform(lambda x, y: x - y,
                                    lambda g, x, y: (g, -g),
                                    dim)
            check_bind_with_uniform(lambda x, y: x * y,
                                    lambda g, x, y: (y * g, x * g),
                                    dim)
            check_bind_with_uniform(lambda x, y: x / y,
                                    lambda g, x, y: (g / y, -x * g/ (y**2)),
                                    dim)


