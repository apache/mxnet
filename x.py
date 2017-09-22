from mxnet.test_utils import *

def test_quadratic_function():
    def f(x, a, b, c):
        return a * x**2 + b * x + c

    a = np.random.random_sample()
    b = np.random.random_sample()
    c = np.random.random_sample()
    # check forward
    for ndim in range(1, 6):
        shape = rand_shape_nd(ndim, 5)
        data = rand_ndarray(shape=shape, stype='default')
        data_np = data.asnumpy()
        expected = f(data_np, a, b, c)
        output = mx.nd.quadratic(data, a=a, b=b, c=c)
        assert_almost_equal(output.asnumpy(), expected)

        # check backward using finite difference
        data = mx.sym.Variable('data')
        quad_sym = mx.sym.quadratic(data=data, a=a, b=b, c=c)
        check_numeric_gradient(quad_sym, [data_np])
        assert False

if __name__ == '__main__':
    test_quadratic_function()
