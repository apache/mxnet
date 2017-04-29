import mxnet.ndarray as nd
from mxnet.contrib.autograd import grad, grad_and_loss, train, test
from mxnet.test_utils import *

def autograd_assert(*args, **kwargs):
    func   = kwargs["func"]
    grad_f = kwargs["grad_func"]
    argnum = kwargs["argnum"] if 'argnum' in kwargs else None

    grad_func = grad_and_loss(func, argnum)
    grad_vals, output = grad_func(*args)
    res = func(*args)
    assert same(output.asnumpy(), res.asnumpy())
    grad_res = grad_f(*args)
    assert len(grad_vals) == len(grad_res)
    for a, b in zip(grad_vals, grad_res):
        assert same(a.asnumpy(), b.asnumpy())

def test_unary_func():
    x = nd.uniform(shape=(4, 5))
    f_exp         = lambda x: nd.exp(x)
    f_exp_grad    = lambda x: [nd.exp(x)]
    autograd_assert(x, func=f_exp, grad_func=f_exp_grad)
    f_half        = lambda x: x/2
    f_half_grad   = lambda x: [nd.ones(x.shape) * 0.5]
    autograd_assert(x, func=f_half, grad_func=f_half_grad)
    f_square      = lambda x: x**2
    f_square_grad = lambda x: [2*x]
    autograd_assert(x, func=f_square, grad_func=f_square_grad)

def test_binary_func():
    x = nd.uniform(shape=(4, 5))
    y = nd.uniform(shape=(4, 5))
    f_add      = lambda x, y: x+y
    f_add_grad = lambda x, y: [nd.ones(x.shape), nd.ones(y.shape)]
    autograd_assert(x, y, func=f_add, grad_func=f_add_grad)
    f_mul      = lambda x, y: x*y
    f_mul_grad = lambda x, y: [y, x]
    autograd_assert(x, y, func=f_mul, grad_func=f_mul_grad)
    f_compose  = lambda x, y: x+x*y
    f_compose_grad = lambda x, y: [nd.ones(x.shape) + y, x]
    autograd_assert(x, y, func=f_compose, grad_func=f_compose_grad)

def test_operator_with_state():
    def f_fc(a, b, weight, bias):
        x = a*b
        fc = nd.FullyConnected(
            x, weight, bias, num_hidden=32)
        return fc

    a = nd.uniform(shape=(64, 50))
    b = nd.uniform(shape=(64, 50))
    weight = nd.uniform(shape=(32, 50))
    bias = nd.uniform(shape=(32, ))

    grad_func = grad_and_loss(f_fc)
    grad_vals, outputs = grad_func(a, b, weight, bias)
    # (TODO) assert

def test_argnum():
    def f_with_mode(a, b, mode):
        if mode:
            return a+b
        else:
            return a*b

    a = nd.uniform(shape=(3, 2))
    b = nd.uniform(shape=(3, 2))
    f_add_grad = lambda x, y, mode: [nd.ones(x.shape), nd.ones(y.shape)]
    f_mul_grad = lambda x, y, mode: [y, x]
    autograd_assert(a, b, True,
        argnum=[0, 1], func=f_with_mode, grad_func=f_add_grad)
    autograd_assert(a, b, False,
        argnum=[0, 1], func=f_with_mode, grad_func=f_mul_grad)

def test_training():
    x = nd.ones((10, 10))
    with train():
        y = nd.Dropout(x, p=0.5)
        assert not (y.asnumpy() == x.asnumpy()).all()
        with test():
            y = nd.Dropout(x, p=0.5)
            assert (y.asnumpy() == x.asnumpy()).all()


if __name__ == "__main__":
    test_training()
    test_unary_func()
    test_binary_func()
    test_operator_with_state()
    test_argnum()
