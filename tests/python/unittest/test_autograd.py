import mxnet.ndarray as nd
from mxnet.autograd import grad, grad_and_loss

def f_exp(x):
    return nd.exp(x)

def f_half(x):
    return x/2

def f_square(x):
    return x**2

def f_add(x, y):
    return x+y

def f_mul(x, y):
    return x*y

def f_composition(x, y):
    return x+x*y

def test_unary_func_grad(f):
    inputs = [[1, 2, 3],
              [4, 5, 6]]
    grad_func = grad_and_loss(f)
    grad_vals, outputs = grad_func(inputs)
    print(outputs.asnumpy())
    print(grad_vals.asnumpy())

def test_binary_func_grad(f):
    x = [[1, 2, 3],
         [4, 5, 6]]
    y = [[1, 2, 3],
         [4, 5, 6]]
    grad_func = grad_and_loss(f)
    grad_vals, outputs = grad_func(x, y)
    print(outputs.asnumpy())
    for grad_val in grad_vals:
        print(grad_val.asnumpy())

def test_operator_with_state():
    def f_fc(x, weight, bias):
        out = nd.FullyConnected(
            x, weight, bias, num_hidden=32)
        return out

    a = nd.uniform(shape=(64, 50))
    b = nd.uniform(shape=(64, 50))
    x = a+b
    weight = nd.uniform(shape=(32, 50))
    bias = nd.uniform(shape=(32, ))

    grad_func = grad_and_loss(f_fc)
    grad_vals, outputs = grad_func(x, weight, bias)


test_unary_func_grad(f_exp)
test_unary_func_grad(f_half)
test_unary_func_grad(f_square)
test_binary_func_grad(f_add)
test_binary_func_grad(f_mul)
test_binary_func_grad(f_composition)
test_operator_with_state()
