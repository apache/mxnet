import mxnet as mx
import mxnet.ndarray as nd
import numpy as np

ctx = mx.gpu(0)
dtype = np.int8
n = 4

def test_lowbit_relu():
    a_ = np.random.uniform(low=-100, high=100, size=(n,n))
    a = nd.array(a_, ctx=ctx, dtype=dtype)
    b = nd.lowbit_relu(a)

def test_lowbit_max_pool():
    a_ = np.random.uniform(low=-128, high=127, size=(1, 1, n, n))
    a = nd.array(a_, ctx=ctx, dtype=dtype)
    b = nd.lowbit_max_pool(a, kernel=[2, 2])

def test_lowbit_lrn():
    n = 5
    x_ = np.random.uniform(low=-100, high=100, size=(1,1,n,n))
    x = nd.array(x_, ctx=ctx, dtype=dtype)
    y = nd.lowbit_lrn(x, nsize=3)

def test_lowbit_fully_connected():
    x_ = np.random.uniform(low=-100, high=100, size=(n,n))
    x = nd.array(x_, ctx=ctx, dtype=dtype)
    w = nd.array(x_, ctx=ctx, dtype=dtype)
    b_ = np.random.uniform(low=-100, high=100, size=(n,))
    b = nd.array(b_, ctx=ctx, dtype=dtype)
    c = nd.lowbit_fully_connected(x, w, b, num_hidden=n)

def test_lowbit_convolution():
    x_ = np.random.uniform(low=-100, high=100, size=(1, 1, 5, 5))
    k_ = np.random.uniform(low=-100, high=100, size=(1, 1, 3, 3))
    x = nd.array(x_, ctx=ctx, dtype=dtype)
    k = nd.array(k_, ctx=ctx, dtype=dtype)
    y = nd.lowbit_convolution(x, k, num_filter=1,
            kernel=[3, 3], stride=[1, 1], pad=[1, 1])

if __name__ == "__main__":
    test_lowbit_relu()
    test_lowbit_max_pool()
    test_lowbit_lrn()
