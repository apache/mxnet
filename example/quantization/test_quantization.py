import mxnet as mx
import mxnet.ndarray as nd
from mxnet.test_utils import *
import numpy as np

ctx = mx.gpu(0)
dtype = np.int8

# TODO
# make fully connected support bias
# test_quantized_lrn

def test_quantize():
    pass


def test_quantize1():
    min0 = nd.array([-1.0], ctx=ctx)
    max0 = nd.array([1.0], ctx=ctx)
    a_ = nd.array([-1.0, -0.9999, -0.5, -0.0001, 0, 0.0001, 0.5, 0.9999, 1.0], ctx=ctx)
    qa, min1, max1 = mx.contrib.nd.quantize(a_, min0, max0)
    a  = nd.array([-127, -127, -64, 0, 0, 0, 64, 127, 127], ctx=ctx)
    assert same(a.asnumpy(), qa.asnumpy())


def test_dequantize():
    pass


def test_dequantize1():
    N = 5
    min_range = 1.0
    max_range = 1.0
    min0 = nd.array([min_range], ctx=ctx)
    max0 = nd.array([max_range], ctx=ctx)
    a_ = nd.array([-128, -127, -64, -63, 1, 0, 1, 63, 64, 127], dtype=dtype, ctx=ctx)
    da = mx.contrib.nd.dequantize(a_, min0, max0)
    a = a_.asnumpy() * (max(abs(min_range), abs(max_range)) / 127)
    same(da.asnumpy(), a)


def test_quantized_fully_connected():
    M = 12
    N = 16
    K = 4

    min_range = -0.5
    max_range = 1.0
    x_ = nd.array(np.random.uniform(size=(M, N)), ctx=ctx)
    min0x = nd.array([min_range], ctx=ctx)
    max0x = nd.array([max_range], ctx=ctx)
    qx, min1x, max1x = mx.contrib.nd.quantize(x_, min0x, max0x)
    w_ = nd.array(np.random.uniform(size=(K, N)), ctx=ctx)
    min0w = nd.array([min_range], ctx=ctx)
    max0w = nd.array([max_range], ctx=ctx)
    qw, min1w, max1w = mx.contrib.nd.quantize(w_, min0w, max0w)
    y_, min1, max1 = nd.quantized_fully_connected(qx, qw, min1x, max1x, min1w, max1w,
        num_hidden=K, no_bias=True)

    x = qx.astype(np.float32)
    w = qw.astype(np.float32)
    y = nd.FullyConnected(x, w, num_hidden=K, no_bias=True)
    assert same(y.asnumpy(), y_.asnumpy())


def test_quantized_flatten():
    BATCH = 32
    IN  = 4
    OUT = 4
    XN = 32

    min_range = -0.5
    max_range = 1.0
    x_ = nd.array(np.random.uniform(size=(BATCH, IN, XN, XN)), ctx=ctx)
    min0 = nd.array([min_range], ctx=ctx)
    max0 = nd.array([max_range], ctx=ctx)
    qx, min1, max1 = mx.contrib.nd.quantize(x_, min0, max0)
    y_, min2, max2 = nd.quantized_flatten(qx, min1, max1)

    x = qx.astype(np.float32)
    y = nd.flatten(x)
    assert same(y.asnumpy(), y_.asnumpy())


def test_quantized_conv2d_NCHW():
    BATCH = 1
    IN  = 4
    OUT = 4
    XN = 5
    KN = 3

    min_range = -0.5
    max_range = 1.0
    x_ = nd.array(np.random.uniform(size=(BATCH, IN, XN, XN)), ctx=ctx)
    min0x = nd.array([min_range], ctx=ctx)
    max0x = nd.array([max_range], ctx=ctx)
    qx, min1x, max1x = mx.contrib.nd.quantize(x_, min0x, max0x)
    k_ = nd.array(np.random.uniform(size=(OUT, IN, KN, KN)), ctx=ctx)
    min0k = nd.array([min_range], ctx=ctx)
    max0k = nd.array([max_range], ctx=ctx)
    qk, min1k, max1k = mx.contrib.nd.quantize(k_, min0k, max0k)
    y_, min1, max1 = nd.quantized_conv2d(qx, qk, min1x, max1x, min1k, max1k,
        kernel=(KN, KN), num_filter=OUT, no_bias=True)

    x = qx.astype(np.float32)
    k = qk.astype(np.float32)
    y = nd.Convolution(x, k, kernel=(KN, KN), num_filter=OUT, no_bias=True)
    assert same(y.asnumpy(), y_.asnumpy())


def test_quantized_conv2d_NHWC():
    BATCH = 1
    IN  = 4
    OUT = 4
    XN = 5
    KN = 3

    min_range = -1.0
    max_range = 0.5
    x_ = nd.uniform(shape=(BATCH, XN, XN, IN), low=min_range, high=max_range, ctx=ctx)
    min0x = nd.array([min_range], ctx=ctx)
    max0x = nd.array([max_range], ctx=ctx)
    qx, min1x, max1x = mx.contrib.nd.quantize(x_, min0x, max0x)
    k_ = nd.uniform(shape=(OUT, KN, KN, IN), low=min_range, high=max_range, ctx=ctx)
    min0k = nd.array([min_range], ctx=ctx)
    max0k = nd.array([max_range], ctx=ctx)
    qk, min1k, max1k = mx.contrib.nd.quantize(k_, min0k, max0k)
    y_, min1, max1 = nd.quantized_conv2d(qx, qk, min1x, max1x, min1k, max1k,
        kernel=(KN, KN), num_filter=OUT, no_bias=True, layout='NHWC')

    x = qx.astype(np.float32)
    k = qk.astype(np.float32)
    y = nd.Convolution(x, k, kernel=(KN, KN), num_filter=OUT, no_bias=True, layout='NHWC')
    assert same(y.asnumpy(), y_.asnumpy())


def test_quantized_relu():
    N = 13
    min_range = -1.0
    max_range = 0.5
    x_ = nd.uniform(shape=(N, N), low=min_range, high=max_range, ctx=ctx)
    min0x = nd.array([min_range], ctx=ctx)
    max0x = nd.array([max_range], ctx=ctx)
    qx, min1x, max1x = mx.contrib.nd.quantize(x_, min0x, max0x)
    y_, min1, max1 = nd.quantized_relu(qx, min1x, max1x)

    x = qx.astype(np.float32)
    y = nd.relu(x)
    assert same(y.asnumpy(), y_.asnumpy())


def test_quantized_max_pool():
    BATCH = 1
    IN = 3
    N = 28
    K = 4
    min_range = -1.0
    max_range = 0.5
    x_ = nd.uniform(shape=(BATCH, IN, N, N), low=min_range, high=max_range, ctx=ctx)
    min0 = nd.array([min_range], ctx=ctx)
    max0 = nd.array([max_range], ctx=ctx)
    qx, min1, max1 = mx.contrib.nd.quantize(x_, min0, max0)
    y_, min2, max2 = nd.quantized_max_pool(qx, min1, max1, kernel=[K, K])

    x = qx.astype(np.float32)
    y = nd.Pooling(x, kernel=(K, K), pool_type='max')
    assert same(y.asnumpy(), y_.asnumpy())


def test_quantized_lrn():
    BATCH = 1
    IN = 1
    N = 5
    min_range = -1.0
    max_range = 0.5
    x_ = nd.uniform(shape=(BATCH, IN, N, N), low=min_range, high=max_range, ctx=ctx)
    min0 = nd.array([min_range], ctx=ctx)
    max0 = nd.array([max_range], ctx=ctx)
    qx, min1, max1 = mx.contrib.nd.quantize(x_, min0, max0)
    y_, min2, max2 = nd.quantized_lrn(qx, min1, max1, nsize=3)

    x = qx.astype(np.float32)
    y = nd.LRN(x, nsize=3)
    assert same(y.asnumpy(), y_.asnumpy())


def debug_quantize_dequantize():
    N = 8
    a = nd.uniform(low=-0.1, high=0.5, shape=(N,N), ctx=ctx)
    min0 = nd.min(a)
    max0 = nd.max(a)
    qa, min1, max1 = mx.contrib.nd.quantize(a, min0, max0)
    a_ = mx.contrib.nd.dequantize(qa, min1, max1)

    min_ = min0.asnumpy()[0]
    max_ = max0.asnumpy()[0]
    print( a.asnumpy())
    print(qa.asnumpy())
    print(a_.asnumpy())
    print('rate: {}'.format(max(abs(max_), abs(min_))/128))


def test_quantized_down_and_shrink_range_fully_connected():
    M = 2
    N = 4
    K = 3
    min_range = -1.0
    max_range =  1.0
    a_ = nd.uniform(low=min_range, high=max_range, shape=(M, N), dtype=np.float32, ctx=ctx)
    b_ = nd.uniform(low=min_range, high=max_range, shape=(K, N), dtype=np.float32, ctx=ctx)
    c_ = nd.FullyConnected(a_, b_, num_hidden=K, no_bias=True)
    print('a_:\n{}'.format(a_.asnumpy()))
    print('b_:\n{}'.format(b_.asnumpy()))
    print('c_:\n{}'.format(c_.asnumpy()))

    min0a = nd.array([min_range], ctx=ctx)
    max0a = nd.array([max_range], ctx=ctx)
    qa, min1a, max1a = mx.contrib.nd.quantize(a_, min0a, max0a)
    min0b = nd.array([min_range], ctx=ctx)
    max0b = nd.array([max_range], ctx=ctx)
    qb, min1b, max1b = mx.contrib.nd.quantize(b_, min0b, max0b)
    print('qa:\n{}'.format(qa.asnumpy()))
    print('min1a:\n{}'.format(min1a.asnumpy()))
    print('max1a:\n{}'.format(max1a.asnumpy()))
    print('qb:\n{}'.format(qb.asnumpy()))
    print('min1b:\n{}'.format(min1b.asnumpy()))
    print('max1b:\n{}'.format(max1b.asnumpy()))

    qc_, min2, max2 = nd.quantized_fully_connected(qa, qb, min1a, max1a, min1b, max1b,
        num_hidden=K, no_bias=True)
    print('qc_:\n{}'.format(qc_.asnumpy()))
    print('min2:\n{}'.format(min2.asnumpy()))
    print('max2:\n{}'.format(max2.asnumpy()))

    qc, min3, max3 = nd.quantize_down_and_shrink_range(qc_, min2, max2)
    print('qc:\n{}'.format(qc.asnumpy()))
    print('min3:\n{}'.format(min3.asnumpy()))
    print('max3:\n{}'.format(max3.asnumpy()))
    c = mx.contrib.nd.dequantize(qc, min3, max3)
    print('c:\n{}'.format(c.asnumpy()))


# if __name__ == "__main__":
#     test_quantized_relu()
#     test_quantized_max_pool()
#     test_quantized_conv2d()
