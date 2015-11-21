import mxnet as mx
from numpy.testing import assert_allclose

if __name__ == '__main__':
    x = mx.nd.zeros((10,), ctx=mx.gpu(0))
    x[:] = 1
    y = mx.nd.zeros((10,), ctx=mx.gpu(0))
    y[:] = 2
    rtc = mx.rtc('abc', [('x', x)], [('y', y)], """y[threadIdx.x] = x[threadIdx.x]*5.0;""")
    rtc.push([x], [y], (1, 1, 1), (10,1,1))
    assert_allclose(y.asnumpy(), x.asnumpy()*5.0)