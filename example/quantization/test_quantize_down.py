import mxnet as mx
import mxnet.ndarray as nd

ctx = mx.gpu(0)
a = nd.array([ 5001,  6164, 264, 32255, 27232,  -18225, 2551,  3185,
              18162, 11226,   6,   600, 16793, 18225, 2987, 12637],
             dtype='int32', ctx=ctx)
min0 = nd.array([-38964.81640625], ctx=ctx)
max0 = nd.array([ 38964.81640625], ctx=ctx)

b, min1, max1 = nd.quantize_down_and_shrink_range(a, min0, max0)
print(a.asnumpy())
print(b.asnumpy())
