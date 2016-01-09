import mxnet as mx
x = mx.th.randn(2, 2, ctx=mx.cpu(0))
print x.asnumpy()
y = mx.th.abs(x)
print y.asnumpy()

x = mx.th.randn(2, 2, ctx=mx.cpu(0))
print x.asnumpy()
mx.th.abs(x, x) # in-place
print x.asnumpy()