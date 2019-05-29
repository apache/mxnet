import mxnet as mx
a = mx.nd.array([1, 2, 3, 4, 5], ctx=mx.cpu(0))
b = mx.nd.array([5, 4, 3, 2, 1], ctx=mx.cpu(0))
c = mx.nd.tvm_vector_add(a, b)
print("a =", a.asnumpy())
print("b =", b.asnumpy())
print("a + b =", c.asnumpy())