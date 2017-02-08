import mxnet as mx
a=mx.nd.zeros((10,10))
b =mx.nd.zeros((10,), dtype='int32')
mx.nd.batch_take(a=a,indices=b)
