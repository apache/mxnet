import os
import mxnet as mx
import numpy as np

def same(a, b):
    return np.all(a == b)

def check_with_device(device):
    with mx.Context(device):
        shape = (10,10, 10, 10)
        a = mx.nd.ones(shape, dtype=np.float32)
        c = mx.symbol.Variable('input')
        e = mx.symbol.DropoutAllchannels(data=c)
        be = e.bind(device, { 'input' : a })
        be.forward(True)
        ret = be.outputs[0].asnumpy()
        for i in range(shape[0]):
          for j in range(shape[1]-1):
            # Making sure all channels have been dropped out the same way
            assert(same(ret[i,j,:,:], ret[i,j+1,:,:]))

def test_dropout_allchannels():
    check_with_device(mx.cpu())

if __name__ == '__main__':
    test_dropout_allchannels()

