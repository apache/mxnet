#!/usr/bin/env python
"""test if we can get identical data each time"""
from common import cifar10, mnist
import numpy as np

def get_XY(data_iter):
    data_iter.reset()
    Y = np.concatenate([y.asnumpy() for _, y in data_iter])
    data_iter.reset()
    X = np.concatenate([x.asnumpy() for x, _ in data_iter])
    assert X.shape[0] == Y.shape[0]
    return (X,Y)

def test_iter(data_iter):
    X, Y = get_XY(data_iter)
    print X.shape, Y.shape
    for i in range(4):
        A, B = get_XY(data_iter)
        assert(A.shape == X.shape)
        assert(B.shape == Y.shape)
        assert(np.sum(A != X) == 0)
        assert(np.sum(B != Y) == 0)


(train, val) = mnist(batch_size = 100, input_shape = (784,))
test_iter(train)
test_iter(val)

(train, val) = mnist(batch_size = 100, input_shape=(1,28,28))
test_iter(train)
test_iter(val)

(train, val) = cifar10(batch_size = 128, input_shape=(3,28,28))
test_iter(train)
test_iter(val)
