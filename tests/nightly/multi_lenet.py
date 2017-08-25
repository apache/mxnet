#!/usr/bin/env python
# lenet with multiple gpus
#
# using different kvstore will get almost identical results
#
# must disable CUDNN, which results randomness
#
# where the gradients are aggregated and the weights are updated results
# different results, it might be gpu's precision is much lower than
# cpu. however, the results should be still identical if not too many iterations
# are performed, which can be controlled by either increasing the batch size or
# decreasing the number of epochs

import os, sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../../example/image-classification"))
sys.path.append(os.path.join(curr_path, "../../python"))
import mxnet as mx
import numpy as np
import logging
import logging
import train_mnist

# number of gpus
ngpus = 4
# the batch size
batch_size = 200

def mnist(batch_size, input_shape):
    """return mnist iters without randomness"""
    flat = len(input_shape)==1
    train = mx.io.MNISTIter(
        image      = "data/mnist/train-images-idx3-ubyte",
        label      = "data/mnist/train-labels-idx1-ubyte",
        data_shape = input_shape,
        batch_size = batch_size,
        shuffle    = False,
        flat       = flat,
        silent     = True)
    val = mx.io.MNISTIter(
        image      = "data/mnist/t10k-images-idx3-ubyte",
        label      = "data/mnist/t10k-labels-idx1-ubyte",
        data_shape = input_shape,
        batch_size = batch_size,
        shuffle    = False,
        flat       = flat,
        silent     = True)
    return (train, val)

def accuracy(model, data):
    """evaluate acc"""
    data.reset()
    prob = model.predict(data)
    py = np.argmax(prob, axis=1)
    # get label
    data.reset()
    y = np.concatenate([d.label[0].asnumpy() for d in data]).astype('int')
    y = y[0:len(py)]
    acc = float(np.sum(py == y)) / len(y)
    logging.info('Accuracy = %f', acc)
    return acc

def get_XY(data_iter):
    data_iter.reset()
    Y = np.concatenate([d.label[0].asnumpy() for d in data_iter])
    data_iter.reset()
    X = np.concatenate([d.data[0].asnumpy() for d in data_iter])
    assert X.shape[0] == Y.shape[0]
    return (X,Y)

def test_data(data_iter):
    # test whether we will get the identical data each time
    X, Y = get_XY(data_iter)
    print X.shape, Y.shape
    for i in range(4):
        A, B = get_XY(data_iter)
        assert(A.shape == X.shape)
        assert(B.shape == Y.shape)
        assert(np.sum(A != X) == 0)
        assert(np.sum(B != Y) == 0)

# must use the same net, since the system will automatically assign namees to
# layers, which affects the order of the weight initialization
net = train_mnist.get_lenet()

def test_lenet(devs, kv_type):
    logging.basicConfig(level=logging.DEBUG)
    (train, val) = mnist(batch_size = batch_size, input_shape=(1,28,28))
    # guarantee the same weight init for each run
    mx.random.seed(0)
    model = mx.model.FeedForward(
        ctx           = devs,
        symbol        = net,
        num_epoch     = 2,
        learning_rate = 0.1,
        momentum      = 0.9,
        wd            = 0.00001)
    model.fit(
        kvstore       = kv_type,
        X             = train)
    return accuracy(model, val)


if __name__ == "__main__":
    # test data
    (train, val) = mnist(batch_size = 100, input_shape=(1,28,28))
    test_data(train)
    test_data(val)

    (train, val) = mnist(batch_size = 110, input_shape=(784,))
    test_data(train)
    test_data(val)

    # test model
    gpus = [mx.gpu(i) for i in range(ngpus)]

    base = test_lenet(mx.gpu(), 'none')
    acc1 = test_lenet(mx.gpu(), 'none')
    acc2 = test_lenet(gpus, 'local_update_cpu')
    acc3 = test_lenet(gpus, 'local_allreduce_cpu')
    acc4 = test_lenet(gpus, 'local_allreduce_device')

    assert base > 0.95
    assert abs(base - acc1) < 1e-4
    assert abs(base - acc2) < 1e-3
    assert abs(base - acc3) < 1e-3
    assert abs(acc3 - acc4) < 1e-4
