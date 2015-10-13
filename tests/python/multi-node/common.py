# pylint: skip-file
""" common for multi-node

- all iterators are disabled randomness

"""
import sys
sys.path.insert(0, "../common/")
sys.path.insert(0, "../../python/")
import mxnet as mx
import get_data
import numpy as np
import logging

def mnist(batch_size, input_shape, num_parts=1, part_index=0):
    """return mnist iters"""
    get_data.GetMNIST_ubyte()
    flat = len(input_shape)==1
    train = mx.io.MNISTIter(
        image      = "data/train-images-idx3-ubyte",
        label      = "data/train-labels-idx1-ubyte",
        data_shape = input_shape,
        batch_size = batch_size,
        num_parts  = num_parts,
        part_index = part_index,
        shuffle    = False,
        flat       = flat,
        silent     = False)
    val = mx.io.MNISTIter(
        image      = "data/t10k-images-idx3-ubyte",
        label      = "data/t10k-labels-idx1-ubyte",
        data_shape = input_shape,
        batch_size = batch_size,
        shuffle    = False,
        flat       = flat,
        silent     = False)
    return (train, val)

def cifar10(batch_size, input_shape, num_parts=1, part_index=0):
    """return cifar10 iterator"""
    get_data.GetCifar10()

    train = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/train.rec",
        mean_img    = "data/cifar/cifar_mean.bin",
        data_shape  = input_shape,
        batch_size  = batch_size,
        rand_crop   = False,
        rand_mirror = False,
        shuffle     = False,
        round_batch = False,
        num_parts   = num_parts,
        part_index  = part_index)
    val = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/test.rec",
        mean_img    = "data/cifar/cifar_mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        shuffle     = False,
        round_batch = False,
        data_shape  = (3,28,28),
        batch_size  = batch_size)
    return (train, val)

def accuracy(model, data):
    """evaluate acc"""
    # predict
    data.reset()
    prob = model.predict(data)
    py = np.argmax(prob, axis=1)
    # get label
    data.reset()
    y = np.concatenate([label.asnumpy() for _, label in data]).astype('int')
    y = y[0:len(py)]
    acc = float(np.sum(py == y)) / len(y)
    logging.info('Accuracy = %f', acc)

    return acc

def mlp():
    """return symbol for mlp"""
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
    softmax = mx.symbol.Softmax(fc3, name = 'sm')
    return softmax
