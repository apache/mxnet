# pylint: skip-file
""" common for multi-node
all iterators are disabled randomness
"""
import sys
sys.path.insert(0, "../common/")
sys.path.insert(0, "../../../python/")
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
        data_shape  = input_shape,
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
    softmax = mx.symbol.SoftmaxOutput(fc3, name = 'softmax')
    return softmax

def lenet():
    """return the symbol for lenect"""
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu"):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type=act_type)
    return act

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1))
    # pool
    pool = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max')
    # concat
    concat = mx.symbol.Concat(*[conv, pool])
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3)
    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3])
    return concat

def inception():
    data = mx.symbol.Variable(name="data")
    conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu")
    in3a = SimpleFactory(conv1, 32, 32)
    in3b = SimpleFactory(in3a, 32, 48)
    in3c = DownsampleFactory(in3b, 80)
    in4a = SimpleFactory(in3c, 112, 48)
    in4b = SimpleFactory(in4a, 96, 64)
    in4c = SimpleFactory(in4b, 80, 80)
    in4d = SimpleFactory(in4c, 48, 96)
    in4e = DownsampleFactory(in4d, 96)
    in5a = SimpleFactory(in4e, 176, 160)
    in5b = SimpleFactory(in5a, 176, 160)
    pool = mx.symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_pool")
    flatten = mx.symbol.Flatten(data=pool, name="flatten1")
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=10, name="fc1")
    softmax = mx.symbol.SoftmaxOutput(data=fc, name="softmax")
    return softmax
