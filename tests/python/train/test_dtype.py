# pylint: skip-file
import sys
sys.path.insert(0, '../../python')
import mxnet as mx
import numpy as np
import os, pickle, gzip
import logging
from common import get_data

batch_size = 128

# inception-bn-28-small start
# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu", mirror_attr={}):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type=act_type, attr=mirror_attr)
    return act

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3, mirror_attr):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1), mirror_attr=mirror_attr)
    # pool
    pool = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', attr=mirror_attr)
    # concat
    concat = mx.symbol.Concat(*[conv, pool])
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3, mirror_attr):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1, mirror_attr=mirror_attr)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3, mirror_attr=mirror_attr)
    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3])
    return concat

def get_net(num_classes = 10, force_mirroring=False):
    if force_mirroring:
        attr = {'force_mirroring': 'true'}
    else:
        attr = {}

    data = mx.symbol.Variable(name="data")
    # cast to float32 for uint8 input
    float_data = mx.symbol.Cast(data=data, dtype="float32")
    conv1 = ConvFactory(data=float_data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu", mirror_attr=attr)
    in3a = SimpleFactory(conv1, 32, 32, mirror_attr=attr)
    in3b = SimpleFactory(in3a, 32, 48, mirror_attr=attr)
    in3c = DownsampleFactory(in3b, 80, mirror_attr=attr)
    in4a = SimpleFactory(in3c, 112, 48, mirror_attr=attr)
    in4b = SimpleFactory(in4a, 96, 64, mirror_attr=attr)
    in4c = SimpleFactory(in4b, 80, 80, mirror_attr=attr)
    in4d = SimpleFactory(in4c, 48, 96, mirror_attr=attr)
    in4e = DownsampleFactory(in4d, 96, mirror_attr=attr)
    in5a = SimpleFactory(in4e, 176, 160, mirror_attr=attr)
    in5b = SimpleFactory(in5a, 176, 160, mirror_attr=attr)
    pool = mx.symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_pool", attr=attr)
    flatten = mx.symbol.Flatten(data=pool, name="flatten1", attr=attr)
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name="fc1")
    softmax = mx.symbol.SoftmaxOutput(data=fc, name="softmax")
    return softmax
# inception-bn-28-small end

# check data
get_data.GetCifar10()

def get_iterator_uint8(kv):
    data_shape = (3, 28, 28)

    train = mx.io.ImageRecordUInt8Iter(
        path_imgrec = "data/train.rec",
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)
    train = mx.io.PrefetchingIter(train)

    val = mx.io.ImageRecordUInt8Iter(
        path_imgrec = "data/test.rec",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

def get_iterator_float32(kv):
    data_shape = (3, 28, 28)

    train = mx.io.ImageRecordIter(
        path_imgrec = "data/train.rec",
        mean_img    = "data/mean.bin",
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)
    train = mx.io.PrefetchingIter(train)

    val = mx.io.ImageRecordIter(
        path_imgrec = "data/test.rec",
        mean_img    = "data/mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

num_epoch = 1

def test_cifar10(train, val, use_module=False):
    train.reset()
    val.reset()
    devs = [mx.gpu(0)]
    net = get_net()
    mod = mx.mod.Module(net, context=devs)
    optim_args = {'learning_rate': 0.05, 'wd': 0.00001, 'momentum': 0.9}
    eval_metrics = ['accuracy']
    if use_module:
        executor = mx.mod.Module(net, context=devs)
        executor.fit(
            train,
            eval_data=val,
            optimizer_params=optim_args,
            eval_metric=eval_metrics,
            num_epoch=num_epoch,
            arg_params=None,
            aux_params=None,
            begin_epoch=0,
            batch_end_callback=mx.callback.Speedometer(batch_size, 50),
            epoch_end_callback=None)
    else:
        executor = mx.model.FeedForward.create(
            net,
            train,
            ctx=devs,
            eval_data=val,
            eval_metric=eval_metrics,
            num_epoch=num_epoch,
            arg_params=None,
            aux_params=None,
            begin_epoch=0,
            batch_end_callback=mx.callback.Speedometer(batch_size, 50),
            epoch_end_callback=None,
            **optim_args)

    ret = executor.score(val, eval_metrics)
    if use_module:
        logging.info('final accuracy = %f', ret[0][1])
        assert (ret[0][1] > 0.4)
    else:
        logging.info('final accuracy = %f', ret[0])
        assert (ret[0] > 0.4)

if __name__ == "__main__":
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    kv = mx.kvstore.create("local")
    (train, val) = get_iterator_float32(kv)
    test_cifar10(train, val, use_module=False)
    test_cifar10(train, val, use_module=True)
    (train, val) = get_iterator_uint8(kv)
    test_cifar10(train, val, use_module=False)
    test_cifar10(train, val, use_module=True)
