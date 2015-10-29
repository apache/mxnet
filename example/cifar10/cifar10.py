# pylint: skip-file
import sys, os
# code to automatically download dataset
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../../tests/python/common"))
import get_data
import mxnet as mx
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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
softmax = mx.symbol.Softmax(data=fc, name="loss")

#########################################################

get_data.GetCifar10()
batch_size = 128
num_epoch = 10
num_gpus = 1

train_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/train.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=True,
        rand_mirror=True,
        data_shape=(3,28,28),
        batch_size=batch_size,
        preprocess_threads=1)
test_dataiter = mx.io.ImageRecordIter(
        path_imgrec="data/cifar/test.rec",
        mean_img="data/cifar/cifar_mean.bin",
        rand_crop=False,
        rand_mirror=False,
        data_shape=(3,28,28),
        batch_size=batch_size,
        preprocess_threads=1)

def test_cifar():
    logging.basicConfig(level=logging.DEBUG)
    gpus = [mx.gpu(i) for i in range(num_gpus)]
    model = mx.model.FeedForward(ctx=gpus, symbol=softmax, num_epoch=num_epoch,
                                 learning_rate=0.05, momentum=0.9, wd=0.0001,
                                 initializer=mx.init.Uniform(0.07))

    model.fit(X=train_dataiter, eval_data=test_dataiter,
              batch_end_callback=mx.callback.Speedometer(batch_size))

if __name__ == "__main__":
    test_cifar()
