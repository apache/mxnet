# pylint: skip-file
import os, pickle, gzip
import logging
import sys
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../../../python'))
import mxnet as mx


def get_conv():
    data = mx.symbol.Variable('data')
    conv1= mx.symbol.Convolution(data = data, num_filter=32, kernel=(3,3), stride=(2,2))
    bn1 = mx.symbol.BatchNorm(data = conv1)
    act1 = mx.symbol.Activation(data = bn1, act_type="relu")
    mp1 = mx.symbol.Pooling(data = act1, kernel=(2,2), stride=(2,2), pool_type='max')

    conv2= mx.symbol.Convolution(data = mp1, num_filter=32, kernel=(3,3), stride=(2,2))
    bn2 = mx.symbol.BatchNorm(data = conv2)
    act2 = mx.symbol.Activation(data = bn2, act_type="relu")
    mp2 = mx.symbol.Pooling(data = act2, kernel=(2,2), stride=(2,2), pool_type='max')

    fl = mx.symbol.Flatten(data = mp2)
    fc2 = mx.symbol.FullyConnected(data = fl, num_hidden=10)
    softmax = mx.symbol.SoftmaxOutput(data = fc2, name='softmax')
    return softmax

def get_iter(data_dir):
    if data_dir == 'data':
        sys.path.append(os.path.join(curr_path, '../common/'))
        import get_data
        get_data.GetMNIST_ubyte()
    batch_size = 100
    train_dataiter = mx.io.MNISTIter(
        image = data_dir + "/train-images-idx3-ubyte",
        label = data_dir + "/train-labels-idx1-ubyte",
        data_shape=(1, 28, 28),
        batch_size=batch_size, shuffle=True, flat=False, silent=False)
    val_dataiter = mx.io.MNISTIter(
        image = data_dir + "/t10k-images-idx3-ubyte",
        label = data_dir + "/t10k-labels-idx1-ubyte",
        data_shape=(1, 28, 28),
        batch_size=batch_size, shuffle=True, flat=False, silent=False)
    return (train_dataiter, val_dataiter)

logging.basicConfig(level=logging.DEBUG)

num_gpus = 1
data_dir = 's3://dmcl/mnist'

(train, val) = get_iter(data_dir)

model = mx.model.FeedForward.create(
    symbol        = get_conv(),
    ctx           = [mx.gpu(i) for i in range(num_gpus)],
    X             = train,
    eval_data     = val,
    num_epoch     = 3,
    learning_rate = 0.1,
    wd            = 0.0001,
    momentum      = 0.9)
