# pylint: skip-file
import sys
sys.path.insert(0, '../../python')
import mxnet as mx
import numpy as np
import os, pickle, gzip, argparse
import logging
from common import get_data

def get_model(use_gpu):
    # symbol net
    data = mx.symbol.Variable('data')
    conv1= mx.symbol.Convolution(data = data, name='conv1', num_filter=32, kernel=(3,3), stride=(2,2))
    bn1 = mx.symbol.BatchNorm(data = conv1, name="bn1")
    act1 = mx.symbol.Activation(data = bn1, name='relu1', act_type="relu")
    mp1 = mx.symbol.Pooling(data = act1, name = 'mp1', kernel=(2,2), stride=(2,2), pool_type='max')

    conv2= mx.symbol.Convolution(data = mp1, name='conv2', num_filter=32, kernel=(3,3), stride=(2,2))
    bn2 = mx.symbol.BatchNorm(data = conv2, name="bn2")
    act2 = mx.symbol.Activation(data = bn2, name='relu2', act_type="relu")
    mp2 = mx.symbol.Pooling(data = act2, name = 'mp2', kernel=(2,2), stride=(2,2), pool_type='max')


    fl = mx.symbol.Flatten(data = mp2, name="flatten")
    fc2 = mx.symbol.FullyConnected(data = fl, name='fc2', num_hidden=10)
    softmax = mx.symbol.SoftmaxOutput(data = fc2, name = 'sm')

    num_epoch = 1
    ctx=mx.gpu() if use_gpu else mx.cpu()
    model = mx.model.FeedForward(softmax, ctx,
                                 num_epoch=num_epoch,
                                 learning_rate=0.1, wd=0.0001,
                                 momentum=0.9)
    return model

def get_iters():
    # check data
    get_data.GetMNIST_ubyte()

    batch_size = 100
    train_dataiter = mx.io.MNISTIter(
            image="data/train-images-idx3-ubyte",
            label="data/train-labels-idx1-ubyte",
            data_shape=(1, 28, 28),
            label_name='sm_label',
            batch_size=batch_size, shuffle=True, flat=False, silent=False, seed=10)
    val_dataiter = mx.io.MNISTIter(
            image="data/t10k-images-idx3-ubyte",
            label="data/t10k-labels-idx1-ubyte",
            data_shape=(1, 28, 28),
            label_name='sm_label',
            batch_size=batch_size, shuffle=True, flat=False, silent=False)
    return  train_dataiter, val_dataiter

# run default with unit test framework
def test_mnist():
    iters = get_iters()
    exec_mnist(get_model(False), iters[0], iters[1])

def exec_mnist(model, train_dataiter, val_dataiter):
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    model.fit(X=train_dataiter,
              eval_data=val_dataiter)
    logging.info('Finish fit...')
    prob = model.predict(val_dataiter)
    logging.info('Finish predict...')
    val_dataiter.reset()
    y = np.concatenate([batch.label[0].asnumpy() for batch in val_dataiter]).astype('int')
    py = np.argmax(prob, axis=1)
    acc1 = float(np.sum(py == y)) / len(y)
    logging.info('final accuracy = %f', acc1)
    assert(acc1 > 0.94)

# run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='use gpu to train')
    args = parser.parse_args()
    iters = get_iters()
    exec_mnist(get_model(args.gpu), iters[0], iters[1])

