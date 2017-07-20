# pylint: skip-file
import mxnet as mx
import numpy as np
import os, sys
import pickle as pickle
import logging
from common import get_data
from numpy import count_nonzero as nz

def equal(a, b):
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

@mx.init.register
class CustomInit(mx.init.Initializer):
    def __init__(self):
        super(CustomInit, self).__init__()
    def _init_weight(self, _, arr):
        for i in range(10):
            arr[i][:] = 0.0001
        for i in range(10, 20):
            arr[i][:] = -0.0001
        for i in range(20, 30):
            arr[i][:] = 0.0007
        for i in range(30, 40):
            arr[i][:] = -0.0007
        for i in range(40, 50):
            arr[i][:] = 2
        for i in range(50,60):
            arr[i][:] = -2

def accuracy(label, pred):
    py = np.argmax(pred, axis=1)
    return np.sum(py == label) / float(label.size)

# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=30)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
w2 = mx.symbol.Variable(name = 'w2', shape = (60,30), init = CustomInit())
fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=60, weight = w2)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(fc3, name='sm')

num_epoch = 2
prefix = './mlp'

# get data
get_data.GetMNIST_ubyte()

train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        data_shape=(784,),
        label_name='sm_label',
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        data_shape=(784,),
        label_name='sm_label',
        batch_size=batch_size, shuffle=True, flat=True, silent=False)

def test_mlp():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    # fit model
    model = mx.mod.Module(
        softmax,
        context=[mx.cpu(i) for i in range(2)],
        data_names=['data'],
        label_names=['sm_label'])
    optimizer_params = {
        'learning_rate'             : 0.1,
        'wd'                        : 0.004,
        'momentum'                  : 0.9,
        'do_pruning'                : True,
        'pruning_switch_epoch'      : [1,1],
        'weight_sparsity_threshold' : [0.0005,0.1],
        'bias_sparsity_threshold'   : [0,0],
        'batches_per_epoch'         : 600}
    model.fit(train_dataiter,
        eval_data=val_dataiter,
        eval_metric=mx.metric.np(accuracy),
        epoch_end_callback=mx.callback.do_checkpoint(prefix),
        num_epoch=num_epoch,
        optimizer_params=optimizer_params)
    logging.info('Finish traning...')

    # check pruning
    logging.info('Check pruning...')
    matrices = [np.ones((60,30)), np.ones((60,30))]
    matrices[0][0:20,:] = 0
    matrices[1][0:40,:] = 0
    for i in range(1, num_epoch + 1):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, i)
        matrix = arg_params['w2'].asnumpy()
        idx = i - 1
        assert equal(np.nonzero(matrix), np.nonzero(matrices[idx]))

    # remove files
    for i in range(num_epoch):
        os.remove('%s-%04d.params' % (prefix, i + 1))
    os.remove('%s-symbol.json' % prefix)


if __name__ == "__main__":
    test_mlp()
