# pylint: skip-file
import mxnet as mx
import numpy as np
import os, sys
import pickle as pickle
import logging
from common import get_data
from numpy import count_nonzero as nz

# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(fc3, name = 'sm')

def accuracy(label, pred):
    py = np.argmax(pred, axis=1)
    return np.sum(py == label) / float(label.size)

num_epoch = 7
prefix = './mlp'

#check data
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

    model = mx.model.FeedForward.create(
        softmax,
        X=train_dataiter,
        eval_data=val_dataiter,
        eval_metric=mx.metric.np(accuracy),
        epoch_end_callback=mx.callback.do_checkpoint(prefix),
        ctx=[mx.cpu(i) for i in range(2)],
        num_epoch=num_epoch,
        learning_rate=0.1, wd=0.0004,
        momentum=0.9,
        do_pruning=True,
        pruning_switch_epoch=[1,3,5,7],
        weight_sparsity=[0,25,50,75],
        bias_sparsity=[0,0,50,50],
        batches_per_epoch=600)

    logging.info('Finish traning...')

    # check pruning
    logging.info('Check pruning...')
    weight_percent = [1,0.75,0.75,0.5,0.5,0.25,0.25]
    bias_percent = [1,1,1,0.5,0.5,0.5,0.5]
    for i in range(1, num_epoch + 1):
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, i)
        weight_params = [arg_params['fc1_weight'], arg_params['fc2_weight'], arg_params['fc3_weight']]
        bias_params = [arg_params['fc1_bias'], arg_params['fc2_bias'], arg_params['fc3_bias']]
        idx = i - 1
        for param in weight_params:
            assert nz(param.asnumpy())/float(param.size) == weight_percent[idx]
        for param in bias_params:
            assert nz(param.asnumpy())/float(param.size) == bias_percent[idx]

    for i in range(num_epoch):
        os.remove('%s-%04d.params' % (prefix, i + 1))
    os.remove('%s-symbol.json' % prefix)


if __name__ == "__main__":
    test_mlp()
