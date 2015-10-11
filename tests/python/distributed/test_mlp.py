#!/usr/bin/env python
# pylint: skip-file

import mxnet as mx
import numpy as np
import os, sys
import pickle as pickle
import logging
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../common/'))
import models
import get_data

# symbol net
batch_size = 100
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128)
act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64)
act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=10)
softmax = mx.symbol.Softmax(fc3, name = 'sm')

def accuracy(label, pred):
    py = np.argmax(pred, axis=1)
    return np.sum(py == label) / float(label.size)

num_round = 4
prefix = './mlp'

kv = mx.kvstore.create('dist')
batch_size /= kv.get_num_workers()

#check data
get_data.GetMNIST_ubyte()

train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        data_shape=(784,), num_parts=kv.get_num_workers(), part_index=kv.get_rank(),
        batch_size=batch_size, shuffle=True, flat=True, silent=False, seed=10)
val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        data_shape=(784,),
        batch_size=batch_size, shuffle=True, flat=True, silent=False)

def test_mlp():
    logging.basicConfig(level=logging.DEBUG)

    model = mx.model.FeedForward.create(
        softmax,
        X=train_dataiter,
        eval_data=val_dataiter,
        eval_metric=mx.metric.np(accuracy),
        ctx=[mx.cpu(i) for i in range(1)],
        num_round=num_round,
        learning_rate=0.05, wd=0.0004,
        momentum=0.9,
        kvstore=kv,
        )
    logging.info('Finish traning...')
    prob = model.predict(val_dataiter)
    logging.info('Finish predict...')
    val_dataiter.reset()
    y = np.concatenate([label.asnumpy() for _, label in val_dataiter]).astype('int')
    py = np.argmax(prob, axis=1)
    acc = float(np.sum(py == y)) / len(y)
    logging.info('final accuracy = %f', acc)
    assert(acc > 0.93)

if __name__ == "__main__":
    test_mlp()
