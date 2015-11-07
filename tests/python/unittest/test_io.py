# pylint: skip-file
import mxnet as mx
import numpy as np
import os, gzip
import pickle as pickle
import time
import sys
from common import get_data

def test_MNISTIter():
    # prepare data
    get_data.GetMNIST_ubyte()

    batch_size = 100
    train_dataiter = mx.io.MNISTIter(
            image="data/train-images-idx3-ubyte",
            label="data/train-labels-idx1-ubyte",
            data_shape=(784,),
            batch_size=batch_size, shuffle=1, flat=1, silent=0, seed=10)
    # test_loop
    nbatch = 60000 / batch_size
    batch_count = 0
    for batch in train_dataiter:
        batch_count += 1
    assert(nbatch == batch_count)
    # test_reset
    train_dataiter.reset()
    train_dataiter.iter_next()
    label_0 = train_dataiter.getlabel().asnumpy().flatten()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.reset()
    train_dataiter.iter_next()
    label_1 = train_dataiter.getlabel().asnumpy().flatten()
    assert(sum(label_0 - label_1) == 0)

def test_Cifar10Rec():
    # skip-this test for saving time
    return
    get_data.GetCifar10()
    dataiter = mx.io.ImageRecordIter(
            path_imgrec="data/cifar/train.rec",
            mean_img="data/cifar/cifar10_mean.bin",
            rand_crop=False,
            and_mirror=False,
            shuffle=False,
            data_shape=(3,28,28),
            batch_size=100,
            preprocess_threads=4,
            prefetch_buffer=1)
    labelcount = [0 for i in range(10)]
    batchcount = 0
    for batch in dataiter:
        npdata = batch.data[0].asnumpy().flatten().sum()
        sys.stdout.flush()
        batchcount += 1
        nplabel = batch.label[0].asnumpy()
        for i in range(nplabel.shape[0]):
            labelcount[int(nplabel[i])] += 1
    for i in range(10):
        assert(labelcount[i] == 5000)

def test_NDArrayIter():
    datas = np.ones([1000, 2, 2])
    labels = np.ones([1000, 1])
    for i in range(1000):
        datas[i] = i / 100
        labels[i] = i / 100
    dataiter = mx.io.NDArrayIter(datas, labels, 128, True, last_batch_handle='pad')
    batchidx = 0
    for batch in dataiter:
        batchidx += 1
    assert(batchidx == 8)
    dataiter = mx.io.NDArrayIter(datas, labels, 128, False, last_batch_handle='pad')
    batchidx = 0
    labelcount = [0 for i in range(10)]
    for batch in dataiter:
        label = batch.label[0].asnumpy().flatten()
        assert((batch.data[0].asnumpy()[:,0,0] == label).all())
        for i in range(label.shape[0]):
            labelcount[int(label[i])] += 1

    for i in range(10):
        if i == 0:
            assert(labelcount[i] == 124)
        else:
            assert(labelcount[i] == 100)

if __name__ == "__main__":
    test_NDArrayIter()
    test_MNISTIter()
    test_Cifar10Rec()
