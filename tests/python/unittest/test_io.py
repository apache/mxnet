# pylint: skip-file
import mxnet as mx
import numpy as np
import os, gzip
import pickle as pickle
import time
import sys
from common import get_data
from PIL import Image


def test_MNISTIter():
    # prepare data
    get_data.GetMNIST_ubyte()

    batch_size = 100
    train_dataiter = mx.io.MNISTIter(
            image="data/train-images-idx3-ubyte",
            label="data/train-labels-idx1-ubyte",
            input_shape=(784,),
            batch_size=batch_size, shuffle=1, flat=1, silent=0, seed=10)
    # test_loop
    nbatch = 60000 / batch_size
    batch_count = 0
    for data, label in train_dataiter:
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
    get_data.GetCifar10()
    dataiter = mx.io.ImageRecordIter(
            path_imgrec="data/cifar/train.rec",
            mean_img="data/cifar/cifar10_mean.bin",
            rand_crop=False,
            and_mirror=False,
            shuffle=False,
            input_shape=(3,28,28),
            batch_size=100,
            nthread=4,
            prefetch_capacity=1)
    labelcount = [0 for i in range(10)] 
    batchcount = 0
    for data, label in dataiter:
        npdata = data.asnumpy().flatten().sum()
        sys.stdout.flush()
        batchcount += 1
        nplabel = label.asnumpy()
        for i in range(nplabel.shape[0]):
            labelcount[int(nplabel[i])] += 1
    for i in range(10):
        assert(labelcount[i] == 5000)

if __name__ == "__main__":
    #test_MNISTIter()
    test_Cifar10Rec()
