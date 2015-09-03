# pylint: skip-file
import mxnet as mx
import numpy as np
import os, gzip
import pickle as pickle
import sys
import get_data

# prepare data
get_data.GetMNIST_ubyte()

batch_size = 100
train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        batch_size=batch_size, shuffle=1, flat=1, silent=0, seed=10)
val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        batch_size=batch_size, shuffle=0, flat=1, silent=0)

def test_MNISTIter_loop():
    nbatch = 60000 / batch_size
    batch_count = 0
    for data, label in train_dataiter:
        batch_count += 1
    assert(nbatch == batch_count)

def test_MNISTIter_reset():
    train_dataiter.reset()
    train_dataiter.iter_next()
    label_0 = train_dataiter.getlabel().numpy.flatten()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.iter_next()
    train_dataiter.reset()
    train_dataiter.iter_next()
    label_1 = train_dataiter.getlabel().numpy.flatten()
    assert(sum(label_0 - label_1) == 0)

def test_ImageRecIter():
    dataiter = mx.io.ImageRecordIter(path_imgrec="data/val_cxxnet.rec",
            image_mean="data/val_cxxnet_mean.bin",
            rand_crop=True,
            rand_mirror=True,
            input_shape="3,224,224",
            batch_size=128)








