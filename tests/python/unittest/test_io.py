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
            batch_size=batch_size, shuffle=1, flat=1, silent=0, seed=10)
    val_dataiter = mx.io.MNISTIter(
            image="data/t10k-images-idx3-ubyte",
            label="data/t10k-labels-idx1-ubyte",
            batch_size=batch_size, shuffle=0, flat=1, silent=0)
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

'''
def test_ImageRecIter():
    dataiter = mx.io.ImageRecordIter(
            path_imgrec="data/val_cxxnet.rec",
            mean_img="data/smallset/image_net_mean.bin",
            rand_crop=True,
            mirror=True,
            input_shape=(3,227,227),
            batch_size=100,
            nthread=1,
            seed=10)
    labelcount = [0 for i in range(1000)] 
    batchcount = 0
    for data, label in dataiter:
        npdata = data.numpy
        print npdata[0,:,:,:]
        imgdata = np.zeros([227, 227, 3], dtype=np.uint8)
        imgdata[:,:,0] = npdata[10,2,:,:]
        imgdata[:,:,1] = npdata[10,1,:,:]
        imgdata[:,:,2] = npdata[10,0,:,:]
        img = Image.fromarray(imgdata)
        imgpath = "data/smallset/test_3.jpg"
        img.save(imgpath, format='JPEG')
        exit(0)
        print batchcount
        sys.stdout.flush()
        batchcount += 1
        nplabel = label.numpy
        for i in range(nplabel.shape[0]):
            labelcount[int(nplabel[i])] += 1
'''

def test_Cifar10Rec():
    dataiter = mx.io.ImageRecordIter(
            path_imgrec="data/cifar/train.rec",
            mean_img="data/cifar/cifar10_mean_1.bin",
            rand_crop=False,
            rand_mirror=False,
            shuffle=False,
            input_shape=(3,28,28),
            batch_size=100,
            nthread=1,
            capacity=1)
    labelcount = [0 for i in range(10)] 
    batchcount = 0
    
    for data, label in dataiter:
        npdata = data.asnumpy().flatten().sum()
        print "Batch: ", batchcount
        sys.stdout.flush()
        batchcount += 1
        nplabel = label.asnumpy()
        for i in range(nplabel.shape[0]):
            labelcount[int(nplabel[i])] += 1
    for i in range(10):
        assert(labelcount[i] == 1000)

if __name__ == "__main__":
    CheckEqual()
    #test_Cifar10Rec()

