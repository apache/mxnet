#pylint: skip-file
import mxnet as mx
import numpy as np
import os

dataiter = mx.io.MNISTIterator(path_img="/home/tianjun/data/mnist/train-images-idx3-ubyte",
        path_label="/home/tianjun/data/mnist/train-labels-idx1-ubyte",
        batch_size=100, shuffle=1, silent=1, input_flat="flat")

dataiter.beforefirst()

idx = 0
while dataiter.next():
    info = "Batch %d" % (idx)
    idx += 1
    print info
    '''
    label = dataiter.getlabel()
    print label.numpy
    '''
