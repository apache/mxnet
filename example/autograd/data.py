# pylint: skip-file
""" data iterator for mnist """
import sys
import os
# code to automatically download dataset
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../../tests/python/common"))
import get_data
import mxnet as mx

def mnist_iterator(batch_size, input_shape):
    """return train and val iterators for mnist"""
    # download data
    get_data.GetMNIST_ubyte()
    flat = False if len(input_shape) == 3 else True

    train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat)

    val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat)

    return (train_dataiter, val_dataiter)


def cifar10_iterator(batch_size, data_shape):
    train = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/train.rec",
        # mean_img    = "data/cifar/mean.bin",
        data_shape  = data_shape,
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = True)

    val = mx.io.ImageRecordIter(
        path_imgrec = "data/cifar/test.rec",
        # mean_img    = "data/cifar/mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = batch_size)

    return train, val
