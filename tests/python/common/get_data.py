# pylint: skip-file
import os, gzip
import pickle as pickle
import sys
from mxnet.test_utils import download
import zipfile
import mxnet as mx

# download mnist.pkl.gz
def GetMNIST_pkl():
    if not os.path.isdir("data"):
        os.makedirs('data')
    if not os.path.exists('data/mnist.pkl.gz'):
        download('http://deeplearning.net/data/mnist/mnist.pkl.gz',
                 dirname='data')

# download ubyte version of mnist and untar
def GetMNIST_ubyte():
    if not os.path.isdir("data"):
        os.makedirs('data')
    if (not os.path.exists('data/train-images-idx3-ubyte')) or \
       (not os.path.exists('data/train-labels-idx1-ubyte')) or \
       (not os.path.exists('data/t10k-images-idx3-ubyte')) or \
       (not os.path.exists('data/t10k-labels-idx1-ubyte')):
        zip_file_path = download('http://data.mxnet.io/mxnet/data/mnist.zip',
                                 dirname='data')
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall('data')

# download cifar
def GetCifar10():
    if not os.path.isdir("data"):
        os.makedirs('data')
    if (not os.path.exists('data/cifar/train.rec')) or \
       (not os.path.exists('data/cifar/test.rec')) or \
       (not os.path.exists('data/cifar/train.lst')) or \
       (not os.path.exists('data/cifar/test.lst')):
        zip_file_path = download('http://data.mxnet.io/mxnet/data/cifar10.zip',
                                 dirname='data')
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall('data')

def MNISTIterator(batch_size, input_shape):
    """return train and val iterators for mnist"""
    # download data
    GetMNIST_ubyte()
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
