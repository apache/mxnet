# pylint: skip-file
import os, gzip
import pickle as pickle
import sys

# download mnist.pkl.gz
def GetMNIST_pkl():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if not os.path.exists('data/mnist.pkl.gz'):
        os.system("wget http://deeplearning.net/data/mnist/mnist.pkl.gz -P data/")

# download ubyte version of mnist and untar
def GetMNIST_ubyte():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if not os.path.exists('data/train-images-idx3-ubyte'):
        os.system("wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -P data/")
        os.system("gunzip data/train-images-idx3-ubyte.gz")
    if not os.path.exists('data/train-labels-idx1-ubyte'):
        os.system("wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -P data/")
        os.system("gunzip data/train-labels-idx1-ubyte.gz")
    if not os.path.exists('data/t10k-images-idx3-ubyte'):
        os.system("wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P data/")
        os.system("gunzip data/t10k-images-idx3-ubyte.gz")
    if not os.path.exists('data/t10k-labels-idx1-ubyte'):
        os.system("wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P data/")
        os.system("gunzip data/t10k-labels-idx1-ubyte.gz")

# download cifar
def GetCifar10():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if not os.path.exists('data/cifar10.zip'):
        os.system("wget http://webdocs.cs.ualberta.ca/~bx3/data/cifar10.zip -P data/")
        os.chdir("./data")
        os.system("unzip cifar10.zip")
        os.chidr("..")
