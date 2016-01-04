# pylint: skip-file
import os, gzip
import pickle as pickle
import sys
import requests
import zipfile

def download_file(url, target_file):
    r = requests.get(url, stream=True)
    with open(target_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

# download mnist.pkl.gz
def GetMNIST_pkl():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if not os.path.exists('data/mnist.pkl.gz'):
        download_file("http://deeplearning.net/data/mnist/mnist.pkl.gz", "data/mnist.pkl.gz")

# download ubyte version of mnist and untar
def GetMNIST_ubyte():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if (not os.path.exists('data/train-images-idx3-ubyte')) or \
       (not os.path.exists('data/train-labels-idx1-ubyte')) or \
       (not os.path.exists('data/t10k-images-idx3-ubyte')) or \
       (not os.path.exists('data/t10k-labels-idx1-ubyte')):
        download_file("http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip", "data/mnist.zip")
        os.chdir("./data")
        with zipfile.ZipFile('mnist.zip', "r") as z:
            z.extractall()
        os.chdir("..")

# download cifar
def GetCifar10():
    if not os.path.isdir("data/"):
        os.system("mkdir data/")
    if not os.path.exists('data/cifar10.zip'):
        download_file("http://webdocs.cs.ualberta.ca/~bx3/data/cifar10.zip", "data/cfar10.zip")
        os.chdir("./data")
        with zipfile.ZipFile('cifar10.zip', "r") as z:
            z.extractall()
        os.chdir("..")
