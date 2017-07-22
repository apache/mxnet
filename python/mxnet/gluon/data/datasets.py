# coding: utf-8
# pylint: disable=
"""Dataset container."""

import os
import gzip
import tarfile
import struct
try:
    import requests
except ImportError:
    class requests_failed_to_import(object):
        pass
    requests = requests_failed_to_import
import numpy as np

from . import dataset


def download(url, path=None, overwrite=False):
    """Download an given URL

    Parameters
    ----------
    url : str
        URL to download
    fname : str, optional
        Filename of the downloaded file. If None, then will guess a filename
        from url.
    dirname : str, optional
        Output directory name. If None, use the current directory
    overwrite : bool, optional
        Default is false, which means skipping download if the local file
        exists. If true, then download the url to overwrite the local file if
        exists.

    Returns
    -------
    str
        The filename of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    elif os.path.isdir(path):
        fname = os.path.join(path, url.split('/')[-1])
    else:
        fname = path

    if overwrite or not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        with open(fname, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    return fname


class _DownloadedDataset(dataset.Dataset):
    """Base class for MNIST, cifar10, etc."""
    def __init__(self, root, train, transform):
        self._root = os.path.expanduser(root)
        self._train = train
        self._transform = transform
        self._data = None
        self._label = None

        self._get_data()

    def __getitem__(self, idx):
        return self._transform(self._data[idx], self._label[idx])

    def __len__(self):
        return len(self._label)

    def _get_data(self):
        raise NotImplementedError


class MNIST(_DownloadedDataset):
    """MNIST handwritten digits dataset from `http://yann.lecun.com/exdb/mnist`_.

    Parameters
    ----------
    root : str
        Path to temp folder for storing data.
    train : bool
        Whether to load the training or testing set.
    transform : function
        A user defined callback that transforms each instance. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)
    """
    def __init__(self, root, train=True, transform=lambda data, label: (data, label)):
        super(MNIST, self).__init__(root, train, transform)

    def _get_data(self):
        if not os.path.isdir(self._root):
            os.makedirs(self._root)
        url = 'http://data.mxnet.io/data/mnist/'
        if self._train:
            data_file = download(url+'train-images-idx3-ubyte.gz', self._root)
            label_file = download(url+'train-labels-idx1-ubyte.gz', self._root)
        else:
            data_file = download(url+'t10k-images-idx3-ubyte.gz', self._root)
            label_file = download(url+'t10k-labels-idx1-ubyte.gz', self._root)

        with gzip.open(label_file, 'rb') as fin:
            struct.unpack(">II", fin.read(8))
            label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)

        with gzip.open(data_file, 'rb') as fin:
            struct.unpack(">IIII", fin.read(16))
            data = np.fromstring(fin.read(), dtype=np.uint8)
            data = data.reshape(len(label), 1, 28, 28)

        self._label = label
        self._data = data


class CIFAR10(_DownloadedDataset):
    """CIFAR10 image classification dataset from `https://www.cs.toronto.edu/~kriz/cifar.html`_.

    Parameters
    ----------
    root : str
        Path to temp folder for storing data.
    train : bool
        Whether to load the training or testing set.
    transform : function
        A user defined callback that transforms each instance. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)
    """
    def __init__(self, root, train=True, transform=lambda data, label: (data, label)):
        super(CIFAR10, self).__init__(root, train, transform)
        print self._data.shape, self._label.shape

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.fromstring(fin.read(), dtype=np.uint8).reshape(-1, 3072+1)

        return data[:, 1:].reshape(-1, 3, 32, 32), data[:, 0].astype(np.int32)

    def _get_data(self):
        if not os.path.isdir(self._root):
            os.makedirs(self._root)
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
        filename = download(url, self._root)

        with tarfile.open(filename) as tar:
            tar.extractall(self._root)

        if self._train:
            filename = os.path.join(self._root, 'cifar-10-batches-bin/data_batch_%d.bin')
            data, label = zip(*[self._read_batch(filename%i) for i in range(1,6)])
            data = np.concatenate(data)
            label = np.concatenate(label)
        else:
            filename = os.path.join(self._root, 'cifar-10-batches-bin/test_batch.bin')
            data, label = self._read_batch(filename)

        self._data = data
        self._label = label
