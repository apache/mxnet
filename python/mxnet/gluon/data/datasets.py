# coding: utf-8
# pylint: disable=
"""Dataset container."""

import os
import gzip
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

        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        with open(fname, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    return fname


class MNIST(dataset.Dataset):
    """MNIST handwritten digits dataset from `http://yann.lecun.com/exdb/mnist`.

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
        self._root = os.path.expanduser(root)
        self._train = train
        self._transform = transform

        self._get_data()

    def __getitem__(self, idx):
        return self._transform(self._data[idx], self._label[idx])

    def __len__(self):
        return len(self._label)

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
            _, _, rows, cols = struct.unpack(">IIII", fin.read(16))
            data = np.fromstring(fin.read(), dtype=np.uint8)
            data = data.reshape(len(label), rows*cols)

        self._label = label
        self._data = data
