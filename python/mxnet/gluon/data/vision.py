# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=
"""Dataset container."""

import os
import gzip
import tarfile
import struct
import numpy as np

from . import dataset
from ..utils import download
from ... import nd


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

    Each sample is an image (in 3D NDArray) with shape (28, 28, 1).

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
            data = data.reshape(len(label), 28, 28, 1)

        self._data = [nd.array(x, dtype=x.dtype) for x in data]
        self._label = label


class CIFAR10(_DownloadedDataset):
    """CIFAR10 image classification dataset from `https://www.cs.toronto.edu/~kriz/cifar.html`_.

    Each sample is an image (in 3D NDArray) with shape (32, 32, 1).

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

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.fromstring(fin.read(), dtype=np.uint8).reshape(-1, 3072+1)

        return data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), \
               data[:, 0].astype(np.int32)

    def _get_data(self):
        if not os.path.isdir(self._root):
            os.makedirs(self._root)
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
        filename = download(url, self._root)

        with tarfile.open(filename) as tar:
            tar.extractall(self._root)

        if self._train:
            filename = os.path.join(self._root, 'cifar-10-batches-bin/data_batch_%d.bin')
            data, label = zip(*[self._read_batch(filename%i) for i in range(1, 6)])
            data = np.concatenate(data)
            label = np.concatenate(label)
        else:
            filename = os.path.join(self._root, 'cifar-10-batches-bin/test_batch.bin')
            data, label = self._read_batch(filename)

        self._data = [nd.array(x, dtype=x.dtype) for x in data]
        self._label = label
