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
import warnings
import numpy as np

from .. import dataset
from ...utils import download, check_sha1
from .... import nd, image, recordio

from .utils import _DownloadedDataset


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

    def __init__(self, root='~/.mxnet/datasets/mnist', train=True,
                 transform=None):
        self._base_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com' \
                         '/gluon/dataset/mnist/'
        self._train_data = ('train-images-idx3-ubyte.gz',
                            '6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d')
        self._train_label = ('train-labels-idx1-ubyte.gz',
                             '2a80914081dc54586dbdf242f9805a6b8d2a15fc')
        self._test_data = ('t10k-images-idx3-ubyte.gz',
                           'c3a25af1f52dad7f726cce8cacb138654b760d48')
        self._test_label = ('t10k-labels-idx1-ubyte.gz',
                            '763e7fa3757d93b0cdec073cef058b2004252c17')
        super(MNIST, self).__init__(root, train, transform)

    def _get_data(self):
        if self._train:
            data, label = self._train_data, self._train_label
        else:
            data, label = self._test_data, self._test_label

        data_file = download(self._base_url + data[0], self._root,
                             sha1_hash=data[1])
        label_file = download(self._base_url + label[0], self._root,
                              sha1_hash=label[1])

        with gzip.open(label_file, 'rb') as fin:
            struct.unpack(">II", fin.read(8))
            label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)

        with gzip.open(data_file, 'rb') as fin:
            struct.unpack(">IIII", fin.read(16))
            data = np.fromstring(fin.read(), dtype=np.uint8)
            data = data.reshape(len(label), 28, 28, 1)

        self._data = nd.array(data, dtype=data.dtype)
        self._label = label


class FashionMNIST(MNIST):
    """A dataset of Zalando's article images consisting of fashion products,
    a drop-in replacement of the original MNIST dataset from
    `https://github.com/zalandoresearch/fashion-mnist`_.

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

    def __init__(self, root='~/.mxnet/datasets/fashion-mnist', train=True,
                 transform=None):
        self._base_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com' \
                         '/gluon/dataset/fashion-mnist/'
        self._train_data = ('train-images-idx3-ubyte.gz',
                            '0cf37b0d40ed5169c6b3aba31069a9770ac9043d')
        self._train_label = ('train-labels-idx1-ubyte.gz',
                             '236021d52f1e40852b06a4c3008d8de8aef1e40b')
        self._test_data = ('t10k-images-idx3-ubyte.gz',
                           '626ed6a7c06dd17c0eec72fa3be1740f146a2863')
        self._test_label = ('t10k-labels-idx1-ubyte.gz',
                            '17f9ab60e7257a1620f4ad76bbbaf857c3920701')
        super(MNIST, self).__init__(root, train, transform)  # pylint: disable=bad-super-call
