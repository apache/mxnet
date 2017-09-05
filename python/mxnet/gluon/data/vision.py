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

from . import dataset
from ..utils import download, check_sha1
from ... import nd, image, recordio


class _DownloadedDataset(dataset.Dataset):
    """Base class for MNIST, cifar10, etc."""
    def __init__(self, root, train, transform):
        self._root = os.path.expanduser(root)
        self._train = train
        self._transform = transform
        self._data = None
        self._label = None

        if not os.path.isdir(self._root):
            os.makedirs(self._root)
        self._get_data()

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data[idx], self._label[idx])
        return self._data[idx], self._label[idx]

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
    def __init__(self, root='~/.mxnet/datasets/mnist', train=True,
                 transform=None):
        self._base_url = 'https://apache-mxnet.s3.amazonaws.com/gluon/dataset/mnist/'
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
        self._base_url = 'https://apache-mxnet.s3.amazonaws.com/gluon/dataset/fashion-mnist/'
        self._train_data = ('train-images-idx3-ubyte.gz',
                            '0cf37b0d40ed5169c6b3aba31069a9770ac9043d')
        self._train_label = ('train-labels-idx1-ubyte.gz',
                             '236021d52f1e40852b06a4c3008d8de8aef1e40b')
        self._test_data = ('t10k-images-idx3-ubyte.gz',
                           '626ed6a7c06dd17c0eec72fa3be1740f146a2863')
        self._test_label = ('t10k-labels-idx1-ubyte.gz',
                            '17f9ab60e7257a1620f4ad76bbbaf857c3920701')
        super(MNIST, self).__init__(root, train, transform) # pylint: disable=bad-super-call


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
    def __init__(self, root='~/.mxnet/datasets/cifar10', train=True,
                 transform=None):
        self._file_hashes = {'data_batch_1.bin': 'aadd24acce27caa71bf4b10992e9e7b2d74c2540',
                             'data_batch_2.bin': 'c0ba65cce70568cd57b4e03e9ac8d2a5367c1795',
                             'data_batch_3.bin': '1dd00a74ab1d17a6e7d73e185b69dbf31242f295',
                             'data_batch_4.bin': 'aab85764eb3584312d3c7f65fd2fd016e36a258e',
                             'data_batch_5.bin': '26e2849e66a845b7f1e4614ae70f4889ae604628',
                             'test_batch.bin': '67eb016db431130d61cd03c7ad570b013799c88c'}
        super(CIFAR10, self).__init__(root, train, transform)

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.fromstring(fin.read(), dtype=np.uint8).reshape(-1, 3072+1)

        return data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), \
               data[:, 0].astype(np.int32)

    def _get_data(self):
        file_paths = [(name, os.path.join(self._root, 'cifar-10-batches-bin/', name))
                      for name in self._file_hashes]
        if any(not os.path.exists(path) or not check_sha1(path, self._file_hashes[name])
               for name, path in file_paths):
            url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
            filename = download(url, self._root,
                                sha1_hash='e8aa088b9774a44ad217101d2e2569f823d2d491')

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

        self._data = nd.array(data, dtype=data.dtype)
        self._label = label


class ImageRecordDataset(dataset.RecordFileDataset):
    """A dataset wrapping over a RecordIO file containing images.

    Each sample is an image and its corresponding label.

    Parameters
    ----------
    filename : str
        Path to rec file.
    flag : {0, 1}, default 1
        If 0, always convert images to greyscale.

        If 1, always convert images to colored (RGB).
    transform : function
        A user defined callback that transforms each instance. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)
    """
    def __init__(self, filename, flag=1, transform=None):
        super(ImageRecordDataset, self).__init__(filename)
        self._flag = flag
        self._transform = transform

    def __getitem__(self, idx):
        record = super(ImageRecordDataset, self).__getitem__(idx)
        header, img = recordio.unpack(record)
        if self._transform is not None:
            return self._transform(image.imdecode(img, self._flag), header.label)
        return image.imdecode(img, self._flag), header.label


class ImageFolderDataset(dataset.Dataset):
    """A dataset for loading image files stored in a folder structure like::

        root/car/0001.jpg
        root/car/xxxa.jpg
        root/car/yyyb.jpg
        root/bus/123.jpg
        root/bus/023.jpg
        root/bus/wwww.jpg

    Parameters
    ----------
    root : str
        Path to root directory.
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).
    transform : callable
        A function that takes data and label and transforms them::

            transform = lambda data, label: (data.astype(np.float32)/255, label)

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all images in (filename, label) pairs.
    """
    def __init__(self, root, flag=1, transform=None):
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_iamges(self._root)

    def _list_iamges(self, root):
        self.synsets = []
        self.items = []

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.'%path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s'%(
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, label))

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)
