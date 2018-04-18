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
__all__ = ['Dataset', 'SimpleDataset', 'ArrayDataset',
           'RecordFileDataset']

import os

from ... import recordio, ndarray


class Dataset(object):
    """Abstract dataset class. All datasets should have this interface.

    Subclasses need to override `__getitem__`, which returns the i-th
    element, and `__len__`, which returns the total number elements.

    .. note:: An mxnet or numpy array can be directly used as a dataset.
    """
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def transform(self, fn, lazy=True):
        """Returns a new dataset with each sample transformed by the
        transformer function `fn`.

        Parameters
        ----------
        fn : callable
            A transformer function that takes a sample as input and
            returns the transformed sample.
        lazy : bool, default True
            If False, transforms all samples at once. Otherwise,
            transforms each sample on demand. Note that if `fn`
            is stochastic, you must set lazy to True or you will
            get the same result on all epochs.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        trans = _LazyTransformDataset(self, fn)
        if lazy:
            return trans
        return SimpleDataset([i for i in trans])

    def transform_first(self, fn, lazy=True):
        """Returns a new dataset with the first element of each sample
        transformed by the transformer function `fn`.

        This is useful, for example, when you only want to transform data
        while keeping label as is.

        Parameters
        ----------
        fn : callable
            A transformer function that takes the first elemtn of a sample
            as input and returns the transformed element.
        lazy : bool, default True
            If False, transforms all samples at once. Otherwise,
            transforms each sample on demand. Note that if `fn`
            is stochastic, you must set lazy to True or you will
            get the same result on all epochs.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        def base_fn(x, *args):
            if args:
                return (fn(x),) + args
            return fn(x)
        return self.transform(base_fn, lazy)


class SimpleDataset(Dataset):
    """Simple Dataset wrapper for lists and arrays.

    Parameters
    ----------
    data : dataset-like object
        Any object that implements `len()` and `[]`.
    """
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class _LazyTransformDataset(Dataset):
    """Lazily transformed dataset."""
    def __init__(self, data, fn):
        self._data = data
        self._fn = fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        if isinstance(item, tuple):
            return self._fn(*item)
        return self._fn(item)


class ArrayDataset(Dataset):
    """A dataset that combines multiple dataset-like objects, e.g.
    Datasets, lists, arrays, etc.

    The i-th sample is defined as `(x1[i], x2[i], ...)`.

    Parameters
    ----------
    *args : one or more dataset-like objects
        The data arrays.
    """
    def __init__(self, *args):
        assert len(args) > 0, "Needs at least 1 arrays"
        self._length = len(args[0])
        self._data = []
        for i, data in enumerate(args):
            assert len(data) == self._length, \
                "All arrays must have the same length; array[0] has length %d " \
                "while array[%d] has %d." % (self._length, i+1, len(data))
            if isinstance(data, ndarray.NDArray) and len(data.shape) == 1:
                data = data.asnumpy()
            self._data.append(data)

    def __getitem__(self, idx):
        if len(self._data) == 1:
            return self._data[0][idx]
        else:
            return tuple(data[idx] for data in self._data)

    def __len__(self):
        return self._length


class RecordFileDataset(Dataset):
    """A dataset wrapping over a RecordIO (.rec) file.

    Each sample is a string representing the raw content of an record.

    Parameters
    ----------
    filename : str
        Path to rec file.
    """
    def __init__(self, filename):
        idx_file = os.path.splitext(filename)[0] + '.idx'
        self._record = recordio.MXIndexedRecordIO(idx_file, filename, 'r')

    def __getitem__(self, idx):
        return self._record.read_idx(self._record.keys[idx])

    def __len__(self):
        return len(self._record.keys)


class _DownloadedDataset(Dataset):
    """Base class for MNIST, cifar10, etc."""
    def __init__(self, root, transform):
        super(_DownloadedDataset, self).__init__()
        self._transform = transform
        self._data = None
        self._label = None
        root = os.path.expanduser(root)
        self._root = root
        if not os.path.isdir(root):
            os.makedirs(root)
        self._get_data()

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data[idx], self._label[idx])
        return self._data[idx], self._label[idx]

    def __len__(self):
        return len(self._label)

    def _get_data(self):
        raise NotImplementedError
