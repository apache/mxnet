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

    def filter(self, filter_fn):
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
        return self.transform(_TransformFirstClosure(fn), lazy)

class RangeFilter(Filter):
    """RangeFilter filters the data samples based on the range [start_idx, end_idx)
    from the dataset. Only data samples within the range passes the filter.
    Parameters
    ----------
    start_idx : int
        The start index (included).
    end_idx : int or None
        The end index (excluded). If set to None, it is set to infinity.
    Example
    -------
    >>> data =  "a,b,c\n"
    >>> data += "d,e,f\n"
    >>> data += "g,h,i\n"
    >>> data += "j,k,l\n"
    >>> data += "m,n,o\n"
    >>> with open('test_range_filter.txt', 'w') as fout:
    >>>     fout.write(data)
    >>>
    >>> # create 2 partitions, and read partition 0 only
    >>> filter_fn = nlp.data.RangeFilter(1, 3)
    >>> dataset = nlp.data.TextLineDataset('test_range_filter.txt', filter_fn=filter_fn)
    >>> len(dataset)
    2
    >>> dataset[0]
    "d,e,f"
    >>> dataset[1]
    "g,h,i"
    """
    def __init__(self, start_idx, end_idx):
        self.start = start_idx
        self.end = end_idx
        if end_idx is not None:
            assert self.start < self.end, 'end_idx must be greater than start_idx'

    def __call__(self, index, data):
        """Check if the data sample passes the filter.
        Parameters
        ----------
        index : int
            The original dataset index before filtering is applied.
        sample : object
            The original data sample object at the provided index.
        """
        if self.end is not None:
            return index >= self.start and index < self.end
        else:
            return index >= self.start


class SplitFilter(object):
    """SplitFilter filters the data samples based on the number of partitions
    and partition index of the dataset. Only data samples for the target
    partition index passes the filter.

    Parameters
    ----------
    num_parts : int
        The number of partitions.
    part_idx : int
        The target partition index that will pass the filter.

    Example
    -------
    >>> data =  "a,b,c\n"
    >>> data += "d,e,f\n"
    >>> data += "g,h,i\n"
    >>> data += "j,k,l\n"
    >>> data += "m,n,o\n"
    >>> with open('test_split_filter.txt', 'w') as fout:
    >>>     fout.write(data)
    >>>
    >>> # create 2 partitions, and read partition 0 only
    >>> filter_fn = nlp.data.SplitFilter(2, 0)
    >>> dataset = nlp.data.TextLineDataset('test_split_filter.txt', filter_fn=filter_fn)
    >>> len(dataset)
    3
    >>> dataset[0]
    "a,b,c"
    >>> dataset[1]
    "g,h,i"
    >>> dataset[2]
    "m,n,o"
    """
    def __init__(self, num_parts, part_idx):
        self.num_parts = num_parts
        self.part_idx = part_idx
        assert self.part_idx < self.num_parts, 'part_idx should be less than num_parts'

    def __call__(self, index, data):
        """Check if the data sample passes the filter.
        Parameters
        ----------
        index : int
            The original dataset index before filtering is applied.
        sample : object
            The original data sample object at the provided index.
        """
        return index % self.num_parts == self.part_idx


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

    def filter(self, filter_fn):
        data = []
        for i in range(len(self)):
            sample = self[i]
            if filter_fn(i, sample):
                data.append(sample)
        return SimpleDataset(data)


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


class _TransformFirstClosure(object):
    """Use callable object instead of nested function, it can be pickled."""
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, x, *args):
        if args:
            return (self._fn(x),) + args
        return self._fn(x)

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
        self.idx_file = os.path.splitext(filename)[0] + '.idx'
        self.filename = filename
        self._record = recordio.MXIndexedRecordIO(self.idx_file, self.filename, 'r')

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
