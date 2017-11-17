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
__all__ = ['Dataset', 'ArrayDataset', 'RecordFileDataset']

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


class ArrayDataset(Dataset):
    """A dataset of multiple arrays.

    The i-th sample is `(x1[i], x2[i], ...)`.

    Parameters
    ----------
    *args : one or more arrays
        The data arrays.
    """
    def __init__(self, *args):
        assert len(args) > 0, "Needs at least 1 arrays"
        self._length = len(args[0])
        self._data = []
        for i, data in enumerate(args):
            assert len(data) == self._length, \
                "All arrays must have the same length. But the first has %s " \
                "while the %d-th has %d."%(length, i+1, len(data))
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
