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
    """A dataset with a data array and a label array.

    The i-th sample is `(data[i], lable[i])`.

    Parameters
    ----------
    data : array-like object
        The data array. Can be mxnet or numpy array.
    label : array-like object
        The label array. Can be mxnet or numpy array.
    """
    def __init__(self, data, label):
        assert len(data) == len(label)
        self._data = data
        if isinstance(label, ndarray.NDArray) and len(label.shape) == 1:
            self._label = label.asnumpy()
        else:
            self._label = label

    def __getitem__(self, idx):
        return self._data[idx], self._label[idx]

    def __len__(self):
        return len(self._data)


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
        return self._record.read_idx(idx)

    def __len__(self):
        return len(self._record.keys)
