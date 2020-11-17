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
"""Contrib data iterators for common data formats."""
from ..io import DataIter, DataDesc
from .. import ndarray as nd


class DataLoaderIter(DataIter):
    """Returns an iterator for ``mx.gluon.data.Dataloader`` so gluon dataloader
    can be used in symbolic module.

    Parameters
    ----------
    loader : mxnet.gluon.data.Dataloader
        Gluon dataloader instance
    data_name : str, optional
        The data name.
    label_name : str, optional
        The label name.
    dtype : str, optional
        The dtype specifier, can be float32 or float16

    Examples
    --------
    >>> import mxnet as mx
    >>> from mxnet.gluon.data.vision import MNIST
    >>> from mxnet.gluon.data import DataLoader
    >>> train_dataset = MNIST(train=True)
    >>> train_data = mx.gluon.data.DataLoader(train_dataset, 32, shuffle=True, num_workers=4)
    >>> dataiter = mx.io.DataloaderIter(train_data)
    >>> for batch in dataiter:
    ...     batch.data[0].shape
    ...
    (32L, 28L, 28L, 1L)
    """
    def __init__(self, loader, data_name='data', label_name='softmax_label', dtype='float32'):
        super(DataLoaderIter, self).__init__()
        self._loader = loader
        self._iter = iter(self._loader)
        data, label = next(self._iter)
        self.batch_size = data.shape[0]
        self.dtype = dtype
        self.provide_data = [DataDesc(data_name, data.shape, dtype)]
        self.provide_label = [DataDesc(label_name, label.shape, dtype)]
        self._current_batch = None
        self.reset()

    def reset(self):
        self._iter = iter(self._loader)

    def iter_next(self):
        try:
            self._current_batch = next(self._iter)
        except StopIteration:
            self._current_batch = None
        return self._current_batch is not None

    def getdata(self):
        if self.getpad():
            dshape = self._current_batch[0].shape
            ret = nd.empty(shape=([self.batch_size] + list(dshape[1:])))
            ret[:dshape[0]] = self._current_batch[0].astype(self.dtype)
            return [ret]
        return [self._current_batch[0].astype(self.dtype)]

    def getlabel(self):
        if self.getpad():
            lshape = self._current_batch[1].shape
            ret = nd.empty(shape=([self.batch_size] + list(lshape[1:])))
            ret[:lshape[0]] = self._current_batch[1].astype(self.dtype)
            return [ret]
        return [self._current_batch[1].astype(self.dtype)]

    def getpad(self):
        return self.batch_size - self._current_batch[0].shape[0]

    def getindex(self):
        return None
