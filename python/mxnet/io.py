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

"""Data iterators for common data formats."""
from __future__ import absolute_import
from collections import OrderedDict, namedtuple

import sys
import ctypes
import logging
import threading
try:
    import h5py
except ImportError:
    h5py = None
import numpy as np
from .base import _LIB
from .base import c_str_array, mx_uint, py_str
from .base import DataIterHandle, NDArrayHandle
from .base import mx_real_t
from .base import check_call, build_param_doc as _build_param_doc
from .ndarray import NDArray
from .ndarray.sparse import CSRNDArray
from .ndarray.sparse import array as sparse_array
from .ndarray import _ndarray_cls
from .ndarray import array
from .ndarray import concatenate

class DataDesc(namedtuple('DataDesc', ['name', 'shape'])):
    """DataDesc is used to store name, shape, type and layout
    information of the data or the label.

    The `layout` describes how the axes in `shape` should be interpreted,
    for example for image data setting `layout=NCHW` indicates
    that the first axis is number of examples in the batch(N),
    C is number of channels, H is the height and W is the width of the image.

    For sequential data, by default `layout` is set to ``NTC``, where
    N is number of examples in the batch, T the temporal axis representing time
    and C is the number of channels.

    Parameters
    ----------
    cls : DataDesc
         The class.
    name : str
         Data name.
    shape : tuple of int
         Data shape.
    dtype : np.dtype, optional
         Data type.
    layout : str, optional
         Data layout.
    """
    def __new__(cls, name, shape, dtype=mx_real_t, layout='NCHW'): # pylint: disable=super-on-old-class
        ret = super(cls, DataDesc).__new__(cls, name, shape)
        ret.dtype = dtype
        ret.layout = layout
        return ret

    def __repr__(self):
        return "DataDesc[%s,%s,%s,%s]" % (self.name, self.shape, self.dtype,
                                          self.layout)

    @staticmethod
    def get_batch_axis(layout):
        """Get the dimension that corresponds to the batch size.

        When data parallelism is used, the data will be automatically split and
        concatenated along the batch-size dimension. Axis can be -1, which means
        the whole array will be copied for each data-parallelism device.

        Parameters
        ----------
        layout : str
            layout string. For example, "NCHW".

        Returns
        -------
        int
            An axis indicating the batch_size dimension.
        """
        if layout is None:
            return 0
        return layout.find('N')

    @staticmethod
    def get_list(shapes, types):
        """Get DataDesc list from attribute lists.

        Parameters
        ----------
        shapes : a tuple of (name, shape)
        types : a tuple of  (name, type)
        """
        if types is not None:
            type_dict = dict(types)
            return [DataDesc(x[0], x[1], type_dict[x[0]]) for x in shapes]
        else:
            return [DataDesc(x[0], x[1]) for x in shapes]

class DataBatch(object):
    """A data batch.

    MXNet's data iterator returns a batch of data for each `next` call.
    This data contains `batch_size` number of examples.

    If the input data consists of images, then shape of these images depend on
    the `layout` attribute of `DataDesc` object in `provide_data` parameter.

    If `layout` is set to 'NCHW' then, images should be stored in a 4-D matrix
    of shape ``(batch_size, num_channel, height, width)``.
    If `layout` is set to 'NHWC' then, images should be stored in a 4-D matrix
    of shape ``(batch_size, height, width, num_channel)``.
    The channels are often in RGB order.

    Parameters
    ----------
    data : list of `NDArray`, each array containing `batch_size` examples.
          A list of input data.
    label : list of `NDArray`, each array often containing a 1-dimensional array. optional
          A list of input labels.
    pad : int, optional
          The number of examples padded at the end of a batch. It is used when the
          total number of examples read is not divisible by the `batch_size`.
          These extra padded examples are ignored in prediction.
    index : numpy.array, optional
          The example indices in this batch.
    bucket_key : int, optional
          The bucket key, used for bucketing module.
    provide_data : list of `DataDesc`, optional
          A list of `DataDesc` objects. `DataDesc` is used to store
          name, shape, type and layout information of the data.
          The *i*-th element describes the name and shape of ``data[i]``.
    provide_label : list of `DataDesc`, optional
          A list of `DataDesc` objects. `DataDesc` is used to store
          name, shape, type and layout information of the label.
          The *i*-th element describes the name and shape of ``label[i]``.
    """
    def __init__(self, data, label=None, pad=None, index=None,
                 bucket_key=None, provide_data=None, provide_label=None):
        if data is not None:
            assert isinstance(data, (list, tuple)), "Data must be list of NDArrays"
        if label is not None:
            assert isinstance(label, (list, tuple)), "Label must be list of NDArrays"
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index

        self.bucket_key = bucket_key
        self.provide_data = provide_data
        self.provide_label = provide_label

    def __str__(self):
        data_shapes = [d.shape for d in self.data]
        if self.label:
            label_shapes = [l.shape for l in self.label]
        else:
            label_shapes = None
        return "{}: data shapes: {} label shapes: {}".format(
            self.__class__.__name__,
            data_shapes,
            label_shapes)

class DataIter(object):
    """The base class for an MXNet data iterator.

    All I/O in MXNet is handled by specializations of this class. Data iterators
    in MXNet are similar to standard-iterators in Python. On each call to `next`
    they return a `DataBatch` which represents the next batch of data. When
    there is no more data to return, it raises a `StopIteration` exception.

    Parameters
    ----------
    batch_size : int, optional
        The batch size, namely the number of items in the batch.

    See Also
    --------
    NDArrayIter : Data-iterator for MXNet NDArray or numpy-ndarray objects.
    CSVIter : Data-iterator for csv data.
    LibSVMIter : Data-iterator for libsvm data.
    ImageIter : Data-iterator for images.
    """
    def __init__(self, batch_size=0):
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator to the begin of the data."""
        pass

    def next(self):
        """Get next data batch from iterator.

        Returns
        -------
        DataBatch
            The data of next batch.

        Raises
        ------
        StopIteration
            If the end of the data is reached.
        """
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        """Move to the next batch.

        Returns
        -------
        boolean
            Whether the move is successful.
        """
        pass

    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        list of NDArray
            The data of the current batch.
        """
        pass

    def getlabel(self):
        """Get label of the current batch.

        Returns
        -------
        list of NDArray
            The label of the current batch.
        """
        pass

    def getindex(self):
        """Get index of the current batch.

        Returns
        -------
        index : numpy.array
            The indices of examples in the current batch.
        """
        return None

    def getpad(self):
        """Get the number of padding examples in the current batch.

        Returns
        -------
        int
            Number of padding examples in the current batch.
        """
        pass

class ResizeIter(DataIter):
    """Resize a data iterator to a given number of batches.

    Parameters
    ----------
    data_iter : DataIter
        The data iterator to be resized.
    size : int
        The number of batches per epoch to resize to.
    reset_internal : bool
        Whether to reset internal iterator on ResizeIter.reset.


    Examples
    --------
    >>> nd_iter = mx.io.NDArrayIter(mx.nd.ones((100,10)), batch_size=25)
    >>> resize_iter = mx.io.ResizeIter(nd_iter, 2)
    >>> for batch in resize_iter:
    ...     print(batch.data)
    [<NDArray 25x10 @cpu(0)>]
    [<NDArray 25x10 @cpu(0)>]
    """
    def __init__(self, data_iter, size, reset_internal=True):
        super(ResizeIter, self).__init__()
        self.data_iter = data_iter
        self.size = size
        self.reset_internal = reset_internal
        self.cur = 0
        self.current_batch = None

        self.provide_data = data_iter.provide_data
        self.provide_label = data_iter.provide_label
        self.batch_size = data_iter.batch_size
        if hasattr(data_iter, 'default_bucket_key'):
            self.default_bucket_key = data_iter.default_bucket_key

    def reset(self):
        self.cur = 0
        if self.reset_internal:
            self.data_iter.reset()

    def iter_next(self):
        if self.cur == self.size:
            return False
        try:
            self.current_batch = self.data_iter.next()
        except StopIteration:
            self.data_iter.reset()
            self.current_batch = self.data_iter.next()

        self.cur += 1
        return True

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad

class PrefetchingIter(DataIter):
    """Performs pre-fetch for other data iterators.

    This iterator will create another thread to perform ``iter_next`` and then
    store the data in memory. It potentially accelerates the data read, at the
    cost of more memory usage.

    Parameters
    ----------
    iters : DataIter or list of DataIter
        The data iterators to be pre-fetched.
    rename_data : None or list of dict
        The *i*-th element is a renaming map for the *i*-th iter, in the form of
        {'original_name' : 'new_name'}. Should have one entry for each entry
        in iter[i].provide_data.
    rename_label : None or list of dict
        Similar to ``rename_data``.

    Examples
    --------
    >>> iter1 = mx.io.NDArrayIter({'data':mx.nd.ones((100,10))}, batch_size=25)
    >>> iter2 = mx.io.NDArrayIter({'data':mx.nd.ones((100,10))}, batch_size=25)
    >>> piter = mx.io.PrefetchingIter([iter1, iter2],
    ...                               rename_data=[{'data': 'data_1'}, {'data': 'data_2'}])
    >>> print(piter.provide_data)
    [DataDesc[data_1,(25, 10L),<type 'numpy.float32'>,NCHW],
     DataDesc[data_2,(25, 10L),<type 'numpy.float32'>,NCHW]]
    """
    def __init__(self, iters, rename_data=None, rename_label=None):
        super(PrefetchingIter, self).__init__()
        if not isinstance(iters, list):
            iters = [iters]
        self.n_iter = len(iters)
        assert self.n_iter > 0
        self.iters = iters
        self.rename_data = rename_data
        self.rename_label = rename_label
        self.batch_size = self.provide_data[0][1][0]
        self.data_ready = [threading.Event() for i in range(self.n_iter)]
        self.data_taken = [threading.Event() for i in range(self.n_iter)]
        for i in self.data_taken:
            i.set()
        self.started = True
        self.current_batch = [None for i in range(self.n_iter)]
        self.next_batch = [None for i in range(self.n_iter)]
        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    self.next_batch[i] = self.iters[i].next()
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()
        self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) \
                                 for i in range(self.n_iter)]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for i in self.data_taken:
            i.set()
        for thread in self.prefetch_threads:
            thread.join()

    @property
    def provide_data(self):
        if self.rename_data is None:
            return sum([i.provide_data for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_data
            ] for r, i in zip(self.rename_data, self.iters)], [])

    @property
    def provide_label(self):
        if self.rename_label is None:
            return sum([i.provide_label for i in self.iters], [])
        else:
            return sum([[
                DataDesc(r[x.name], x.shape, x.dtype)
                if isinstance(x, DataDesc) else DataDesc(*x)
                for x in i.provide_label
            ] for r, i in zip(self.rename_label, self.iters)], [])

    def reset(self):
        for i in self.data_ready:
            i.wait()
        for i in self.iters:
            i.reset()
        for i in self.data_ready:
            i.clear()
        for i in self.data_taken:
            i.set()

    def iter_next(self):
        for i in self.data_ready:
            i.wait()
        if self.next_batch[0] is None:
            for i in self.next_batch:
                assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            for batch in self.next_batch:
                assert batch.pad == self.next_batch[0].pad, \
                    "Number of entry mismatches between iterators"
            self.current_batch = DataBatch(sum([batch.data for batch in self.next_batch], []),
                                           sum([batch.label for batch in self.next_batch], []),
                                           self.next_batch[0].pad,
                                           self.next_batch[0].index,
                                           provide_data=self.provide_data,
                                           provide_label=self.provide_label)
            for i in self.data_ready:
                i.clear()
            for i in self.data_taken:
                i.set()
            return True

    def next(self):
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad

def _init_data(data, allow_empty, default_name):
    """Convert data into canonical form."""
    assert (data is not None) or allow_empty
    if data is None:
        data = []

    if isinstance(data, (np.ndarray, NDArray, h5py.Dataset)
                  if h5py else (np.ndarray, NDArray)):
        data = [data]
    if isinstance(data, list):
        if not allow_empty:
            assert(len(data) > 0)
        if len(data) == 1:
            data = OrderedDict([(default_name, data[0])]) # pylint: disable=redefined-variable-type
        else:
            data = OrderedDict( # pylint: disable=redefined-variable-type
                [('_%d_%s' % (i, default_name), d) for i, d in enumerate(data)])
    if not isinstance(data, dict):
        raise TypeError("Input must be NDArray, numpy.ndarray, h5py.Dataset " + \
                "a list of them or dict with them as values")
    for k, v in data.items():
        if not isinstance(v, (NDArray, h5py.Dataset) if h5py else NDArray):
            try:
                data[k] = array(v)
            except:
                raise TypeError(("Invalid type '%s' for %s, "  % (type(v), k)) + \
                                "should be NDArray, numpy.ndarray or h5py.Dataset")

    return list(data.items())

def _has_instance(data, dtype):
    """Return True if ``data`` has instance of ``dtype``.
    This function is called after _init_data.
    ``data`` is a list of (str, NDArray)"""
    for item in data:
        _, arr = item
        if isinstance(arr, dtype):
            return True
    return False

def _shuffle(data, idx):
    """Shuffle the data."""
    shuffle_data = []

    for k, v in data:
        if (isinstance(v, h5py.Dataset) if h5py else False):
            shuffle_data.append((k, v))
        elif isinstance(v, CSRNDArray):
            shuffle_data.append((k, sparse_array(v.asscipy()[idx], v.context)))
        else:
            shuffle_data.append((k, array(v.asnumpy()[idx], v.context)))

    return shuffle_data

class NDArrayIter(DataIter):
    """Returns an iterator for ``mx.nd.NDArray``, ``numpy.ndarray``, ``h5py.Dataset``
    ``mx.nd.sparse.CSRNDArray`` or ``scipy.sparse.csr_matrix``.

    Example usage:
    ----------
    >>> data = np.arange(40).reshape((10,2,2))
    >>> labels = np.ones([10, 1])
    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, True, last_batch_handle='discard')
    >>> for batch in dataiter:
    ...     print batch.data[0].asnumpy()
    ...     batch.data[0].shape
    ...
    [[[ 36.  37.]
      [ 38.  39.]]
     [[ 16.  17.]
      [ 18.  19.]]
     [[ 12.  13.]
      [ 14.  15.]]]
    (3L, 2L, 2L)
    [[[ 32.  33.]
      [ 34.  35.]]
     [[  4.   5.]
      [  6.   7.]]
     [[ 24.  25.]
      [ 26.  27.]]]
    (3L, 2L, 2L)
    [[[  8.   9.]
      [ 10.  11.]]
     [[ 20.  21.]
      [ 22.  23.]]
     [[ 28.  29.]
      [ 30.  31.]]]
    (3L, 2L, 2L)
    >>> dataiter.provide_data # Returns a list of `DataDesc`
    [DataDesc[data,(3, 2L, 2L),<type 'numpy.float32'>,NCHW]]
    >>> dataiter.provide_label # Returns a list of `DataDesc`
    [DataDesc[softmax_label,(3, 1L),<type 'numpy.float32'>,NCHW]]

    In the above example, data is shuffled as `shuffle` parameter is set to `True`
    and remaining examples are discarded as `last_batch_handle` parameter is set to `discard`.

    Usage of `last_batch_handle` parameter:

    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, True, last_batch_handle='pad')
    >>> batchidx = 0
    >>> for batch in dataiter:
    ...     batchidx += 1
    ...
    >>> batchidx  # Padding added after the examples read are over. So, 10/3+1 batches are created.
    4
    >>> dataiter = mx.io.NDArrayIter(data, labels, 3, True, last_batch_handle='discard')
    >>> batchidx = 0
    >>> for batch in dataiter:
    ...     batchidx += 1
    ...
    >>> batchidx # Remaining examples are discarded. So, 10/3 batches are created.
    3

    `NDArrayIter` also supports multiple input and labels.

    >>> data = {'data1':np.zeros(shape=(10,2,2)), 'data2':np.zeros(shape=(20,2,2))}
    >>> label = {'label1':np.zeros(shape=(10,1)), 'label2':np.zeros(shape=(20,1))}
    >>> dataiter = mx.io.NDArrayIter(data, label, 3, True, last_batch_handle='discard')

    `NDArrayIter` also supports ``mx.nd.sparse.CSRNDArray``
    with `last_batch_handle` set to `discard`.

    >>> csr_data = mx.nd.array(np.arange(40).reshape((10,4))).tostype('csr')
    >>> labels = np.ones([10, 1])
    >>> dataiter = mx.io.NDArrayIter(csr_data, labels, 3, last_batch_handle='discard')
    >>> [batch.data[0] for batch in dataiter]
    [
    <CSRNDArray 3x4 @cpu(0)>,
    <CSRNDArray 3x4 @cpu(0)>,
    <CSRNDArray 3x4 @cpu(0)>]

    Parameters
    ----------
    data: array or list of array or dict of string to array
        The input data.
    label: array or list of array or dict of string to array, optional
        The input label.
    batch_size: int
        Batch size of data.
    shuffle: bool, optional
        Whether to shuffle the data.
        Only supported if no h5py.Dataset inputs are used.
    last_batch_handle : str, optional
        How to handle the last batch. This parameter can be 'pad', 'discard' or
        'roll_over'. 'roll_over' is intended for training and can cause problems
        if used for prediction.
    data_name : str, optional
        The data name.
    label_name : str, optional
        The label name.
    """
    def __init__(self, data, label=None, batch_size=1, shuffle=False,
                 last_batch_handle='pad', data_name='data',
                 label_name='softmax_label'):
        super(NDArrayIter, self).__init__(batch_size)

        self.data = _init_data(data, allow_empty=False, default_name=data_name)
        self.label = _init_data(label, allow_empty=True, default_name=label_name)

        if ((_has_instance(self.data, CSRNDArray) or _has_instance(self.label, CSRNDArray)) and
                (last_batch_handle != 'discard')):
            raise NotImplementedError("`NDArrayIter` only supports ``CSRNDArray``" \
                                      " with `last_batch_handle` set to `discard`.")

        self.idx = np.arange(self.data[0][1].shape[0])
        # shuffle data
        if shuffle:
            np.random.shuffle(self.idx)
            self.data = _shuffle(self.data, self.idx)
            self.label = _shuffle(self.label, self.idx)

        # batching
        if last_batch_handle == 'discard':
            new_n = self.data[0][1].shape[0] - self.data[0][1].shape[0] % batch_size
            self.idx = self.idx[:new_n]

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)
        self.num_data = self.idx.shape[0]
        assert self.num_data >= batch_size, \
            "batch_size needs to be smaller than data size."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator."""
        return [
            DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype)
            for k, v in self.data
        ]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        return [
            DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype)
            for k, v in self.label
        ]

    def hard_reset(self):
        """Ignore roll over data and set to start."""
        self.cursor = -self.batch_size

    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only."""
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            return [
                # np.ndarray or NDArray case
                x[1][self.cursor:self.cursor + self.batch_size]
                if isinstance(x[1], (np.ndarray, NDArray)) else
                # h5py (only supports indices in increasing order)
                array(x[1][sorted(self.idx[
                    self.cursor:self.cursor + self.batch_size])][[
                        list(self.idx[self.cursor:
                                      self.cursor + self.batch_size]).index(i)
                        for i in sorted(self.idx[
                            self.cursor:self.cursor + self.batch_size])
                    ]]) for x in data_source
            ]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return [
                # np.ndarray or NDArray case
                concatenate([x[1][self.cursor:], x[1][:pad]])
                if isinstance(x[1], (np.ndarray, NDArray)) else
                # h5py (only supports indices in increasing order)
                concatenate([
                    array(x[1][sorted(self.idx[self.cursor:])][[
                        list(self.idx[self.cursor:]).index(i)
                        for i in sorted(self.idx[self.cursor:])
                    ]]),
                    array(x[1][sorted(self.idx[:pad])][[
                        list(self.idx[:pad]).index(i)
                        for i in sorted(self.idx[:pad])
                    ]])
                ]) for x in data_source
            ]

    def getdata(self):
        return self._getdata(self.data)

    def getlabel(self):
        return self._getdata(self.label)

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0


class MXDataIter(DataIter):
    """A python wrapper a C++ data iterator.

    This iterator is the Python wrapper to all native C++ data iterators, such
    as `CSVIter`, `ImageRecordIter`, `MNISTIter`, etc. When initializing
    `CSVIter` for example, you will get an `MXDataIter` instance to use in your
    Python code. Calls to `next`, `reset`, etc will be delegated to the
    underlying C++ data iterators.

    Usually you don't need to interact with `MXDataIter` directly unless you are
    implementing your own data iterators in C++. To do that, please refer to
    examples under the `src/io` folder.

    Parameters
    ----------
    handle : DataIterHandle, required
        The handle to the underlying C++ Data Iterator.
    data_name : str, optional
        Data name. Default to "data".
    label_name : str, optional
        Label name. Default to "softmax_label".

    See Also
    --------
    src/io : The underlying C++ data iterator implementation, e.g., `CSVIter`.
    """
    def __init__(self, handle, data_name='data', label_name='softmax_label', **_):
        super(MXDataIter, self).__init__()
        self.handle = handle
        # debug option, used to test the speed with io effect eliminated
        self._debug_skip_load = False

        # load the first batch to get shape information
        self.first_batch = None
        self.first_batch = self.next()
        data = self.first_batch.data[0]
        label = self.first_batch.label[0]

        # properties
        self.provide_data = [DataDesc(data_name, data.shape, data.dtype)]
        self.provide_label = [DataDesc(label_name, label.shape, label.dtype)]
        self.batch_size = data.shape[0]

    def __del__(self):
        check_call(_LIB.MXDataIterFree(self.handle))

    def debug_skip_load(self):
        # Set the iterator to simply return always first batch. This can be used
        # to test the speed of network without taking the loading delay into
        # account.
        self._debug_skip_load = True
        logging.info('Set debug_skip_load to be true, will simply return first batch')

    def reset(self):
        self._debug_at_begin = True
        self.first_batch = None
        check_call(_LIB.MXDataIterBeforeFirst(self.handle))

    def next(self):
        if self._debug_skip_load and not self._debug_at_begin:
            return  DataBatch(data=[self.getdata()], label=[self.getlabel()], pad=self.getpad(),
                              index=self.getindex())
        if self.first_batch is not None:
            batch = self.first_batch
            self.first_batch = None
            return batch
        self._debug_at_begin = False
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        if next_res.value:
            return DataBatch(data=[self.getdata()], label=[self.getlabel()], pad=self.getpad(),
                             index=self.getindex())
        else:
            raise StopIteration

    def iter_next(self):
        if self.first_batch is not None:
            return True
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        return next_res.value

    def getdata(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetData(self.handle, ctypes.byref(hdl)))
        return _ndarray_cls(hdl, False)

    def getlabel(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetLabel(self.handle, ctypes.byref(hdl)))
        return _ndarray_cls(hdl, False)

    def getindex(self):
        index_size = ctypes.c_uint64(0)
        index_data = ctypes.POINTER(ctypes.c_uint64)()
        check_call(_LIB.MXDataIterGetIndex(self.handle,
                                           ctypes.byref(index_data),
                                           ctypes.byref(index_size)))
        if index_size.value:
            address = ctypes.addressof(index_data.contents)
            dbuffer = (ctypes.c_uint64* index_size.value).from_address(address)
            np_index = np.frombuffer(dbuffer, dtype=np.uint64)
            return np_index.copy()
        else:
            return None

    def getpad(self):
        pad = ctypes.c_int(0)
        check_call(_LIB.MXDataIterGetPadNum(self.handle, ctypes.byref(pad)))
        return pad.value

def _make_io_iterator(handle):
    """Create an io iterator by handle."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXDataIterGetIterInfo( \
            handle, ctypes.byref(name), ctypes.byref(desc), \
            ctypes.byref(num_args), \
            ctypes.byref(arg_names), \
            ctypes.byref(arg_types), \
            ctypes.byref(arg_descs)))
    iter_name = py_str(name.value)

    narg = int(num_args.value)
    param_str = _build_param_doc(
        [py_str(arg_names[i]) for i in range(narg)],
        [py_str(arg_types[i]) for i in range(narg)],
        [py_str(arg_descs[i]) for i in range(narg)])

    doc_str = ('%s\n\n' +
               '%s\n' +
               'Returns\n' +
               '-------\n' +
               'MXDataIter\n'+
               '    The result iterator.')
    doc_str = doc_str % (desc.value, param_str)

    def creator(*args, **kwargs):
        """Create an iterator.
        The parameters listed below can be passed in as keyword arguments.

        Parameters
        ----------
        name : string, required.
            Name of the resulting data iterator.

        Returns
        -------
        dataiter: Dataiter
            The resulting data iterator.
        """
        param_keys = []
        param_vals = []

        for k, val in kwargs.items():
            param_keys.append(k)
            param_vals.append(str(val))
        # create atomic symbol
        param_keys = c_str_array(param_keys)
        param_vals = c_str_array(param_vals)
        iter_handle = DataIterHandle()
        check_call(_LIB.MXDataIterCreateIter(
            handle,
            mx_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(iter_handle)))

        if len(args):
            raise TypeError('%s can only accept keyword arguments' % iter_name)

        return MXDataIter(iter_handle, **kwargs)

    creator.__name__ = iter_name
    creator.__doc__ = doc_str
    return creator

def _init_io_module():
    """List and add all the data iterators to current module."""
    plist = ctypes.POINTER(ctypes.c_void_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListDataIters(ctypes.byref(size), ctypes.byref(plist)))
    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = ctypes.c_void_p(plist[i])
        dataiter = _make_io_iterator(hdl)
        setattr(module_obj, dataiter.__name__, dataiter)

_init_io_module()
