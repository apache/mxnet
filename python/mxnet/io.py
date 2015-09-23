# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments

"""NDArray interface of mxnet"""
from __future__ import absolute_import

import ctypes
import sys
import numpy as np
import math
from .base import _LIB
from .base import c_array, c_str, mx_uint, py_str
from .base import DataIterHandle, NDArrayHandle
from .base import check_call, ctypes2docstring
from .ndarray import NDArray
from .ndarray import array

class DataIter(object):
    """DataIter object in mxnet. """

    def __init__(self):
        """constructor of dataiter

        """
        pass

    def __del__(self):
        """destructor of dataiter

        """
        pass

    def __iter__(self):
        """make the class iterable

        """
        return self

    def reset(self):
        """reset the iter

        """
        pass

    def next(self):
        """get next data batch from iterator

        Returns
        -------
        labels and images for the next batch
        """
        pass

    # make it work for both python2 and 3
    __next__ = next

    def iter_next(self):
        """iterate to next data with return value

        Returns
        -------
        return true if success
        """
        pass

    def getdata(self):
        """get data from batch

        Returns
        -------
        data ndarray for the next batch
        """
        pass

    def getlabel(self):
        """get label from batch

        Returns
        -------
        label ndarray for the next batch
        """
        pass

class NumpyIter(DataIter):
    """NumpyIter object in mxnet. Taking Numpy Array to get dataiter.

    Parameters
    ----------
    data : numpy.array
        Numpy ndarray for data
    label : numpy.array
        Numpy ndarray for label
    batch_size: int
        Batch Size
    shuffle: bool
        Whether to shuffle the data
    data_pad: float
        padding value for data
    label_pad: float
        padding value for label
    """
    def __init__(self, data, label, batch_size, shuffle=True, data_pad=0, label_pad=0):
        super(NumpyIter, self).__init__()
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_pad = data_pad
        self.label_pad = label_pad
        # shuffle data
        if self.shuffle:
            idx = np.arange(self.data.shape[0])
            np.random.shuffle(idx)
            new_data = np.zeros(self.data.shape)
            new_label = np.zeros(self.label.shape)
            for i in range(self.data.shape[0]):
                new_data[i] = self.data[idx[i]]
                new_label[i] = self.label[idx[i]]
            self.data = new_data
            self.label = new_label
        # batching
        self.batch_num = int(math.ceil(float(self.data.shape[0]) / self.batch_size))
        batch_data_shape = []
        batch_data_shape.append(self.batch_num)
        batch_data_shape.append(self.batch_size)
        for i in range(1, len(self.data.shape)):
            batch_data_shape.append(self.data.shape[i])
        batch_label_shape = []
        batch_label_shape.append(self.batch_num)
        batch_label_shape.append(self.batch_size)
        for i in range(1, len(self.label.shape)):
            batch_label_shape.append(self.label.shape[i])
        self.batch_data = np.ones(batch_data_shape, dtype=self.data.dtype) * self.data_pad
        self.batch_label = np.ones(batch_label_shape, dtype=self.label.dtype) * self.label_pad
        self.loc = 0
        for i in range(self.batch_num):
            actual_size = min(self.data.shape[0] - self.loc, self.batch_size)
            self.batch_data[i, 0:actual_size, ::] = self.data[self.loc:self.loc+actual_size, ::]
            self.batch_label[i, 0:actual_size, ::] = self.label[self.loc:self.loc+actual_size, ::]
            self.loc += self.batch_size
        self.out_data = None
        self.out_label = None
        self.current_batch = -1

    def reset(self):
        """set current batch to 0

        """
        self.current_batch = -1

    def iter_next(self):
        if self.current_batch < self.batch_num - 1:
            self.current_batch += 1
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            return self.getdata(), self.getlabel()
        else:
            raise StopIteration

    def getdata(self):
        assert(self.current_batch >= 0)
        return array(self.batch_data[self.current_batch])

    def getlabel(self):
        assert(self.current_batch >= 0)
        return array(self.batch_label[self.current_batch])

class MXDataIter(DataIter):
    """DataIter built in MXNet. List all the needed functions here.

    Parameters
    ----------
    handle : DataIterHandle
        the handle to the underlying C++ Data Iterator
    """
    def __init__(self, handle):
        super(MXDataIter, self).__init__()
        self.handle = handle

    def __del__(self):
        check_call(_LIB.MXDataIterFree(self.handle))

    def reset(self):
        check_call(_LIB.MXDataIterBeforeFirst(self.handle))

    def next(self):
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        if next_res.value:
            return self.getdata(), self.getlabel()
        else:
            raise StopIteration

    def iter_next(self):
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        return next_res.value

    def getdata(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetData(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl, False)

    def getlabel(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetLabel(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl, False)

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
    param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)

    doc_str = ('%s\n\n' +
               '%s\n' +
               'name : string, required.\n' +
               '    Name of the resulting data iterator.\n\n' +
               'Returns\n' +
               '-------\n' +
               'iterator: DataIter\n'+
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
            the resulting data iterator
        """
        param_keys = []
        param_vals = []

        for k, val in kwargs.items():
            param_keys.append(c_str(k))
            param_vals.append(c_str(str(val)))
        # create atomic symbol
        param_keys = c_array(ctypes.c_char_p, param_keys)
        param_vals = c_array(ctypes.c_char_p, param_vals)
        iter_handle = DataIterHandle()
        check_call(_LIB.MXDataIterCreateIter(
            handle, len(param_keys),
            param_keys, param_vals,
            ctypes.byref(iter_handle)))

        if len(args):
            raise TypeError('%s can only accept keyword arguments' % iter_name)

        return MXDataIter(iter_handle)

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

# Initialize the io in startups
_init_io_module()
