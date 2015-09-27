# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments

"""NDArray interface of mxnet"""
from __future__ import absolute_import

import ctypes
import sys
import numpy as np
import math
import logging
from .base import _LIB
from .base import c_array, c_str, mx_uint, py_str
from .base import DataIterHandle, NDArrayHandle
from .base import check_call, ctypes2docstring
from .ndarray import NDArray
from .ndarray import array

class DataIter(object):
    """DataIter object in mxnet. """

    def __init__(self):
        pass

    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator. """
        pass

    def next(self):
        """Get next data batch from iterator

        Returns
        -------
        data : NDArray
            The data of next batch.

        label : NDArray
            The label of next batch.
        """
        pass

    def __next__(self):
        return self.next()

    def iter_next(self):
        """Iterate to next batch.

        Returns
        -------
        has_next : boolean
            Whether the move is successful.
        """
        pass

    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        data : NDArray
            The data of current batch.
        """
        pass

    def getlabel(self):
        """Get label of current batch.

        Returns
        -------
        label : NDArray
            The label of current batch.
        """
        pass

    def getpad(self):
        """Get the number of padding examples in current batch.

        Returns
        -------
        pad : int
            Number of padding examples in current batch
        """
        pass


class NDArrayIter(DataIter):
    """NDArrayIter object in mxnet. Taking NDArray or numpy array to get dataiter.

    Parameters
    ----------
    data : NDArray or numpy.ndarray
        NDArray for data

    label : NDArray or numpy.ndarray
        NDArray for label

    batch_size: int
        Batch Size

    shuffle: bool
        Whether to shuffle the data

    data_pad_value: float, optional
        Padding value for data

    label_pad_value: float, optionl
        Padding value for label

    Note
    ----
    This iterator will pad the last batch if
    the size of data does not match batch_size.
    """
    def __init__(self, data, label,
                 batch_size,
                 shuffle=False,
                 data_pad_value=0,
                 label_pad_value=0):
        super(NDArrayIter, self).__init__()
        if isinstance(data, NDArray):
            data = data.asnumpy()
        if isinstance(label, NDArray):
            label = label.asnumpy()
        # shuffle data
        if shuffle:
            idx = np.arange(data.shape[0])
            np.random.shuffle(idx)
            new_data = np.zeros(data.shape)
            new_label = np.zeros(label.shape)
            for i in range(data.shape[0]):
                new_data[i] = data[idx[i]]
                new_label[i] = label[idx[i]]
            data = new_data
            label = new_label

        # batching
        self.batch_num = int(math.ceil(float(data.shape[0]) / batch_size))
        batch_data_shape = []
        batch_data_shape.append(self.batch_num)
        batch_data_shape.append(batch_size)
        for i in range(1, len(data.shape)):
            batch_data_shape.append(data.shape[i])
        batch_label_shape = []
        batch_label_shape.append(self.batch_num)
        batch_label_shape.append(batch_size)
        for i in range(1, len(label.shape)):
            batch_label_shape.append(label.shape[i])
        self.batch_data = np.ones(batch_data_shape, dtype='float32') * data_pad_value
        self.batch_label = np.ones(batch_label_shape, dtype='float32') * label_pad_value
        loc = 0
        for i in range(self.batch_num):
            actual_size = min(data.shape[0] - loc, batch_size)
            self.batch_data[i, 0:actual_size, ::] = data[loc:loc+actual_size, ::]
            self.batch_label[i, 0:actual_size] = label[loc:loc+actual_size]
            loc += batch_size
        if data.shape[0] > batch_size:
            self.num_pad = data.shape[0] % batch_size
        else:
            self.num_pad = batch_size - data.shape[0]
        self.out_data = None
        self.out_label = None
        self.current_batch = -1

    def reset(self):
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

    def getpad(self):
        if self.current_batch == self.batch_num - 1:
            return self.num_pad
        else:
            return 0


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
        # debug option, used to test the speed with io effect eliminated
        self._debug_skip_load = False
        self._debug_at_begin = True

    def __del__(self):
        check_call(_LIB.MXDataIterFree(self.handle))

    def debug_skip_load(self):
        """Set the iterator to simply return always first batch.

        Notes
        -----
        This can be used to test the speed of network without taking
        the loading delay into account.
        """
        self._debug_skip_load = True
        logging.info('Set debug_skip_load to be true, will simply return first batch')

    def reset(self):
        self._debug_at_begin = True
        check_call(_LIB.MXDataIterBeforeFirst(self.handle))

    def next(self):
        if self._debug_skip_load and not self._debug_at_begin:
            return  self.getdata(), self.getlabel()
        self._debug_at_begin = False
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
            handle,
            mx_uint(len(param_keys)),
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
