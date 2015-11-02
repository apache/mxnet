# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments, W0221, W0201

"""NDArray interface of mxnet"""
from __future__ import absolute_import

import ctypes
import sys
import numpy as np
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

    def getdata(self, index=0):
        """Get data of current batch.

        Parameters
        ----------
        index : int
            The index of data source to retrieve.

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
        return self.getdata(-1)

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
    data_list or data, label: a list of, or two separate NDArray or numpy.ndarray
        list of NDArray for data. The last one is treated as label.

    batch_size: int
        Batch Size

    shuffle: bool
        Whether to shuffle the data

    data_pad_value: float, optional
        Padding value for data

    label_pad_value: float, optionl
        Padding value for label

    last_batch_handle: 'pad', 'discard' or 'roll_over'
        How to handle the last batch

    Note
    ----
    This iterator will pad, discard or roll over the last batch if
    the size of data does not match batch_size. Roll over is intended
    for training and can cause problems if used for prediction.
    """
    def __init__(self, *args, **kwargs):
        super(NDArrayIter, self).__init__()
        if isinstance(args[0], list) or isinstance(args[0], tuple):
            self._init(*args, **kwargs)
        else:
            self._init((args[0], args[1]), *args[2:], **kwargs)

    def _init(self, data_list,
              batch_size=1,
              shuffle=False,
              data_pad_value=0,
              label_pad_value=0,
              last_batch_handle='pad'):
        """Actual constructor"""
        # pylint: disable=W0201
        self.num_source = len(data_list)
        assert self.num_source > 0, "Need at least one data source."
        data_list = list(data_list)
        for i in range(self.num_source):
            if isinstance(data_list[i], NDArray):
                data_list[i] = data_list[i].asnumpy()
        # shuffle data
        if shuffle:
            idx = np.arange(data_list[0].shape[0])
            np.random.shuffle(idx)
            for i in range(self.num_source):
                assert data_list[i].shape[0] == len(idx)
                data_list[i] = data_list[i][idx]

        # batching
        if last_batch_handle == 'discard':
            new_n = data_list[0].shape[0] - data_list[0].shape[0] % batch_size
            for i in range(self.num_source):
                data_list[i] = data_list[i][:new_n]
        self.num_data = data_list[0].shape[0]
        assert self.num_data > batch_size, \
            "batch_size need to be smaller than data size when not padding."
        self.cursor = -batch_size
        self.data_list = data_list
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    def hard_reset(self):
        self.cursor = -self.batch_size

    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            return (self.getdata(i) for i in range(self.num_source))
        else:
            raise StopIteration

    def getdata(self, index=0):
        assert(index < self.num_source)
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            return array(self.data_list[index][self.cursor:self.cursor+self.batch_size])
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return array(np.concatenate((self.data_list[index][self.cursor:],
                                         self.data_list[index][:pad]),
                                        axis=0))

    def getlabel(self):
        return self.getdata(-1)

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
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
