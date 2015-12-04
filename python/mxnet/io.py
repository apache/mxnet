# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments, W0221, W0201

"""NDArray interface of mxnet"""
from __future__ import absolute_import
from collections import namedtuple, OrderedDict

import ctypes
import sys
import numpy as np
import logging
try:
    from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
except ImportError:
    pass
from .base import _LIB
from .base import c_array, c_str, mx_uint, py_str
from .base import DataIterHandle, NDArrayHandle
from .base import check_call, ctypes2docstring
from .ndarray import NDArray
from .ndarray import array
try:
    import cPickle as pickle
except ImportError:
    import pickle
from multiprocessing import Process, JoinableQueue
try:
    from Queue import Empty
except ImportError:
    from queue import Empty

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

    def getindex(self):
        """
        Retures
        -------
        index : numpy.array
            The index of current batch
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


DataBatch = namedtuple('DataBatch', ['data', 'label', 'pad', 'index'])

def _init_data(data, allow_empty, default_name):
    """Convert data into canonical form."""
    assert (data is not None) or allow_empty
    if data is None:
        data = []
    if isinstance(data, (np.ndarray, NDArray)):
        data = [data]
    if isinstance(data, list):
        if not allow_empty:
            assert(len(data) > 0)
        if len(data) == 1:
            data = OrderedDict([(default_name, data[0])])
        else:
            data = OrderedDict([('_%d_%s' % (i, default_name), d) for i, d in enumerate(data)])
    if not isinstance(data, dict):
        raise TypeError("Input must be NDArray, numpy.ndarray, " + \
                "a list of them or dict with them as values")
    for k, v in data.items():
        if isinstance(v, NDArray):
            data[k] = v.asnumpy()
    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            raise TypeError(("Invalid type '%s' for %s, "  % (type(v), k)) + \
                    "should be NDArray or numpy.ndarray")

    return list(data.items())

class NDArrayIter(DataIter):
    """NDArrayIter object in mxnet. Taking NDArray or numpy array to get dataiter.
    Parameters
    ----------
    data: NDArray or numpy.ndarray, a list of them, or a dict of string to them.
        NDArrayIter supports single or multiple data and label.
    label: NDArray or numpy.ndarray, a list of them, or a dict of them.
        Same as data, but is not fed to the model during testing.
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
    def __init__(self, data, label=None, batch_size=1, shuffle=False, last_batch_handle='pad'):
        # pylint: disable=W0201

        super(NDArrayIter, self).__init__()

        self.data = _init_data(data, allow_empty=False, default_name='data')
        self.label = _init_data(label, allow_empty=True, default_name='softmax_label')

        # shuffle data
        if shuffle:
            idx = np.arange(self.data[0][1].shape[0])
            np.random.shuffle(idx)
            self.data = [(k, v[idx]) for k, v in self.data]
            self.label = [(k, v[idx]) for k, v in self.label]

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)

        # batching
        if last_batch_handle == 'discard':
            new_n = self.data_list[0].shape[0] - self.data_list[0].shape[0] % batch_size
            for i, v in enumerate(self.data):
                self.data[i] = (v[0], v[1][:new_n])
            for i, v in enumerate(self.label):
                self.label[i] = (v[0], v[1][:new_n])
        self.num_data = self.data_list[0].shape[0]
        assert self.num_data >= batch_size, \
            "batch_size need to be smaller than data size when not padding."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]


    def hard_reset(self):
        """Igore roll over data and set to start"""
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
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            return [array(x[1][self.cursor:self.cursor+self.batch_size]) for x in data_source]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return [array(np.concatenate((x[1][self.cursor:], x[1][:pad]),
                                         axis=0)) for x in data_source]

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


class PickleIter(DataIter):
    """PickleIter object in mxnet. Taking path list of files which store data and label \
       using pickle to get data iter.
    Parameters
    ----------
    datapath: list of str
        list of file paths where data and label are stored in a dictionary with attributes \
        'data' and 'label'. The data and label should be NDArray, or numpy.ndarray, or list \
        of them or dict with them as values.
    max_queue: int
        max queue buffer size
    batch_size: int
        Batch Size
    shuffle: bool
        Whether to shuffle the data in one data file
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
    eof_flag = False
    queue = None

    class pickle_file_processor(Process):
        """Process to read data from files using pickle"""
        def __init__(self, num_datafile, datapath):
            Process.__init__(self)
            self.num_datafile = num_datafile
            self.datapath = datapath
        def run(self):
            file_cursor = -1
            while True:
                file_cursor += 1
                if file_cursor < self.num_datafile:
                    with open(self.datapath[file_cursor]) as f:
                        PickleIter.queue.put(pickle.load(f))
                else:
                    PickleIter.eof_flag = True
                    break

    def get_data_from_new_file(self):
        """Load data from underlying arrays, internal use only"""
        d = self.queue.get()
        self.queue.task_done()
        label = d['label'].toarray() if d['label'] != None else None
        data = d['data']
        try:
            if isinstance(data, csr_matrix) or isinstance(data, coo_matrix) \
               or isinstance(data, csc_matrix):
                data = data.toarray()
        except NameError:
            pass
        self.data = _init_data(data, allow_empty=False, default_name='data')
        self.label = _init_data(label, allow_empty=True, default_name='softmax_label')

        # shuffle data
        if self.shuffle:
            idx = np.arange(self.data[0][1].shape[0])
            np.random.shuffle(idx)
            self.data = [(k, v[idx]) for k, v in self.data]
            self.label = [(k, v[idx]) for k, v in self.label]

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)

        # batching
        if self.last_batch_handle == 'discard':
            new_n = self.data_list[0].shape[0] - self.data_list[0].shape[0] % self.batch_size
            for k, v in enumerate(self.data):
                self.data[k] = (v[0], v[1][:new_n])
            for k, v in enumerate(self.label):
                self.label[k] = (v[0], v[1][:new_n])
        self.num_data = self.data_list[0].shape[0]

        assert self.num_data >= self.batch_size, \
            "batch_size need to be smaller than data size when not padding."
        self.cursor = -self.batch_size

    def __init__(self, datapath, max_queue=1, batch_size=1, shuffle=False, last_batch_handle='pad'):
        # pylint: disable=W0201

        super(PickleIter, self).__init__()

        self.batch_size = batch_size
        PickleIter.queue = JoinableQueue(max_queue)
        self.datapath = [x for x in datapath if x]
        self.num_datafile = len(self.datapath)
        self.shuffle = shuffle
        self.last_batch_handle = last_batch_handle
        self.thread = self.pickle_file_processor(self.num_datafile, self.datapath)
        self.thread.start()
        self.get_data_from_new_file()

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]

    def hard_reset(self):
        """Igore roll over data and set to start"""
        self.cursor = -self.batch_size

    def reset(self):
        self.thread.terminate()
        while not self.queue.empty():
            try:
                self.queue.get(False)
            except Empty:
                self.queue.task_done()
        self.thread = self.pickle_file_processor(self.num_datafile, self.datapath)
        self.thread.start()
        self.get_data_from_new_file()
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor % self.num_data) % self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return 1 # next batch
        elif not self.eof_flag or not self.queue.empty():
            return 0 # next data file
        else:
            return -1 # end of data

    def next(self):
        if self.iter_next() == 1: # next batch
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        elif self.iter_next() == -1: # end of data
            raise StopIteration
        else: # next data file
            self.get_data_from_new_file()
            return self.next()

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        if self.cursor + self.batch_size <= self.num_data:
            return [array(x[1][self.cursor:self.cursor + self.batch_size]) for x in data_source]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return [array(np.concatenate((x[1][self.cursor:], x[1][:pad]),
                                         axis=0)) for x in data_source]
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
    """DataIter built in MXNet. List all the needed functions here.
    Parameters
    ----------
    handle : DataIterHandle
        the handle to the underlying C++ Data Iterator
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
        self.provide_data = [(data_name, data.shape)]
        self.provide_label = [(label_name, label.shape)]
        self.batch_size = data.shape[0]


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
        return NDArray(hdl, False)

    def getlabel(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetLabel(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl, False)

    def getindex(self):
        index_size = ctypes.c_uint64(0)
        index_data = ctypes.POINTER(ctypes.c_uint64)()
        check_call(_LIB.MXDataIterGetIndex(self.handle,
                                           ctypes.byref(index_data),
                                           ctypes.byref(index_size)))
        address = ctypes.addressof(index_data.contents)
        dbuffer = (ctypes.c_uint64* index_size.value).from_address(address)
        np_index = np.frombuffer(dbuffer, dtype=np.uint64)
        return np_index.copy()

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

# Initialize the io in startups
_init_io_module()
