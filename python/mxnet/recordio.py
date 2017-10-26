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

"""Read and write for the RecordIO data format."""
from __future__ import absolute_import
from collections import namedtuple

import sys
import ctypes
import struct
import numbers
import numpy as np
from . import ndarray as nd

from .base import _LIB
from .base import RecordIOHandle, RecordIterHandle, NDArrayHandle
from .base import check_call, ctypes2buffer, build_param_doc as _build_param_doc
from .base import c_str, c_array, mx_uint, py_str
try:
    import cv2
except ImportError:
    cv2 = None

class MXRecordIO(object):
    """Reads/writes `RecordIO` data format, supporting sequential read and write.

    Example usage:
    ----------
    >>> record = mx.recordio.MXRecordIO('tmp.rec', 'w')
    <mxnet.recordio.MXRecordIO object at 0x10ef40ed0>
    >>> for i in range(5):
    ...    record.write('record_%d'%i)
    >>> record.close()
    >>> record = mx.recordio.MXRecordIO('tmp.rec', 'r')
    >>> for i in range(5):
    ...    item = record.read()
    ...    print(item)
    record_0
    record_1
    record_2
    record_3
    record_4
    >>> record.close()

    Parameters
    ----------
    uri : string
        Path to the record file.
    flag : string
        'w' for write or 'r' for read.
    """
    def __init__(self, uri, flag):
        self.uri = c_str(uri)
        self.handle = RecordIOHandle()
        self.flag = flag
        self.is_open = False
        self.open()

    def open(self):
        """Opens the record file."""
        if self.flag == "w":
            check_call(_LIB.MXRecordIOWriterCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = True
        elif self.flag == "r":
            check_call(_LIB.MXRecordIOReaderCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = False
        else:
            raise ValueError("Invalid flag %s"%self.flag)
        self.is_open = True

    def __del__(self):
        self.close()

    def close(self):
        """Closes the record file."""
        if not self.is_open:
            return
        if self.writable:
            check_call(_LIB.MXRecordIOWriterFree(self.handle))
        else:
            check_call(_LIB.MXRecordIOReaderFree(self.handle))
        self.is_open = False

    def reset(self):
        """Resets the pointer to first item.

        If the record is opened with 'w', this function will truncate the file to empty.

        Example usage:
        ----------
        >>> record = mx.recordio.MXRecordIO('tmp.rec', 'r')
        >>> for i in range(2):
        ...    item = record.read()
        ...    print(item)
        record_0
        record_1
        >>> record.reset()  # Pointer is reset.
        >>> print(record.read()) # Started reading from start again.
        record_0
        >>> record.close()
        """
        self.close()
        self.open()

    def write(self, buf):
        """Inserts a string buffer as a record.

        Example usage:
        ----------
        >>> record = mx.recordio.MXRecordIO('tmp.rec', 'w')
        >>> for i in range(5):
        ...    record.write('record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        buf : string (python2), bytes (python3)
            Buffer to write.
        """
        assert self.writable
        check_call(_LIB.MXRecordIOWriterWriteRecord(self.handle,
                                                    ctypes.c_char_p(buf),
                                                    ctypes.c_size_t(len(buf))))

    def read(self):
        """Returns record as a string.

        Example usage:
        ----------
        >>> record = mx.recordio.MXRecordIO('tmp.rec', 'r')
        >>> for i in range(5):
        ...    item = record.read()
        ...    print(item)
        record_0
        record_1
        record_2
        record_3
        record_4
        >>> record.close()

        Returns
        ----------
        buf : string
            Buffer read.
        """
        assert not self.writable
        buf = ctypes.c_char_p()
        size = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOReaderReadRecord(self.handle,
                                                   ctypes.byref(buf),
                                                   ctypes.byref(size)))
        if buf:
            buf = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char*size.value))
            return buf.contents.raw
        else:
            return None

class MXIndexedRecordIO(MXRecordIO):
    """Reads/writes `RecordIO` data format, supporting random access.

    Example usage:
    ----------
    >>> for i in range(5):
    ...     record.write_idx(i, 'record_%d'%i)
    >>> record.close()
    >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
    >>> record.read_idx(3)
    record_3

    Parameters
    ----------
    idx_path : str
        Path to the index file.
    uri : str
        Path to the record file. Only supports seekable file types.
    flag : str
        'w' for write or 'r' for read.
    key_type : type
        Data type for keys.
    """
    def __init__(self, idx_path, uri, flag, key_type=int):
        self.idx_path = idx_path
        self.idx = {}
        self.keys = []
        self.key_type = key_type
        self.fidx = None
        super(MXIndexedRecordIO, self).__init__(uri, flag)

    def open(self):
        super(MXIndexedRecordIO, self).open()
        self.idx = {}
        self.keys = []
        self.fidx = open(self.idx_path, self.flag)
        if not self.writable:
            for line in iter(self.fidx.readline, ''):
                line = line.strip().split('\t')
                key = self.key_type(line[0])
                self.idx[key] = int(line[1])
                self.keys.append(key)

    def close(self):
        """Closes the record file."""
        if not self.is_open:
            return
        super(MXIndexedRecordIO, self).close()
        self.fidx.close()

    def seek(self, idx):
        """Sets the current read pointer position.

        This function is internally called by `read_idx(idx)` to find the current
        reader pointer position. It doesn't return anything."""
        assert not self.writable
        pos = ctypes.c_size_t(self.idx[idx])
        check_call(_LIB.MXRecordIOReaderSeek(self.handle, pos))

    def tell(self):
        """Returns the current position of write head.

        Example usage:
        ----------
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> print(record.tell())
        0
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        ...     print(record.tell())
        16
        32
        48
        64
        80
        """
        assert self.writable
        pos = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOWriterTell(self.handle, ctypes.byref(pos)))
        return pos.value

    def read_idx(self, idx):
        """Returns the record at given index.

        Example usage:
        ----------
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
        >>> record.read_idx(3)
        record_3
        """
        self.seek(idx)
        return self.read()

    def write_idx(self, idx, buf):
        """Inserts input record at given index.

        Example usage:
        ----------
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        idx : int
            Index of a file.
        buf :
            Record to write.
        """
        key = self.key_type(idx)
        pos = self.tell()
        self.write(buf)
        self.fidx.write('%s\t%d\n'%(str(key), pos))
        self.idx[key] = pos
        self.keys.append(key)


IRHeader = namedtuple('HEADER', ['flag', 'label', 'id', 'id2'])
"""An alias for HEADER. Used to store metadata (e.g. labels) accompanying a record.
See mxnet.recordio.pack and mxnet.recordio.pack_img for example uses.

Parameters
----------
    flag : int
        Available for convenience, can be set arbitrarily.
    label : float or an array of float
        Typically used to store label(s) for a record.
    id: int
        Usually a unique id representing record.
    id2: int
        Higher order bits of the unique id, should be set to 0 (in most cases).
"""
_IR_FORMAT = 'IfQQ'
_IR_SIZE = struct.calcsize(_IR_FORMAT)

def pack(header, s):
    """Pack a string into MXImageRecord.

    Parameters
    ----------
    header : IRHeader
        Header of the image record.
        ``header.label`` can be a number or an array. See more detail in ``IRHeader``.
    s : str
        Raw image string to be packed.

    Returns
    -------
    s : str
        The packed string.

    Examples
    --------
    >>> label = 4 # label can also be a 1-D array, for example: label = [1,2,3]
    >>> id = 2574
    >>> header = mx.recordio.IRHeader(0, label, id, 0)
    >>> with open(path, 'r') as file:
    ...     s = file.read()
    >>> packed_s = mx.recordio.pack(header, s)
    """
    header = IRHeader(*header)
    if isinstance(header.label, numbers.Number):
        header = header._replace(flag=0)
    else:
        label = np.asarray(header.label, dtype=np.float32)
        header = header._replace(flag=label.size, label=0)
        s = label.tostring() + s
    s = struct.pack(_IR_FORMAT, *header) + s
    return s

def unpack(s):
    """Unpack a MXImageRecord to string.

    Parameters
    ----------
    s : str
        String buffer from ``MXRecordIO.read``.

    Returns
    -------
    header : IRHeader
        Header of the image record.
    s : str
        Unpacked string.

    Examples
    --------
    >>> record = mx.recordio.MXRecordIO('test.rec', 'r')
    >>> item = record.read()
    >>> header, s = mx.recordio.unpack(item)
    >>> header
    HEADER(flag=0, label=14.0, id=20129312, id2=0)
    """
    header = IRHeader(*struct.unpack(_IR_FORMAT, s[:_IR_SIZE]))
    s = s[_IR_SIZE:]
    if header.flag > 0:
        header = header._replace(label=np.fromstring(s, np.float32, header.flag))
        s = s[header.flag*4:]
    return header, s

def unpack_img(s, iscolor=-1):
    """Unpack a MXImageRecord to image.

    Parameters
    ----------
    s : str
        String buffer from ``MXRecordIO.read``.
    iscolor : int
        Image format option for ``cv2.imdecode``.

    Returns
    -------
    header : IRHeader
        Header of the image record.
    img : numpy.ndarray
        Unpacked image.

    Examples
    --------
    >>> record = mx.recordio.MXRecordIO('test.rec', 'r')
    >>> item = record.read()
    >>> header, img = mx.recordio.unpack_img(item)
    >>> header
    HEADER(flag=0, label=14.0, id=20129312, id2=0)
    >>> img
    array([[[ 23,  27,  45],
            [ 28,  32,  50],
            ...,
            [ 36,  40,  59],
            [ 35,  39,  58]],
           ...,
           [[ 91,  92, 113],
            [ 97,  98, 119],
            ...,
            [168, 169, 167],
            [166, 167, 165]]], dtype=uint8)
    """
    header, s = unpack(s)
    img = np.fromstring(s, dtype=np.uint8)
    assert cv2 is not None
    img = cv2.imdecode(img, iscolor)
    return header, img

def pack_img(header, img, quality=95, img_fmt='.jpg'):
    """Pack an image into ``MXImageRecord``.

    Parameters
    ----------
    header : IRHeader
        Header of the image record.
        ``header.label`` can be a number or an array. See more detail in ``IRHeader``.
    img : numpy.ndarray
        Image to be packed.
    quality : int
        Quality for JPEG encoding in range 1-100, or compression for PNG encoding in range 1-9.
    img_fmt : str
        Encoding of the image (.jpg for JPEG, .png for PNG).

    Returns
    -------
    s : str
        The packed string.

    Examples
    --------
    >>> label = 4 # label can also be a 1-D array, for example: label = [1,2,3]
    >>> id = 2574
    >>> header = mx.recordio.IRHeader(0, label, id, 0)
    >>> img = cv2.imread('test.jpg')
    >>> packed_s = mx.recordio.pack_img(header, img)
    """
    assert cv2 is not None
    jpg_formats = ['.JPG', '.JPEG']
    png_formats = ['.PNG']
    encode_params = None
    if img_fmt.upper() in jpg_formats:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif img_fmt.upper() in png_formats:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]

    ret, buf = cv2.imencode(img_fmt, img, encode_params)
    assert ret, 'failed to encode image'
    return pack(header, buf.tostring())
# pb version of recordio
RecordIOHead = namedtuple('HEAD', ['id', 'reserve'])
RecordIOData = namedtuple('DATA', ['id', 'type', 'nd_value', 'str_value'])
RecordIOExtraData = namedtuple('EXTRA_DATA', ['key', 'type', 'nd_value', 'str_value'])
class RecordIOType(object):
    NDARRAY = 0
    BINARY = 1
    STRING = 2

class MXRecordIO_PB(object):
    """ a recordio from pb-format files.

    Parameters
    ----------
    data : list of `RecordIOData`
          A list of input data.
    label : list of `float`
          A list of input label.
    header : RecordIOHead
          head of input record
    extra_data : RecordIOExtraData
          A list of input extra data.
    """
    def __init__(self, data, label=None, header=None, extra_data=None):
        if data is not None:
            assert isinstance(data, list), "Data must be list"
        if label is not None:
            assert isinstance(label, list), "Label must be list"
        if extra_data is not None:
            assert isinstance(extra_data, list), "extra_data must be list"
        self.data = data
        self.label = label
        self.header = header
        self.extra_data = extra_data

class MXRecordIter(object):
    """A python wrapper a C++ data iterator.

    This iterator is the Python wrapper to native C++ RecordIter iterator.
    When initializing `RecordIter` for example, you will get an `MXRecordIter`
    instance to use in your Python code.
    Calls to `next`, `reset`, etc will be delegated to the
    underlying C++ data iterators.

    Usually you don't need to interact with `MXRecordIter` directly unless you are
    implementing your own data iterators in C++. To do that, please refer to
    examples under the `src/io` folder.

    Parameters
    ----------
    handle : DataIterHandle, required
        The handle to the underlying C++ Data Iterator.

    See Also
    --------
    src/io : The underlying C++ data iterator implementation.
    """
    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        check_call(_LIB.MXRecordIterFree(self.handle))
        self.handle = RecordIterHandle()

    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator to the begin of the data."""
        check_call(_LIB.MXRecordIterBeforeFirst(self.handle))

    def next(self):
        if self.iter_next():
            return MXRecordIO_PB(data = self.get_data(),
                    label = self.get_label(),
                    header = self.get_head(),
                    extra_data = self.get_extra_data())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        """Move to the next record.

        Returns
        -------
        boolean
            Whether the move is successful.
        """
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXRecordIterNext(self.handle, ctypes.byref(next_res)))
        return next_res.value

    def get_data(self):
        """Get data of current record.
        """
        num = self.get_data_num()
        data_list = []
        for i in xrange(num):
            data_id = ctypes.c_uint64(0)
            data_type = ctypes.c_int(0)
            data_ndarray = NDArrayHandle()
            data_string_length = mx_uint()
            data_string = ctypes.POINTER(ctypes.c_char)()
            check_call(_LIB.MXRecordIterGetData(self.handle,
                ctypes.c_uint(i),
                ctypes.byref(data_id),
                ctypes.byref(data_type),
                ctypes.byref(data_ndarray),
                ctypes.byref(data_string_length),
                ctypes.byref(data_string)))
            nd_value = nd.NDArray(data_ndarray, False)
            data_list.append(RecordIOData(id=data_id.value, type=data_type.value,
                nd_value=nd_value,
                str_value=ctypes2buffer(data_string, data_string_length.value)))
        return data_list

    def get_extra_data(self):
        """Get extra data of current record.
        """
        num = self.get_extra_data_num()
        data_list = []
        for i in xrange(num):
            data_key = ctypes.POINTER(ctypes.c_char)()
            data_key_length = mx_uint()
            data_type = ctypes.c_int(0)
            data_ndarray = NDArrayHandle()
            data_string = ctypes.POINTER(ctypes.c_char)()
            data_string_length = mx_uint()
            check_call(_LIB.MXRecordIterGetExtraData(self.handle,
                ctypes.c_uint(i),
                ctypes.byref(data_key),
                ctypes.byref(data_key_length),
                ctypes.byref(data_type),
                ctypes.byref(data_ndarray),
                ctypes.byref(data_string),
                ctypes.byref(data_string_length)))
            nd_value = nd.NDArray(data_ndarray, False)
            data_list.append(RecordIOExtraData(key=ctypes2buffer(data_key, data_key_length.value),
                type=data_type.value,
                nd_value=nd_value,
                str_value=ctypes2buffer(data_string, data_string_length.value)))
        return data_list

    def get_data_num(self):
        """Get data num of current record.
        """
        num = ctypes.c_uint(0)
        check_call(_LIB.MXRecordIterGetDataNum(self.handle, ctypes.byref(num)))
        return num.value

    def get_extra_data_num(self):
        """Get extra data num of current record.
        """
        num = ctypes.c_uint(0)
        check_call(_LIB.MXRecordIterGetExtraDataNum(self.handle, ctypes.byref(num)))
        return num.value

    def get_label(self):
        """Get label of the current record.
        """
        label_size = ctypes.c_uint(0)
        label_data = ctypes.POINTER(ctypes.c_float)()
        check_call(_LIB.MXRecordIterGetLabel(self.handle,
            ctypes.byref(label_data),
            ctypes.byref(label_size)))
        label_list = [label_data[i] for i in xrange(label_size.value)]
        return label_list

    def get_head(self):
        """Get head of the current record.
        """
        record_id = ctypes.c_uint64(0)
        reserve = ctypes.c_uint64(0)
        check_call(_LIB.MXRecordIterGetHead(self.handle,
            ctypes.byref(record_id), ctypes.byref(reserve)))
        return RecordIOHead(id=record_id.value, reserve=reserve.value)

def _make_record_iterator():
    """Create an io iterator by handle."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXRecordIterGetIterInfo( \
            ctypes.byref(name), ctypes.byref(desc), \
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
               'MXRecordIter\n'+
               '    The result iterator.')
    doc_str = doc_str % (desc.value, param_str)

    def creator(*args, **kwargs):
        """Create an iterator.
        The parameters listed below can be passed in as keyword arguments.

        Parameters
        ----------
        kwargs : ...
            arguments for creating MXRecordIter. See MXRecordIterCreate.

        Returns
        -------
        MXRecordIter
            The resulting data iterator.
        """
        param_keys = []
        param_vals = []

        for k, val in kwargs.items():
            param_keys.append(c_str(k))
            param_vals.append(c_str(str(val)))
        # create atomic symbol
        param_keys = c_array(ctypes.c_char_p, param_keys)
        param_vals = c_array(ctypes.c_char_p, param_vals)
        iter_handle = RecordIterHandle()
        check_call(_LIB.MXRecordIterCreate(
            mx_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(iter_handle)))

        if len(args):
            raise TypeError('%s can only accept keyword arguments' % iter_name)

        return MXRecordIter(iter_handle)

    creator.__name__ = iter_name
    creator.__doc__ = doc_str
    return creator

def _init_record_module():
    module_obj = sys.modules[__name__]
    record_iter = _make_record_iterator()
    setattr(module_obj, record_iter.__name__, record_iter)

_init_record_module()
