# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments, no-member

"""Python interface for DLMC RecrodIO data format"""
from __future__ import absolute_import
from collections import namedtuple

import ctypes
import struct
import numbers
import numpy as np

from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
try:
    import cv2
except ImportError:
    cv2 = None

class MXRecordIO(object):
    """Python interface for read/write RecordIO data formmat

    Parameters
    ----------
    uri : string
        uri path to recordIO file.
    flag : string
        "r" for reading or "w" writing.
    """
    def __init__(self, uri, flag):
        self.uri = c_str(uri)
        self.handle = RecordIOHandle()
        self.flag = flag
        self.is_open = False
        self.open()

    def open(self):
        """Open record file"""
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
        """close record file"""
        if not self.is_open:
            return
        if self.writable:
            check_call(_LIB.MXRecordIOWriterFree(self.handle))
        else:
            check_call(_LIB.MXRecordIOReaderFree(self.handle))
        self.is_open = False

    def reset(self):
        """Reset pointer to first item. If record is opened with 'w',
        this will truncate the file to empty"""
        self.close()
        self.open()

    def write(self, buf):
        """Write a string buffer as a record

        Parameters
        ----------
        buf : string (python2), bytes (python3)
            buffer to write.
        """
        assert self.writable
        check_call(_LIB.MXRecordIOWriterWriteRecord(self.handle,
                                                    ctypes.c_char_p(buf),
                                                    ctypes.c_size_t(len(buf))))

    def read(self):
        """Read a record as string

        Returns
        ----------
        buf : string
            buffer read.
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
    """Python interface for read/write RecordIO data formmat with index.
    Support random access.

    Parameters
    ----------
    idx_path : str
        Path to index file
    uri : str
        Path to record file. Only support file types that are seekable.
    flag : str
        'w' for write or 'r' for read
    key_type : type
        data type for keys
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
        if not self.is_open:
            return
        super(MXIndexedRecordIO, self).close()
        self.fidx.close()

    def seek(self, idx):
        """Query current read head position"""
        assert not self.writable
        pos = ctypes.c_size_t(self.idx[idx])
        check_call(_LIB.MXRecordIOReaderSeek(self.handle, pos))

    def tell(self):
        """Query current write head position"""
        assert self.writable
        pos = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOWriterTell(self.handle, ctypes.byref(pos)))
        return pos.value

    def read_idx(self, idx):
        """Read record with index"""
        self.seek(idx)
        return self.read()

    def write_idx(self, idx, buf):
        """Write record with index"""
        key = self.key_type(idx)
        pos = self.tell()
        self.write(buf)
        self.fidx.write('%s\t%d\n'%(str(key), pos))
        self.idx[key] = pos
        self.keys.append(key)


IRHeader = namedtuple('HEADER', ['flag', 'label', 'id', 'id2'])
_IRFormat = 'IfQQ'
_IRSize = struct.calcsize(_IRFormat)

def pack(header, s):
    """pack an string into MXImageRecord

    Parameters
    ----------
    header : IRHeader
        header of the image record.
        header.label can be a number or an array.
    s : str
        string to pack
    """
    header = IRHeader(*header)
    if isinstance(header.label, numbers.Number):
        header = header._replace(flag=0)
    else:
        label = np.asarray(header.label, dtype=np.float32)
        header = header._replace(flag=label.size, label=0)
        s = label.tostring() + s
    s = struct.pack(_IRFormat, *header) + s
    return s

def unpack(s):
    """unpack a MXImageRecord to string

    Parameters
    ----------
    s : str
        string buffer from MXRecordIO.read

    Returns
    -------
    header : IRHeader
        header of the image record
    s : str
        unpacked string
    """
    header = IRHeader(*struct.unpack(_IRFormat, s[:_IRSize]))
    s = s[_IRSize:]
    if header.flag > 0:
        header = header._replace(label=np.fromstring(s, np.float32, header.flag))
        s = s[header.flag*4:]
    return header, s

def unpack_img(s, iscolor=-1):
    """unpack a MXImageRecord to image

    Parameters
    ----------
    s : str
        string buffer from MXRecordIO.read
    iscolor : int
        image format option for cv2.imdecode

    Returns
    -------
    header : IRHeader
        header of the image record
    img : numpy.ndarray
        unpacked image
    """
    header, s = unpack(s)
    img = np.fromstring(s, dtype=np.uint8)
    assert cv2 is not None
    img = cv2.imdecode(img, iscolor)
    return header, img

def pack_img(header, img, quality=95, img_fmt='.jpg'):
    """pack an image into MXImageRecord

    Parameters
    ----------
    header : IRHeader
        header of the image record
        header.label can be a number or an array.
    img : numpy.ndarray
        image to pack
    quality : int
        quality for JPEG encoding. 1-100, or compression for PNG encoding. 1-9.
    img_fmt : str
        Encoding of the image. .jpg for JPEG, .png for PNG.

    Returns
    -------
    s : str
        The packed string
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
    assert ret, 'failed encoding image'
    return pack(header, buf.tostring())
