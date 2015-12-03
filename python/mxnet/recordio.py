# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments, no-member

"""Python interface for DLMC RecrodIO data format"""
from __future__ import absolute_import
from collections import namedtuple

import ctypes
from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
import struct
import numpy as np
try:
    import cv2
    opencv_available = True
except ImportError:
    print('OpenCV is unavailable.')
    opencv_available = False

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
        uri = ctypes.c_char_p(uri)
        self.handle = RecordIOHandle()
        if flag == "w":
            check_call(_LIB.MXRecordIOWriterCreate(uri, ctypes.byref(self.handle)))
            self.writable = True
        elif flag == "r":
            check_call(_LIB.MXRecordIOReaderCreate(uri, ctypes.byref(self.handle)))
            self.writable = False
        else:
            raise ValueError("Invalid flag %s"%flag)

    def __del__(self):
        if self.writable:
            check_call(_LIB.MXRecordIOWriterFree(self.handle))
        else:
            check_call(_LIB.MXRecordIOReaderFree(self.handle))

    def write(self, buf):
        """Write a string buffer as a record

        Parameters
        ----------
        buf : string
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

IRHeader = namedtuple('HEADER', ['flag', 'label', 'id', 'id2'])
_IRFormat = 'IfQQ'
_IRSize = struct.calcsize(_IRFormat)

def pack(header, s):
    """pack an string into MXImageRecord

    Parameters
    ----------
    header : IRHeader
        header of the image record
    s : str
        string to pack
    """
    header = IRHeader(*header)
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
    return header, s[_IRSize:]

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
    assert opencv_available
    img = cv2.imdecode(img, iscolor)
    return header, img

def pack_img(header, img, quality=80):
    """pack an image into MXImageRecord

    Parameters
    ----------
    header : IRHeader
        header of the image record
    img : numpy.ndarray
        image to pack
    quality : int
        quality for JPEG encoding. 1-100

    Returns
    -------
    s : str
        The packed string
    """
    assert opencv_available
    ret, buf = cv2.imencode('.JPEG', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ret
    return pack(header, buf.tostring())
