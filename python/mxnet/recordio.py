# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments

"""Python interface for DLMC RecrodIO data format"""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import RecordIOHandle
from .base import check_call

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
        buf = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char*size.value))
        return buf.contents.raw
