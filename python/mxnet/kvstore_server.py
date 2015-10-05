# coding: utf-8
""" a server node for the key value store """
from __future__ import absolute_import
import ctypes
from .base import check_call, c_array, c_str, string_types, mx_uint
from .base import NDArrayHandle, KVStoreHandle

class KVStoreServer(object):
    """A key-value store server"""
    def __init__(self, handle):
        """Initialize a new KVStore.

        Parameters
        ----------
        handle : KVStoreHandle
            KVStore handle of C API
        """
        assert isinstance(handle, KVStoreHandle)
        self.handle = handle

    def __del__(self):
        yield

    def Run(self):
        check_call(_LIB.MXKVStoreRunServer(self.handle))
