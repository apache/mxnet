# coding: utf-8
""" a server node for the key value store """
from __future__ import absolute_import
import ctypes
from .base import _LIB, check_call, c_array, c_str, string_types, mx_uint
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
        check_call(_LIB.MXKVStoreFree(self.handle))

    def controller(self):
        """return the controller"""
        def server_controller(head, body):
            yield
        return server_controller

    def run(self):
        _ctrl_proto = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
        check_call(_LIB.MXKVStoreRunServer(self.handle, _ctrl_proto(self.controller())))
