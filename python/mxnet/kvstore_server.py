# coding: utf-8
""" a server node for the key value store """
from __future__ import absolute_import
import ctypes
import sys
import pickle
from .base import _LIB, check_call
from .kvstore import create

class KVStoreServer(object):
    """The key-value store server"""
    def __init__(self, kvstore):
        """Initialize a new KVStoreServer.

        Parameters
        ----------
        kvstore : KVStore
        """
        self.kvstore = kvstore
        self.handle = kvstore.handle

    def _controller(self):
        """return the server controller"""
        def server_controller(cmd_id, cmd_body):
            """server controler"""
            if cmd_id == 0:
                try:
                    optimizer = pickle.loads(cmd_body)
                except:
                    raise
                self.kvstore.set_optimizer(optimizer)
            else:
                print "server %d, unknown command (%d, %s)" % (
                    self.kvstore.get_rank(), cmd_id, cmd_body)
        return server_controller

    def run(self):
        """run the server, whose behavior is like


        >>> while receive(x):
        ...     if is_command x: controller(x)
        ...     else if is_key_value x: updater(x)
        """
        _ctrl_proto = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
        check_call(_LIB.MXKVStoreRunServer(self.handle, _ctrl_proto(self._controller())))


def _init_kvstore_server_module():
    """Start server/scheduler"""
    is_worker = ctypes.c_int()
    check_call(_LIB.MXKVStoreIsWorkerNode(ctypes.byref(is_worker)))
    if is_worker.value == 0:
        kvstore = create('dist')
        server = KVStoreServer(kvstore)
        server.run()
        sys.exit()

_init_kvstore_server_module()
