# coding: utf-8
""" a server node for the key value store """

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

    def Run():
        # TODO(mli)
        yield
