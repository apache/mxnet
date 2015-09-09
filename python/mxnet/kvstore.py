# coding: utf-8
""" KVStore in mxnet """
from __future__ import absolute_import
import ctypes
from .narray import NArray
from .context import Context
from .base import _LIB
from .base import check_call, c_array, NArrayHandle

def init_devices(contexts):
    """ Init key-value store with a list of device contexts

    Parameters
    ----------
    contexts : list of Context
       The list of local devices used by this process
    """
    masks = c_array(ctypes.c_int, [c.device_mask for c in contexts])
    ids = c_array(ctypes.c_int, [c.device_id for c in contexts])
    check_call(_LIB.MXKVStoreInitDevices(len(contexts), masks, ids))

def init(kv_list):
    """ Initialize a list of key-value pairs

    Parameters
    ----------
    kv_list : tuple or list/generator of tuples
        a key-value tuple or a list of key-value tuples, where key is int and
        key is
    """
    if isinstance(kv_list, tuple):
        init([kv_list])
    else:
        for kv in kv_list:
            assert len(kv) == 2
            assert isinstance(kv[0], int)
            assert isinstance(kv[1], NArray)
            check_call(_LIB.MXKVStoreInit(kv[0], kv[1].handle))

def push(kv_list):
    """ Push a value into the store

    Parameters
    ----------
    """
    if isinstance(kv_list, tuple):
        push([kv_list])
    else:
        for kv in kv_list:
            assert len(kv) == 2
            assert isinstance(kv[0], int)
            assert isinstance(kv[1], NArray)
            check_call(_LIB.MXKVStorePush(kv[0], kv[1].handle))

def pull(kv_list):
    """ Pull the value from the store

    Parameters
    ----------
    key : int
        The key
    value : NArray
        The value
    """
    if isinstance(kv_list, tuple):
        pull([kv_list])
    else:
        for kv in kv_list:
            assert len(kv) == 2
            assert isinstance(kv[0], int)
            assert isinstance(kv[1], NArray)
            check_call(_LIB.MXKVStorePull(kv[0], kv[1].handle))

def updater_wrapper(updater):
    def updater_handle(lhs_handle, rhs_handle):
        updater(NArray(lhs_handle), NArray(rhs_handle))
    return updater_handle

def void_updater(lhs, rhs):
    pass

updater_proto = ctypes.CFUNCTYPE(None, NArrayHandle, NArrayHandle)
updater_func = updater_proto(updater_wrapper(void_updater))

def register(updater):
    """ Register a updater into the store

    Example:
    def Update(grad, weight):
        weight[:] -= lr * grad  / batch_size

    Parameters
    ----------

    """
    global updater_func
    updater_func = updater_proto(updater)
    check_call(_LIB.MXKVStoreRegister(updater_func))
