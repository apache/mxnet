# coding: utf-8
# pylint: disable=invalid-name, global-statement
""" KVStore in mxnet """
from __future__ import absolute_import
import ctypes
from .ndarray import NDArray
from .base import _LIB
from .base import check_call, c_array, NDArrayHandle
import atexit

def _ctype_key_value(keys, vals):
    """parse key-value args into ctype"""
    if isinstance(keys, int):
        if isinstance(vals, NDArray):
            return (c_array(ctypes.c_int, [keys]),
                    c_array(NDArrayHandle, [vals.handle]))
        else:
            for v in vals:
                assert(isinstance(v, NDArray))
            return (c_array(ctypes.c_int, [keys] * len(vals)),
                    c_array(NDArrayHandle, [v.handle for v in vals]))
    else:
        assert(len(keys) == len(vals))
        for k in keys:
            assert(isinstance(k, int))
        c_keys = []
        c_vals = []
        for i in range(len(keys)):
            c_key_i, c_val_i = _ctype_key_value(keys[i], vals[i])
            c_keys += c_key_i
            c_vals += c_val_i
        return (c_array(ctypes.c_int, c_keys), c_array(NDArrayHandle, c_vals))

def start():
    """start kvstore"""
    check_call(_LIB.MXKVStoreStart())


def init(key, value):
    """ Initialize a list of key-value pairs

    Parameters
    ----------
    keys: int or list of int
        A single key or a list of keys
    values: NDArray or list of NDArray
        A single value of a list of values
    """
    ckeys, cvals = _ctype_key_value(key, value)
    check_call(_LIB.MXKVStoreInit(len(ckeys), ckeys, cvals))

def push(key, value):
    """ Push a value into the store

    Parameters
    ----------
    key : int or list of int
        A single key or a list of key
    value: list of NDArray or list of list of NDArray
        A single value of a list of value
    """
    ckeys, cvals = _ctype_key_value(key, value)
    check_call(_LIB.MXKVStorePush(len(ckeys), ckeys, cvals))

def pull(key, out=None):
    """Pull value from the store

    Parameters
    ----------
    key: int or list of int
        A single key or a list of key
    out: NDArray or list of NDArray
        A single value of a list of value
    """
    assert(out is not None)
    ckeys, cvals = _ctype_key_value(key, out)
    check_call(_LIB.MXKVStorePull(len(ckeys), ckeys, cvals))
    return out


def _updater_wrapper(updater):
    """ a wrapper for the user-defined handle """
    def updater_handle(key, lhs_handle, rhs_handle):
        """ ctypes function """
        lhs = NDArray(NDArrayHandle(lhs_handle))
        rhs = NDArray(NDArrayHandle(rhs_handle))
        updater(key, lhs, rhs)
    return updater_handle

_updater_func = None

def set_updater(updater):
    """ set a updater into the store

    Example:

    def updater(recv, local):
        local += recv
    kvstore.set_updater(updater)

    Parameters
    ----------
    updater: functon
    """
    _updater_proto = ctypes.CFUNCTYPE(
        None, ctypes.c_int, NDArrayHandle, NDArrayHandle)
    global _updater_func
    _updater_func = _updater_proto(_updater_wrapper(updater))
    check_call(_LIB.MXKVStoreSetUpdater(_updater_func))

def stop():
    """ Stop kvstore """
    check_call(_LIB.MXKVStoreStop())
    # need to clear _updater_func before _LIB
    global _updater_func
    _updater_func = None

atexit.register(stop)
