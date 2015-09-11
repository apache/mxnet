# coding: utf-8
# pylint: disable=invalid-name, global-variable-undefined,
""" KVStore in mxnet """
from __future__ import absolute_import
import ctypes
import warnings
from .narray import NArray
from .narray import empty
from .base import _LIB
from .base import check_call, c_array, NArrayHandle

def _ctype_key_value(keys, vals):
    """ parse key-value args into ctype"""
    if isinstance(keys, int):
        if isinstance(vals, NArray):
            return (1,
                    c_array(ctypes.c_int, [keys]),
                    c_array(NArrayHandle, [vals.handle]))
        else:
            for v in vals:
                assert(isinstance(v, NArray))
            return (len(vals),
                    c_array(ctypes.c_int, [keys] * len(vals)),
                    c_array(NArrayHandle, [v.handle for v in vals]))
    else:
        for k in keys:
            assert(isinstance(k, int))
        if len(keys) == 1:
            return _ctype_key_value(keys[0], vals)
        assert(len(keys) == len(vals))
        for v in vals:
            assert(isinstance(v, NArray))
        return (len(keys),
                c_array(ctypes.c_int, keys),
                c_array(NArrayHandle, [v.handle for v in vals]))

def start():
    """start kvstore"""
    check_call(_LIB.MXKVStoreStart())

def stop():
    """ Stop kvstore """
    check_call(_LIB.MXKVStoreStop())


def init(keys, values):
    """ Initialize a list of key-value pairs

    Parameters
    ----------
    keys: int or list of int
        A single key or a list of keys
    values: NArray or list of NArray
        A single value of a list of values
    """
    num, ckeys, cvals = _ctype_key_value(keys, values)
    check_call(_LIB.MXKVStoreInit(num, ckeys, cvals))

def push(keys, values):
    """ Push a value into the store

    Parameters
    ----------
    keys: int or list of int
        A single key or a list of keys
    values: NArray or list of NArray
        A single value of a list of values
    """
    num, ckeys, cvals = _ctype_key_value(keys, values)
    check_call(_LIB.MXKVStorePush(num, ckeys, cvals))

def pull(keys, output=None, shape=None, ctx=None):
    """ Pull the value from the store

    Parameters
    ----------
    keys: int or list of int
        A single key or a list of keys
    output: NArray or list of NArray
        A single value of a list of values
    """
    if output is None:
        assert(isinstance(keys, int))
        assert(shape is not None)
        assert(ctx is not None)
        output = [empty(shape, c) for c in ctx]
    else:
        if shape is not None:
            warnings.warn('ignore shape', RuntimeWarning)
        if ctx is not None:
            warnings.warn('ignore ctx', RuntimeWarning)

    num, ckeys, cvals = _ctype_key_value(keys, output)
    print num

    return output
    # print keys, output, shape, ctx
    # num, ckeys, cvals = _ctype_key_value(keys, values)
    # check_call(_LIB.MXKVStorePull(num, ckeys, cvals))


def _updater_wrapper(updater):
    """ a wrapper for the user-defined handle """
    def updater_handle(lhs_handle, rhs_handle):
        """ ctypes function """
        lhs = NArray(NArrayHandle(lhs_handle))
        rhs = NArray(NArrayHandle(rhs_handle))
        updater(lhs, rhs)
    return updater_handle

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
    _updater_proto = ctypes.CFUNCTYPE(None, NArrayHandle, NArrayHandle)
    global _updater_func
    _updater_func = _updater_proto(_updater_wrapper(updater))
    check_call(_LIB.MXKVStoreSetUpdater(_updater_func))
