# coding: utf-8
# pylint: disable=invalid-name, global-statement
""" KVStore in mxnet """
from __future__ import absolute_import
import ctypes
from .ndarray import NDArray
from .base import _LIB
from .base import check_call, c_array, NDArrayHandle
import atexit

__all__ = ['start', 'init', 'push', 'pull', 'stop', 'set_updater']

def _ctype_key_value(keys, vals):
    """
    Return ctype arrays for the key-value args, for internal use
    """
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
    """
    Start the KV Store. One must call it before calling any other functions.

    Examples:
    ---------
    >>> import mxnet as mx
    >>> mx.kv.start()
    """
    check_call(_LIB.MXKVStoreStart())


def init(key, value):
    """ Initialize a single or a sequence of key-value pairs into the store.

    For each key, one must init it before push and pull

    Parameters
    ----------
    key : int or sequence of int
        The keys
    value : NDArray or sequence of NDArray
        The values

    Examples
    --------
    >>> # init a single key-value pair
    >>> shape = (2,3)
    >>> mx.kv.init(3, mx.nd.ones(shape)*2)
    >>> a = mx.nd.zeros(shape)
    >>> mx.kv.pull(3, out = a)
    >>> print a.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]

    >>> # init a list of key-value pairs
    >>> keys = [5, 7, 9]
    >>> mx.kv.init(keys, [mx.nd.ones(shape)]*len(keys))
    """
    ckeys, cvals = _ctype_key_value(key, value)
    check_call(_LIB.MXKVStoreInit(len(ckeys), ckeys, cvals))

def push(key, value):
    """ Push a single or a sequence of key-value pairs into the store

    Parameters
    ----------
    key : int or list of int
        Keys
    value: NDArray or list of NDArray or list of list of NDArray
        According values

    Examples
    --------
    >>> # push a single key-value pair
    >>> mx.kv.push(3, mx.nd.ones(shape)*8)
    >>> mx.kv.pull(3, out = a) # pull out the value
    >>> print a.asnumpy()
    [[ 8.  8.  8.]
     [ 8.  8.  8.]]

    >>> # aggregate the value and the push
    >>> gpus = [mx.gpu(i) for i in range(4)]
    >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
    >>> mx.kv.push(3, b)
    >>> mx.kv.pull(3, out = a)
    >>> print a.asnumpy()
    [[ 4.  4.  4.]
     [ 4.  4.  4.]]

    >>> # push a list of keys.
    >>> # single device
    >>> mx.kv.push(keys, [mx.nd.ones(shape)]*len(keys))
    >>> b = [mx.nd.zeros(shape)]*len(keys)
    >>> mx.kv.pull(keys, out = b)
    >>> print b[1].asnumpy()
    [[ 1.  1.  1.]
     [ 1.  1.  1.]]

    >>> # multiple devices:
    >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
    >>> mx.kv.push(keys, b)
    >>> mx.kv.pull(keys, out = b)
    >>> print b[1][1].asnumpy()
    [[ 4.  4.  4.]
     [ 4.  4.  4.]]
    """
    ckeys, cvals = _ctype_key_value(key, value)
    check_call(_LIB.MXKVStorePush(len(ckeys), ckeys, cvals))

def pull(key, out=None):
    """ Pull a single value or a sequence of values from the store

    Parameters
    ----------
    key : int or list of int
        Keys
    out: NDArray or list of NDArray or list of list of NDArray
        According values

    Examples
    --------
    >>> # pull a single key-value pair
    >>> a = mx.nd.zeros(shape)
    >>> mx.kv.pull(3, out = a)
    >>> print a.asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]

    >>> # pull into multiple devices
    >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
    >>> mx.kv.pull(3, out = b)
    >>> print b[1].asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]

    >>> # pull a list of key-value pairs.
    >>> # On single device
    >>> keys = [5, 7, 9]
    >>> b = [mx.nd.zeros(shape)]*len(keys)
    >>> mx.kv.pull(keys, out = b)
    >>> print b[1].asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    >>> # On multiple devices
    >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
    >>> mx.kv.pull(keys, out = b)
    >>> print b[1][1].asnumpy()
    [[ 2.  2.  2.]
     [ 2.  2.  2.]]
    """
    assert(out is not None)
    ckeys, cvals = _ctype_key_value(key, out)
    check_call(_LIB.MXKVStorePull(len(ckeys), ckeys, cvals))

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
    """
    Set a push updater into the store

    Parameters
    ----------
    updater: function
       the updater function

    Examples:
    ---------
    >>> def update(key, input, stored):
    >>>     print "update on key: %d" % key
    >>>     stored += input * 2
    >>> mx.kv.set_updater(update)
    >>> mx.kv.pull(3, out=a)
    >>> print a.asnumpy()
    [[ 4.  4.  4.]
     [ 4.  4.  4.]]
    >>> mx.kv.push(3, mx.nd.ones(shape))
    update on key: 3
    >>> mx.kv.pull(3, out=a)
    >>> print a.asnumpy()
    [[ 6.  6.  6.]
     [ 6.  6.  6.]]
    """
    _updater_proto = ctypes.CFUNCTYPE(
        None, ctypes.c_int, NDArrayHandle, NDArrayHandle)
    global _updater_func
    _updater_func = _updater_proto(_updater_wrapper(updater))
    check_call(_LIB.MXKVStoreSetUpdater(_updater_func))

def stop():
    """ Stop the kvstore """
    # need to clear _updater_func before _LIB
    global _updater_func
    _updater_func = None


atexit.register(stop)
