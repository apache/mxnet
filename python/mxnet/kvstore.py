# coding: utf-8
""" Key value store interface of MXNet for parameter synchronization."""
from __future__ import absolute_import

import ctypes
from .ndarray import NDArray
from .base import _LIB
from .base import check_call, c_array, c_str, string_types, mx_uint
from .base import NDArrayHandle, KVStoreHandle


def _ctype_key_value(keys, vals):
    """
    Return ctype arrays for the key-value args, for internal use
    """
    if isinstance(keys, int):
        if isinstance(vals, NDArray):
            return (c_array(ctypes.c_int, [keys]),
                    c_array(NDArrayHandle, [vals.handle]))
        else:
            for value in vals:
                assert(isinstance(value, NDArray))
            return (c_array(ctypes.c_int, [keys] * len(vals)),
                    c_array(NDArrayHandle, [value.handle for value in vals]))
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


def _updater_wrapper(updater):
    """ a wrapper for the user-defined handle """
    def updater_handle(key, lhs_handle, rhs_handle):
        """ ctypes function """
        lhs = NDArray(NDArrayHandle(lhs_handle))
        rhs = NDArray(NDArrayHandle(rhs_handle))
        updater(key, lhs, rhs)
    return updater_handle


class KVStore(object):
    """A key-value store for synchronization of values, over multiple devices."""
    def __init__(self, handle):
        """Initialize a new KVStore.

        Parameters
        ----------
        handle : KVStoreHandle
            KVStore handle of C API
        """
        assert isinstance(handle, KVStoreHandle)
        self.handle = handle
        self._updater_func = None

    def __del__(self):
        check_call(_LIB.MXKVStoreFree(self.handle))

    def init(self, key, value):
        """ Initialize a single or a sequence of key-value pairs into the store.

        For each key, one must init it before push and pull.

        Parameters
        ----------
        key : int or sequence of int
            The keys.
        value : NDArray or sequence of NDArray
            The values.

        Examples
        --------
        >>> # init a single key-value pair
        >>> shape = (2,3)
        >>> kv = mx.kv.create('local')
        >>> kv.init(3, mx.nd.ones(shape)*2)
        >>> a = mx.nd.zeros(shape)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # init a list of key-value pairs
        >>> keys = [5, 7, 9]
        >>> kv.init(keys, [mx.nd.ones(shape)]*len(keys))
        """
        ckeys, cvals = _ctype_key_value(key, value)
        check_call(_LIB.MXKVStoreInit(
            self.handle, mx_uint(len(ckeys)), ckeys, cvals))

    def push(self, key, value, priority=0):
        """ Push a single or a sequence of key-value pairs into the store.

        Parameters
        ----------
        key : int or list of int
            Keys

        value : NDArray or list of NDArray or list of list of NDArray
            According values

        priority : int, optional
            The priority of the push operation.
            The higher the priority, the faster this action is likely
            to be executed before other push actions.

        Examples
        --------
        >>> # push a single key-value pair
        >>> kv.push(3, mx.nd.ones(shape)*8)
        >>> kv.pull(3, out=a) # pull out the value
        >>> print a.asnumpy()
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]

        >>> # aggregate the value and the push
        >>> gpus = [mx.gpu(i) for i in range(4)]
        >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
        >>> kv.push(3, b)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]

        >>> # push a list of keys.
        >>> # single device
        >>> kv.push(keys, [mx.nd.ones(shape)]*len(keys))
        >>> b = [mx.nd.zeros(shape)]*len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1].asnumpy()
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]

        >>> # multiple devices:
        >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
        >>> kv.push(keys, b)
        >>> kv.pull(keys, out=b)
        >>> print b[1][1].asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        """
        ckeys, cvals = _ctype_key_value(key, value)
        check_call(_LIB.MXKVStorePush(
            self.handle, mx_uint(len(ckeys)), ckeys, cvals,
            ctypes.c_int(priority)))

    def pull(self, key, out=None, priority=0):
        """ Pull a single value or a sequence of values from the store.

        Parameters
        ----------
        key : int or list of int
            Keys

        out: NDArray or list of NDArray or list of list of NDArray
            According values

        priority : int, optional
            The priority of the push operation.
            The higher the priority, the faster this action is likely
            to be executed before other push actions.

        Examples
        --------
        >>> # pull a single key-value pair
        >>> a = mx.nd.zeros(shape)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # pull into multiple devices
        >>> b = [mx.nd.ones(shape, gpu) for gpu in gpus]
        >>> kv.pull(3, out=b)
        >>> print b[1].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

        >>> # pull a list of key-value pairs.
        >>> # On single device
        >>> keys = [5, 7, 9]
        >>> b = [mx.nd.zeros(shape)]*len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
        >>> # On multiple devices
        >>> b = [[mx.nd.ones(shape, gpu) for gpu in gpus]] * len(keys)
        >>> kv.pull(keys, out=b)
        >>> print b[1][1].asnumpy()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
        """
        assert(out is not None)
        ckeys, cvals = _ctype_key_value(key, out)
        check_call(_LIB.MXKVStorePull(
            self.handle, mx_uint(len(ckeys)), ckeys, cvals,
            ctypes.c_int(priority)))

    def set_updater(self, updater):
        """Set a push updater into the store.

        Parameters
        ----------
        updater: function
            the updater function

        Examples
        --------
        >>> def update(key, input, stored):
        ...     print "update on key: %d" % key
        ...     stored += input * 2
        >>> kv.set_updater(update)
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        >>> kv.push(3, mx.nd.ones(shape))
        update on key: 3
        >>> kv.pull(3, out=a)
        >>> print a.asnumpy()
        [[ 6.  6.  6.]
        [ 6.  6.  6.]]
        """
        _updater_proto = ctypes.CFUNCTYPE(
            None, ctypes.c_int, NDArrayHandle, NDArrayHandle)
        self._updater_func = _updater_proto(_updater_wrapper(updater))
        check_call(_LIB.MXKVStoreSetUpdater(self.handle, self._updater_func))


def create(name='local'):
    """Create a new KVStore.

    Parameters
    ----------
    name : {'local'}
        The type of KVStore
        - local: KVStore that works on devices with single process.

    Returns
    -------
    kv : KVStore
        The created KVStore
    """
    if not isinstance(name, string_types):
        raise TypeError('name need to be string')
    handle = KVStoreHandle()
    check_call(_LIB.MXKVStoreCreate(c_str(name),
                                    ctypes.byref(handle)))
    return KVStore(handle)
