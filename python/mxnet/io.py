# coding: utf-8

"""NArray interface of mxnet"""
from __future__ import absolute_import

import ctypes
import sys
from .base import _LIB
from .base import c_array, c_str, mx_uint
from .base import DataIterHandle, NArrayHandle
from .base import check_call
from .narray import NArray

class DataIter(object):
    """DataIter object in mxnet. List all the needed functions here. """

    def __init__(self, handle):
        """Initialize with handle

        Parameters
        ----------
        handle : DataIterHandle
            the handle to the underlying C++ Data Iterator
        """
        self.handle = handle

    def __del__(self):
        check_call(_LIB.MXDataIterFree(self.handle))

    def __call__(self, *args, **kwargs):
        """Invoke iterator as function on inputs. Init params.

    def __call__(self, *args, **kwargs):
        """Invoke iterator as function on inputs. Init params.

        Parameters
        ---------
        args:
            provide positional arguments, should not be given.

        kwargs:
            provide keyword arguments
        Returns
        -------
        the inited iterator
        """
        if len(args) != 0:
            raise TypeError('data iterator only accept \
                    keyword arguments')     
        num_args = len(kwargs)
        keys = c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()]) 
        vals = c_array(ctypes.c_char_p, [c_str(val) for val in kwargs.values()])
        check_call(_LIB.MXDataIterSetInit( \
                self.handle, num_args, keys, vals))

    def beforefirst(self):
        """set loc to 0

        """
        check_call(_LIB.MXDataIterBeforeFirst(self.handle))

    def next(self):
        """init dataiter

        """
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        return next_res.value

    def getdata(self):
        """get data from batch

        """
        hdl = NArrayHandle()
        check_call(_LIB.MXDataIterGetData(self.handle, ctypes.byref(hdl)))
        return NArray(hdl)

    def getlabel(self):
        """get label from batch

        """
        hdl = NArrayHandle()
        check_call(_LIB.MXDataIterGetLabel(self.handle, ctypes.byref(hdl)))
        return NArray(hdl)

def _make_io_iterator(handle):
    """Create an io iterator by handle."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXDataIterGetIterInfo( \
            handle, ctypes.byref(name), ctypes.byref(desc), \
            ctypes.byref(num_args), \
            ctypes.byref(arg_names), \
            ctypes.byref(arg_types), \
            ctypes.byref(arg_descs)))
    iter_name = name.value
    param_str = []
    for i in range(num_args.value):
        ret = '%s : %s' % (arg_names[i], arg_types[i])
        if len(arg_descs[i]) != 0:
            ret += '\n    ' + arg_descs[i]
        param_str.append(ret)

    doc_str = ('%s\n\n' +
               'Parameters\n' +
               '----------\n' +
               '%s\n' +
               'name : string, required.\n' +
               '    Name of the resulting data iterator.\n\n' +
               'Returns\n' +
               '-------\n' +
               'iterator: Iterator\n'+
               '    The result iterator.')
    doc_str = doc_str % (desc.value, '\n'.join(param_str))

    def creator(*args, **kwargs):
        """Create an iterator.
        The parameters listed below can be passed in as keyword arguments.

        Parameters
        ----------
        name : string, required.
            Name of the resulting data iterator.

        Returns
        -------
        symbol: Symbol
            the resulting symbol
        """
        param_keys = []
        param_vals = []

        for k, val in kwargs.items():
            param_keys.append(c_str(k))
            param_vals.append(c_str(str(val)))
        # create atomic symbol
        param_keys = c_array(ctypes.c_char_p, param_keys)
        param_vals = c_array(ctypes.c_char_p, param_vals)
        iter_handle = DataIterHandle()
        check_call(_LIB.MXDataIterCreateIter(
            handle, len(param_keys),
            param_keys, param_vals,
            ctypes.byref(iter_handle)))

        if len(args):
            raise TypeError('%s can only accept keyword arguments' % iter_name)

        return DataIter(iter_handle)

    creator.__name__ = iter_name
    creator.__doc__ = doc_str
    return creator


def _init_io_module():
    """List and add all the data iterators to current module."""
    plist = ctypes.POINTER(ctypes.c_void_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListDataIters(ctypes.byref(size), ctypes.byref(plist)))
    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = ctypes.c_void_p(plist[i])
        dataiter = _make_io_iterator(hdl)
        setattr(module_obj, dataiter.__name__, dataiter)

# Initialize the io in startups
_init_io_module()
