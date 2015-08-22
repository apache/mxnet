# coding: utf-8

"""NArray interface of mxnet"""
from __future__ import absolute_import

import ctypes
from .base import _LIB
from .base import DataIterHandle, NArrayHandle
from .base import check_call
from .narray import NArray

class DataIter(object):
    """DataIter object in mxnet

    DataIter is a wrapper for C++ DataIter functions
    """

    def __init__(self):
        """initialize a new dataiter

        """
        self._datahandle = None

    def createfromcfg(self, cfg_path):
        """create a dataiter from config file

        cfg_path is the path of configure file
        """
        hdl = DataIterHandle()
        check_call(_LIB.MXIOCreateFromConfig(ctypes.c_char_p(cfg_path), ctypes.byref(hdl)))
        self._datahandle = hdl

    def createbyname(self, iter_name):
        """create a dataiter by the name

        iter_name can be mnist imgrec or so on
        """
        hdl = DataIterHandle()
        check_call(_LIB.MXIOCreateByName(ctypes.c_char_p(iter_name), ctypes.byref(hdl)))
        self._datahandle = hdl

    def setparam(self, name, val):
        """set param value for dataiter

        name prameter name
        val parameter value
        """
        check_call(_LIB.MXIOSetParam(self._datahandle, ctypes.c_char_p(name), ctypes.c_char_p(val)))

    def init(self):
        """init dataiter

        """
        check_call(_LIB.MXIOInit(self._datahandle))

    def beforefirst(self):
        """set loc to 0

        """
        check_call(_LIB.MXIOBeforeFirst(self._datahandle))

    def next(self):
        """init dataiter

        """
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXIONext(self._datahandle, ctypes.byref(next_res)))
        return next_res.value

    def getdata(self):
        """get data from batch

        """
        hdl = NArrayHandle()
        check_call(_LIB.MXIOGetData(self._datahandle, ctypes.byref(hdl)))
        return NArray(hdl)

    def getlabel(self):
        """get label from batch

        """
        hdl = NArrayHandle()
        check_call(_LIB.MXIOGetLabel(self._datahandle, ctypes.byref(hdl)))
        return NArray(hdl)
