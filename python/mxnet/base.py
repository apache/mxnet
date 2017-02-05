# coding: utf-8
# pylint: disable=invalid-name, no-member
""" ctypes library of mxnet and helper functions """
from __future__ import absolute_import

import sys
import ctypes
import atexit
import warnings
warnings.filterwarnings('default', category=DeprecationWarning)
import numpy as np
from . import libinfo

__all__ = ['MXNetError']
#----------------------------
# library loading
#----------------------------
if sys.version_info[0] == 3:
    string_types = str,
    numeric_types = (float, int, np.float32, np.int32)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    string_types = basestring,
    numeric_types = (float, int, long, np.float32, np.int32)
    py_str = lambda x: x


class MXNetError(Exception):
    """Error that will be throwed by all mxnet functions"""
    pass

def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    # DMatrix functions
    lib.MXGetLastError.restype = ctypes.c_char_p
    return lib

# version number
__version__ = libinfo.__version__
# library instance of mxnet
_LIB = _load_lib()

# type definitions
mx_uint = ctypes.c_uint
mx_float = ctypes.c_float
mx_float_p = ctypes.POINTER(mx_float)
mx_real_t = np.float32
NDArrayHandle = ctypes.c_void_p
FunctionHandle = ctypes.c_void_p
OpHandle = ctypes.c_void_p
SymbolHandle = ctypes.c_void_p
ExecutorHandle = ctypes.c_void_p
DataIterCreatorHandle = ctypes.c_void_p
DataIterHandle = ctypes.c_void_p
KVStoreHandle = ctypes.c_void_p
RecordIOHandle = ctypes.c_void_p
RtcHandle = ctypes.c_void_p
#----------------------------
# helper function definition
#----------------------------
def check_call(ret):
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise MXNetError(py_str(_LIB.MXGetLastError()))

if sys.version_info[0] < 3:
    def c_str(string):
        """Create ctypes char * from a python string

        Parameters
        ----------
        string : string type
            python string

        Returns
        -------
        str : c_char_p
            A char pointer that can be passed to C API
        """
        return ctypes.c_char_p(string)
else:
    def c_str(string):
        """Create ctypes char * from a python string

        Parameters
        ----------
        string : string type
            python string

        Returns
        -------
        str : c_char_p
            A char pointer that can be passed to C API
        """
        return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Create ctypes array from a python array

    Parameters
    ----------
    ctype : ctypes data type
        data type of the array we want to convert to

    values : tuple or list
        data content

    Returns
    -------
    out : ctypes array
        Created ctypes array
    """
    return (ctype * len(values))(*values)

def ctypes2buffer(cptr, length):
    """Convert ctypes pointer to buffer type.

    Parameters
    ----------
    cptr : ctypes.POINTER(ctypes.c_char)
        pointer to the raw memory region
    length : int
        the length of the buffer

    Returns
    -------
    buffer : bytearray
        The raw byte memory buffer
    """
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise TypeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res

def ctypes2numpy_shared(cptr, shape):
    """Convert a ctypes pointer to a numpy array

    The result numpy array shares the memory with the pointer

    Parameters
    ----------
    cptr : ctypes.POINTER(mx_float)
        pointer to the memory region

    shape : tuple
        shape of target ndarray

    Returns
    -------
    out : numpy_array
        A numpy array : numpy array
    """
    if not isinstance(cptr, ctypes.POINTER(mx_float)):
        raise RuntimeError('expected float pointer')
    size = 1
    for s in shape:
        size *= s
    dbuffer = (mx_float * size).from_address(ctypes.addressof(cptr.contents))
    return np.frombuffer(dbuffer, dtype=np.float32).reshape(shape)


def build_param_doc(arg_names, arg_types, arg_descs, remove_dup=True):
    """Build argument docs in python style.

    arg_names : list of str
        Argument names.

    arg_types : list of str
        Argument type information.

    arg_descs : list of str
        Argument description information.

    remove_dup : boolean, optional
        Whether remove duplication or not.

    Returns
    -------
    docstr : str
        Python docstring of parameter sections.
    """
    param_keys = set()
    param_str = []
    for key, type_info, desc in zip(arg_names, arg_types, arg_descs):
        if key in param_keys and remove_dup:
            continue
        param_keys.add(key)
        ret = '%s : %s' % (key, type_info)
        if len(desc) != 0:
            ret += '\n    ' + desc
        param_str.append(ret)
    doc_str = ('Parameters\n' +
               '----------\n' +
               '%s\n')
    doc_str = doc_str % ('\n'.join(param_str))
    return doc_str


def _notify_shutdown():
    """Notify MXNet about a shutdown."""
    check_call(_LIB.MXNotifyShutdown())

atexit.register(_notify_shutdown)
