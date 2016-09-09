from ..base import MXNetError

ctypedef void* SymbolHandle
ctypedef void* OpHandle
ctypedef unsigned nn_uint

cdef py_str(const char* x):
    if PY_MAJOR_VERSION < 3:
        return x
    else:
        return x.decode("utf-8")


cdef c_str(pystr):
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
    return pystr.encode("utf-8")


cdef CALL(int ret):
    if ret != 0:
        raise MXNetError(NNGetLastError())


cdef const char** CBeginPtr(vector[const char*]& vec):
    if (vec.size() != 0):
        return &vec[0]
    else:
        return NULL

cdef vector[const char*] SVec2Ptr(vector[string]& vec):
    cdef vector[const char*] svec
    svec.resize(vec.size())
    for i in range(vec.size()):
        svec[i] = vec[i].c_str()
    return svec
