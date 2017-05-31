from ..base import MXNetError

from libcpp.vector cimport vector
from libcpp.string cimport string
from cpython.version cimport PY_MAJOR_VERSION

ctypedef void* SymbolHandle
ctypedef void* NDArrayHandle
ctypedef void* OpHandle
ctypedef void* CachedOpHandle
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


cdef extern from "nnvm/c_api.h":
    const char* NNGetLastError();
    int NNGetOpHandle(const char *op_name,
                      OpHandle *handle);
    int NNGetOpInfo(OpHandle op,
                    const char **name,
                    const char **description,
                    nn_uint *num_doc_args,
                    const char ***arg_names,
                    const char ***arg_type_infos,
                    const char ***arg_descriptions,
                    const char **return_type);
    int NNSymbolFree(SymbolHandle symbol);
    int NNSymbolCompose(SymbolHandle sym,
                        const char* name,
                        nn_uint num_args,
                        const char** keys,
                        SymbolHandle* args);


cdef extern from "mxnet/c_api.h":
    int MXListAllOpNames(nn_uint *out_size,
                         const char ***out_array);
    int MXSymbolGetAtomicSymbolInfo(OpHandle creator,
                                    const char **name,
                                    const char **description,
                                    nn_uint *num_doc_args,
                                    const char ***arg_names,
                                    const char ***arg_type_infos,
                                    const char ***arg_descriptions,
                                    const char **key_var_args,
                                    const char **return_type);
    int MXSymbolCreateAtomicSymbol(OpHandle op,
                                   nn_uint num_param,
                                   const char **keys,
                                   const char **vals,
                                   SymbolHandle *out);
    int MXSymbolSetAttr(SymbolHandle symbol,
                        const char* key,
                        const char* value);
    int MXImperativeInvoke(OpHandle creator,
                           int num_inputs,
                           NDArrayHandle *inputs,
                           int *num_outputs,
                           NDArrayHandle **outputs,
                           int num_params,
                           const char **param_keys,
                           const char **param_vals);
    int MXNDArrayFree(NDArrayHandle handle);
    int MXCachedCreateOp(OpHandle creator,
                         int num_inputs,
                         int num_params,
                         const char **param_keys,
                         const char **param_vals,
                         CachedOpHandle *out);
    int MXCachedFree(CachedOpHandle handle);
    int MXCachedInvoke(CachedOpHandle handle,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs);
    int MXCachedCreateSymbol(CachedOpHandle handle,
                             const char* name,
                             unsigned num_args,
                             SymbolHandle* args,
                             SymbolHandle* out);


cdef class CachedOp:
    """Cached operator handle."""
    cdef CachedOpHandle chandle
    cdef string cop

    cdef _set_handle(self, handle):
        cdef unsigned long long ptr
        if handle is None:
            self.chandle = NULL
        else:
            ptr = handle.value
            self.chandle = <SymbolHandle>(ptr)

    property handle:
        def __get__(self):
            if self.chandle == NULL:
                return None
            else:
                return _ctypes.cast(<unsigned long long>self.chandle, _ctypes.c_void_p)
        def __set__(self, value):
            self._set_handle(value)

    property op:
        def __get__(self):
            return py_str(self.cop.c_str())
        def __set__(self, value):
            self.cop = c_str(value)

    def __init__(self, op, num_input, **kwargs):
        cdef OpHandle op_handle
        cdef vector[string] ckeys
        cdef vector[string] cvals

        self.op = op
        CALL(NNGetOpHandle(self.cop.c_str(), &op_handle))

        for k, v in kwargs.items():
            ckeys.push_back(c_str(k))
            cvals.push_back(c_str(str(v)))

        cdef vector[const char*] param_keys = SVec2Ptr(ckeys)
        cdef vector[const char*] param_vals = SVec2Ptr(cvals)

        CALL(MXCachedCreateOp(
            op_handle,
            <int>num_input,
            <int>len(kwargs),
            CBeginPtr(param_keys),
            CBeginPtr(param_vals),
            &self.chandle))

    def __del__(self):
        CALL(MXCachedFree(self.chandle))
