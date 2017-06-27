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
    int MXCreateCachedOp(SymbolHandle handle,
                         CachedOpHandle *out);
    int MXFreeCachedOp(CachedOpHandle handle);
    int MXInvokeCachedOp(CachedOpHandle handle,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs);
