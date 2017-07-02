%module "AI::MXNetCAPI"
%rename("%(strip:[MX])s") "";
%include typemaps.i
%include mxnet_typemaps.i
%inline %{
#include <c_api.h>

// Taken as is from http://cpansearch.perl.org/src/COLEMINOR/Games-EternalLands-Binary-Float16-0.01/Float16.xs
/* This method is faster than the OpenEXR implementation (very often
 * used, eg. in Ogre), with the additional benefit of rounding, inspired
 * by James Tursa's half-precision code. */
static inline uint16_t _float_to_half(uint32_t x) {
  uint16_t bits = (x >> 16) & 0x8000;
  uint16_t m = (x >> 12) & 0x07ff;
  unsigned int e = (x >> 23) & 0xff;
  if (e < 103)
    return bits;
  if (e > 142) {
    bits |= 0x7c00u;
    bits |= e == 255 && (x & 0x007fffffu);
    return bits;
  }
  if (e < 113) {
    m |= 0x0800u;
    bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
    return bits;
  }
  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1;
  return bits;
}

static int const shifttable[32] = {
  23, 14, 22, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 20, 0,
  15, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 17, 0, 18, 19, 0,
};
static uint32_t const shiftmagic = 0x07c4acddu;

/* This algorithm is similar to the OpenEXR implementation, except it
 * uses branchless code in the denormal path. This is slower than a
 * table version, but will be more friendly to the cache for occasional
 * uses. */
static inline uint32_t _half_to_float(uint16_t x) {
  uint32_t s = (x & 0x8000u) << 16;
  if ((x & 0x7fffu) == 0)
    return (uint32_t)x << 16;
  uint32_t e = x & 0x7c00u;
  uint32_t m = x & 0x03ffu;
  if (e == 0) {
    uint32_t v = m | (m >> 1);
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    e = shifttable[(v * shiftmagic) >> 27];
    return s | (((125 - e) << 23) + (m << e));
  }
  if (e == 0x7c00u) {
    if (m == 0)
      return s | 0x7f800000u;
    return s | 0x7fc00000u;
  }
  return s | (((e >> 10) + 112) << 23) | (m << 13);
}

union fbits {
  float f;
  uint32_t x;
};

static void KVStore_callback(int index, NDArrayHandle recv, NDArrayHandle local, void* callback)
{
    {
        dSP;
        PUSHMARK(SP);
        XPUSHs(sv_2mortal(newSViv(index)));
        XPUSHs(SWIG_NewPointerObj(SWIG_as_voidptr(recv), SWIGTYPE_p_MXNDArray, 0));
        XPUSHs(SWIG_NewPointerObj(SWIG_as_voidptr(local), SWIGTYPE_p_MXNDArray, 0));
        PUTBACK;
        call_sv((SV*)callback, G_DISCARD);
    }
}

static void KVStoreServer_callback(int head, const char *body, void* callback)
{
    {
        dSP;
        PUSHMARK(SP);
        XPUSHs(sv_2mortal(newSViv(head)));
        XPUSHs(sv_2mortal(newSVpv(body, 0)));
        PUTBACK;
        call_sv((SV*)callback, G_DISCARD);
    }
}

static void ExecutorMonitor_callback(const char* name, NDArrayHandle handle, void* callback)
{
    {
        dSP;
        PUSHMARK(SP);
        XPUSHs(sv_2mortal(newSVpv(name, 0)));
        XPUSHs(SWIG_NewPointerObj(SWIG_as_voidptr(handle), SWIGTYPE_p_MXNDArray, 0));
        PUTBACK;
        call_sv((SV*)callback, G_DISCARD);
    }
}

%} 

%init %{
    /* These SWIG_TypeClientData() calls might break in the future, but
     * %rename should work on these types before that happens. */
    SWIG_TypeClientData(SWIGTYPE_p_MXNDArray, (void *)"NDArrayHandle");
    SWIG_TypeClientData(SWIGTYPE_p_MXFunction, (void *)"FunctionHandle");
    SWIG_TypeClientData(SWIGTYPE_p_MXAtomicSymbolCreator, (void *)"AtomicSymbolCreator");
    SWIG_TypeClientData(SWIGTYPE_p_MXSymbol, (void *)"SymbolHandle");
    SWIG_TypeClientData(SWIGTYPE_p_MXExecutor, (void *)"ExecutorHandle");
    SWIG_TypeClientData(SWIGTYPE_p_MXDataIterCreator, (void *)"DataIterCreator");
    SWIG_TypeClientData(SWIGTYPE_p_MXDataIter, (void *)"DataIterHandle");
    SWIG_TypeClientData(SWIGTYPE_p_MXKVStore, (void *)"KVStoreHandle");
    SWIG_TypeClientData(SWIGTYPE_p_MXRecordIO, (void *)"RecordIOHandle");
    SWIG_TypeClientData(SWIGTYPE_p_MXRtc, (void *)"RtcHandle");
%}

/*! \brief manually define unsigned int */
typedef unsigned int mx_uint;
/*! \brief manually define float */
typedef float mx_float;
// all the handles are simply void *
// will be casted internally to specific pointers types
// these typedefs are mainly used for readablity reasons
/*! \brief handle to NDArray */
typedef MXNDArray *NDArrayHandle;
/*! \brief handle to a mxnet ndarray function that changes NDArray */
typedef MXFunction *FunctionHandle;
/*! \brief handle to a function that takes param and creates symbol */
typedef MXAtomicSymbolCreator *AtomicSymbolCreator;
/*! \brief handle to a symbol that can be bind as operator */
typedef MXSymbol *SymbolHandle;
/*! \brief handle to a AtomicSymbol */
typedef MXAtomicSymbol *AtomicSymbolHandle;
/*! \brief handle to an Executor */
typedef MXExecutor *ExecutorHandle;
/*! \brief handle a dataiter creator */
typedef MXDataIterCreator *DataIterCreator;
/*! \brief handle to a DataIterator */
typedef MXDataIter *DataIterHandle;
/*! \brief handle to KVStore */
typedef MXKVStore *KVStoreHandle;
/*! \brief handle to RecordIO */
typedef MXRecordIO *RecordIOHandle;
/*! \brief handle to MXRtc*/
typedef MXRtc *RtcHandle;

typedef void (*ExecutorMonitorCallback)(const char*,
                                                       NDArrayHandle,
                                                       void *);
struct NativeOpInfo {
  void (*forward)(int, float**, int*, unsigned**, int*, void*);
  void (*backward)(int, float**, int*, unsigned**, int*, void*);
  void (*infer_shape)(int, int*, unsigned**, void*);
  void (*list_outputs)(char***, void*);
  void (*list_arguments)(char***, void*);
  // all functions also pass a payload void* pointer
  void* p_forward;
  void* p_backward;
  void* p_infer_shape;
  void* p_list_outputs;
  void* p_list_arguments;
};

struct NDArrayOpInfo {
  bool (*forward)(int, void**, int*, void*);
  bool (*backward)(int, void**, int*, void*);
  bool (*infer_shape)(int, int*, unsigned**, void*);
  bool (*list_outputs)(char***, void*);
  bool (*list_arguments)(char***, void*);
  bool (*declare_backward_dependency)(const int*, const int*, const int*,
                                      int*, int**, void*);
  // all functions also pass a payload void* pointer
  void* p_forward;
  void* p_backward;
  void* p_infer_shape;
  void* p_list_outputs;
  void* p_list_arguments;
  void* p_declare_backward_dependency;
};

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  MXGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
const char *MXGetLastError();

//-------------------------------------
// Part 0: Global State setups
//-------------------------------------
/*!
 * \brief Seed the global random number generators in mxnet.
 * \param seed the random number seed.
 * \return 0 when success, -1 when failure happens.
 */
int MXRandomSeed(int seed);
/*!
 * \brief Notify the engine about a shutdown,
 *  This can help engine to print less messages into display.
 *
 *  User do not have to call this function.
 * \return 0 when success, -1 when failure happens.
 */
int MXNotifyShutdown();
/*!
 * \brief Set up configuration of profiler
 * \param mode indicate the working mode of profiler,
 *  record anly symbolic operator when mode == 0,
 *  record all operator when mode == 1
 * \param filename where to save trace file
 * \return 0 when success, -1 when failure happens.
 */
int MXSetProfilerConfig(int mode, const char* filename);
/*!
 * \brief Set up state of profiler
 * \param state indicate the working state of profiler,
 *  profiler not running when state == 0,
 *  profiler running when state == 1
 * \return 0 when success, -1 when failure happens.
 */
int MXSetProfilerState(int state);

/*! \brief Save profile and stop profiler */
int MXDumpProfile();

/*! \brief Set the number of OMP threads to use */
int MXSetNumOMPThreads(int thread_num);

//-------------------------------------
// Part 1: NDArray creation and deletion
//-------------------------------------
/*!
 * \brief create a NDArray handle that is not initialized
 *  can be used to pass in as mutate variables
 *  to hold the result of NDArray
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayCreateNone(NDArrayHandle *out);
/*!
 * \brief create a NDArray with specified shape
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *    the ndarray is first mutated
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayCreate(const mx_uint *in,
                              mx_uint ndim,
                              int dev_type,
                              int dev_id,
                              int delay_alloc,
                              NDArrayHandle *out);

/*!
 * \brief create a NDArray with specified shape and data type
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *    the ndarray is first mutated
 * \param dtype data type of created array
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayCreateEx(const mx_uint *in,
                              mx_uint ndim,
                              int dev_type,
                              int dev_id,
                              int delay_alloc,
                              int dtype,
                              NDArrayHandle *out);
/*!
 * \brief create a NDArray handle that is loaded from raw bytes.
 * \param buf the head of the raw bytes
 * \param size size of the raw bytes
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayLoadFromRawBytes(const void *in,
                                        size_t size,
                                        NDArrayHandle *out);
/*!
 * \brief save the NDArray into raw bytes.
 * \param handle the NDArray handle
 * \param out_size size of the raw bytes
 * \param out_buf the head of returning memory bytes.
 * \return 0 when success, -1 when failure happens
 */
int MXNDArraySaveRawBytes(NDArrayHandle handle,
                                    size_t *out_size,
                                    const char **out_array);
/*!
 * \brief Save list of ndarray into the file.
 * \param fname name of the file.
 * \param num_args number of arguments to save.
 * \param args the array of NDArrayHandles to be saved.
 * \param keys the name of the NDArray, optional, can be NULL
 * \return 0 when success, -1 when failure happens
 */
int MXNDArraySave(const char* fname,
                            mx_uint num_args,
                            NDArrayHandle* in,
                            const char** in);
/*!
 * \brief Load list of ndarray from the file.
 * \param fname name of the file.
 * \param out_size number of ndarray loaded.
 * \param out_arr head of the returning ndarray handles.
 * \param out_name_size size of output name arrray.
 * \param out_names the names of returning NDArrays, can be NULL
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayLoad(const char* fname,
                            mx_uint *out_size,
                            NDArrayHandle** out_array,
                            mx_uint *out_size,
                            const char*** out_array);
/*!
 * \brief Perform a synchronize copy from a continugous CPU memory region.
 *
 *  This function will call WaitToWrite before the copy is performed.
 *  This is useful to copy data from existing memory region that are
 *  not wrapped by NDArray(thus dependency not being tracked).
 *
 * \param handle the NDArray handle
 * \param data the data source to copy from.
 * \param size the memory size we want to copy from.
 */
int MXNDArraySyncCopyFromCPU(NDArrayHandle handle,
                                       const void *in,
                                       size_t size);
/*!
 * \brief Perform a synchronize copy to a continugous CPU memory region.
 *
 *  This function will call WaitToRead before the copy is performed.
 *  This is useful to copy data from existing memory region that are
 *  not wrapped by NDArray(thus dependency not being tracked).
 *
 * \param handle the NDArray handle
 * \param data the data source to copy into.
 * \param size the memory size we want to copy into.
 */
int MXNDArraySyncCopyToCPU(NDArrayHandle handle,
                                     void *in,
                                     size_t size);
/*!
 * \brief Wait until all the pending writes with respect NDArray are finished.
 *  Always call this before read data out synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayWaitToRead(NDArrayHandle handle);
/*!
 * \brief Wait until all the pending read/write with respect NDArray are finished.
 *  Always call this before write data into NDArray synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayWaitToWrite(NDArrayHandle handle);
/*!
 * \brief wait until all delayed operations in
 *   the system is completed
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayWaitAll();
/*!
 * \brief free the ndarray handle
 * \param handle the handle to be freed
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayFree(NDArrayHandle handle);
/*!
 * \brief Slice the NDArray along axis 0.
 * \param handle the handle to the NDArray
 * \param slice_begin The beginning index of slice
 * \param slice_end The ending index of slice
 * \param out The NDArrayHandle of sliced NDArray
 * \return 0 when success, -1 when failure happens
 */
int MXNDArraySlice(NDArrayHandle handle,
                             mx_uint slice_begin,
                             mx_uint slice_end,
                             NDArrayHandle *out);
/*!
 * \brief Index the NDArray along axis 0.
 * \param handle the handle to the NDArray
 * \param idx the index
 * \param out The NDArrayHandle of output NDArray
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayAt(NDArrayHandle handle,
                          mx_uint idx,
                          NDArrayHandle *out);
/*!
 * \brief Reshape the NDArray.
 * \param handle the handle to the ndarray
 * \param ndim number of dimensions of new shape
 * \param dims new shape
 * \param out the NDArrayHandle of reshaped NDArray
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayReshape(NDArrayHandle handle,
                               int ndim,
                               int *in,
                               NDArrayHandle *out);
/*!
 * \brief get the shape of the array
 * \param handle the handle to the ndarray
 * \param out_dim the output dimension
 * \param out_pdata pointer holder to get data pointer of the shape
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayGetShape(NDArrayHandle handle,
                                mx_uint *out_dim,
                                const mx_uint **out_pdata);
/*!
 * \brief get the content of the data in NDArray
 * \param handle the handle to the ndarray
 * \param out_pdata pointer holder to get pointer of data
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayGetData(NDArrayHandle handle,
                                void **out_pdata);
/*!
 * \brief get the type of the data in NDArray
 * \param handle the handle to the ndarray
 * \param out_dtype pointer holder to get type of data
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayGetDType(NDArrayHandle handle,
                               int *out);
/*!
 * \brief get the context of the NDArray
 * \param handle the handle to the ndarray
 * \param out_dev_type the output device type
 * \param out_dev_id the output device id
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayGetContext(NDArrayHandle handle,
                                  int *out,
                                  int *out);
/*!
 * \brief detach and ndarray from computation graph by clearing entry_
 * \param handle NDArray handle
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayDetach(NDArrayHandle handle, NDArrayHandle *out);

/*!
 * \brief set the flag for gradient array state.
 * \param handle NDArray handle
 * \param state the new state.
 * \return 0 when success, -1 when failure happens
 */
int MXNDArraySetGradState(NDArrayHandle handle, int state);

/*!
 * \brief set the flag for gradient array state.
 * \param handle NDArray handle
 * \param state the new state.
 * \return 0 when success, -1 when failure happens
 */
int MXNDArrayGetGradState(NDArrayHandle handle, int *out);

//--------------------------------
// Part 2: functions on NDArray
//--------------------------------
/*!
 * \brief list all the available functions handles
 *   most user can use it to list all the needed functions
 * \param out_size the size of returned array
 * \param out_array the output function array
 * \return 0 when success, -1 when failure happens
 */
int MXListFunctions(mx_uint *out_size,
                              FunctionHandle **out_array);
/*!
 * \brief get the function handle by name
 * \param name the name of the function
 * \param out the corresponding function handle
 * \return 0 when success, -1 when failure happens
 */
int MXGetFunction(const char *name,
                            FunctionHandle *out);
/*!
 * \brief Get the information of the function handle.
 * \param fun The function handle.
 * \param name The returned name of the function.
 * \param description The returned description of the function.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type information about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \param return_type Return type of the function.
 * \return 0 when success, -1 when failure happens
 */
int MXFuncGetInfo(FunctionHandle fun,
                            const char **name,
                            const char **description,
                            mx_uint *num_args,
                            const char ***arg_names,
                            const char ***arg_type_infos,
                            const char ***arg_descriptions
                            );
/*!
 * \brief get the argument requirements of the function
 * \param fun input function handle
 * \param num_use_vars how many NDArrays to be passed in as used_vars
 * \param num_scalars scalar variable is needed
 * \param num_mutate_vars how many NDArrays to be passed in as mutate_vars
 * \param type_mask the type mask of this function
 * \return 0 when success, -1 when failure happens
 * \sa MXFuncInvoke
 */
int MXFuncDescribe(FunctionHandle fun,
                             mx_uint *out,
                             mx_uint *out,
                             mx_uint *out,
                             int *out);
/*!
 * \brief invoke a function, the array size of passed in arguments
 *   must match the values in the
 * \param fun the function
 * \param use_vars the normal arguments passed to function
 * \param scalar_args the scalar qarguments
 * \param mutate_vars the mutate arguments
 * \return 0 when success, -1 when failure happens
 * \sa MXFuncDescribeArgs
 */
int MXFuncInvoke(FunctionHandle fun,
                           NDArrayHandle *in,
                           mx_float *in,
                           NDArrayHandle *in);
/*!
 * \brief invoke a function, the array size of passed in arguments
 *   must match the values in the
 * \param fun the function
 * \param use_vars the normal arguments passed to function
 * \param scalar_args the scalar qarguments
 * \param mutate_vars the mutate arguments
 * \param num_params number of keyword parameters
 * \param param_keys keys for keyword parameters
 * \param param_vals values for keyword parameters
 * \return 0 when success, -1 when failure happens
 * \sa MXFuncDescribeArgs
 */
int MXFuncInvokeEx(FunctionHandle fun,
                             NDArrayHandle *in,
                             mx_float *in,
                             NDArrayHandle *in,
                             int num_params,
                             char **keys,
                             char **vals);
/*!
 * \brief invoke a nnvm op and imperative function
 * \param creator the op
 * \param num_inputs number of input NDArrays
 * \param inputs input NDArrays
 * \param num_outputs number of output NDArrays
 * \param outputs output NDArrays
 * \param num_params number of keyword parameters
 * \param param_keys keys for keyword parameters
 * \param param_vals values for keyword parameters
 * \return 0 when success, -1 when failure happens
 */
int MXImperativeInvoke(AtomicSymbolCreator in,
                                 int num_inputs,
                                 NDArrayHandle *in,
                                 int *out_size,
                                 NDArrayHandle **out_array,
                                 int num_params,
                                 const char **keys,
                                 const char **vals);
/*!
 * \brief set whether to record operator for autograd
 * \param is_train 1 when training, 0 when testing
 * \param prev returns the previous status before this set.
 * \return 0 when success, -1 when failure happens
 */
int MXAutogradSetIsTraining(int is_training, int* out);
/*!
 * \brief mark NDArrays as variables to compute gradient for autograd
 * \param num_var number of variable NDArrays
 * \param var_handles variable NDArrays
 * \return 0 when success, -1 when failure happens
 */
int MXAutogradMarkVariables(mx_uint num_var,
                                      NDArrayHandle *in,
                                      mx_uint *in,
                                      NDArrayHandle *in);
/*!
 * \brief compute the gradient of outputs w.r.t variables
 * \param num_output number of output NDArray
 * \param output_handles output NDArrays
 * \return 0 when success, -1 when failure happens
 */
int MXAutogradComputeGradient(mx_uint num_output,
                                        NDArrayHandle* in);
/*!
 * \brief compute the gradient of outputs w.r.t variabels
 * \param num_output number of output NDArray
 * \param output_handles output NDArrays
 * \param ograd_handles head gradient for NDArrays
 * \param retain_graph whether to keep the graph after backward
 * \return 0 when success, -1 when failure happens
 */
int MXAutogradBackward(mx_uint num_output,
                                 NDArrayHandle* in,
                                 NDArrayHandle* in,
                                 int retain_graph);

//--------------------------------------------
// Part 3: symbolic configuration generation
//--------------------------------------------
/*!
 * \brief list all the available operator names, include entries.
 * \param out_size the size of returned array
 * \param out_array the output operator name array.
 * \return 0 when success, -1 when failure happens
 */
int MXListAllOpNames(mx_uint *out_size,
                               const char ***out_array);
/*!
 * \brief list all the available AtomicSymbolEntry
 * \param out_size the size of returned array
 * \param out_array the output AtomicSymbolCreator array
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                               AtomicSymbolCreator **out_array);

/*!
 * \brief Get the name of an atomic symbol.
 * \param creator the AtomicSymbolCreator.
 * \param name The returned name of the creator.
 */
int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator in,
                                          const char **out);
/*!
 * \brief Get the detailed information about atomic symbol.
 * \param creator the AtomicSymbolCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \param key_var_num_args The keyword argument for specifying variable number of arguments.
 *            When this parameter has non-zero length, the function allows variable number
 *            of positional arguments, and will need the caller to pass it in in
 *            MXSymbolCreateAtomicSymbol,
 *            With key = key_var_num_args, and value = number of positional arguments.
 * \param return_type Return type of the function, can be Symbol or Symbol[]
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator in,
                                          const char **name,
                                          const char **description,
                                          mx_uint *num_args,
                                          const char ***arg_names,
                                          const char ***arg_type_infos,
                                          const char ***arg_descriptions,
                                          const char **key_var_num_args
                                          );
/*!
 * \brief Create an AtomicSymbol.
 * \param creator the AtomicSymbolCreator
 * \param num_param the number of parameters
 * \param keys the keys to the params
 * \param vals the vals of the params
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator in,
                                         mx_uint num_param,
                                         const char **keys,
                                         const char **vals,
                                         SymbolHandle *out);
/*!
 * \brief Create a Variable Symbol.
 * \param name name of the variable
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolCreateVariable(const char *name, SymbolHandle *out);
/*!
 * \brief Create a Symbol by grouping list of symbols together
 * \param num_symbols number of symbols to be grouped
 * \param symbols array of symbol handles
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolCreateGroup(mx_uint num_symbols,
                                  SymbolHandle *in,
                                  SymbolHandle *out);
/*!
 * \brief Load a symbol from a json file.
 * \param fname the file name.
 * \param out the output symbol.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolCreateFromFile(const char *fname, SymbolHandle *out);
/*!
 * \brief Load a symbol from a json string.
 * \param json the json string.
 * \param out the output symbol.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolCreateFromJSON(const char *json, SymbolHandle *out);
/*!
 * \brief Save a symbol into a json file.
 * \param symbol the input symbol.
 * \param fname the file name.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolSaveToFile(SymbolHandle symbol, const char *fname);
/*!
 * \brief Save a symbol into a json string
 * \param symbol the input symbol.
 * \param out_json output json string.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolSaveToJSON(SymbolHandle symbol, const char **out);
/*!
 * \brief Free the symbol handle.
 * \param symbol the symbol
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolFree(SymbolHandle symbol);
/*!
 * \brief Copy the symbol to another handle
 * \param symbol the source symbol
 * \param out used to hold the result of copy
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolCopy(SymbolHandle symbol, SymbolHandle *out);
/*!
 * \brief Print the content of symbol, used for debug.
 * \param symbol the symbol
 * \param out_str pointer to hold the output string of the printing.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolPrint(SymbolHandle symbol, const char **out);
/*!
 * \brief Get string name from symbol
 * \param symbol the source symbol
 * \param out The result name.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolGetName(SymbolHandle symbol,
                              const char** out,
                              int *out);
/*!
 * \brief Get string attribute from symbol
 * \param symbol the source symbol
 * \param key The key of the symbol.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolGetAttr(SymbolHandle symbol,
                              const char* key,
                              const char** out,
                              int *out);
/*!
 * \brief Set string attribute from symbol.
 *  NOTE: Setting attribute to a symbol can affect the semantics(mutable/immutable) of symbolic graph.
 *
 *  Safe recommendaton: use  immutable graph
 *  - Only allow set attributes during creation of new symbol as optional parameter
 *
 *  Mutable graph (be careful about the semantics):
 *  - Allow set attr at any point.
 *  - Mutating an attribute of some common node of two graphs can cause confusion from user.
 *
 * \param symbol the source symbol
 * \param key The key of the symbol.
 * \param value The value to be saved.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolSetAttr(SymbolHandle symbol,
                              const char* in,
                              const char* in);
/*!
 * \brief Get all attributes from symbol, including all descendents.
 * \param symbol the source symbol
 * \param out_size The number of output attributes
 * \param out 2*out_size strings representing key value pairs.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolListAttr(SymbolHandle symbol,
                               mx_uint *out_size,
                               const char*** out_array2);
/*!
 * \brief Get all attributes from symbol, excluding descendents.
 * \param symbol the source symbol
 * \param out_size The number of output attributes
 * \param out 2*out_size strings representing key value pairs.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolListAttrShallow(SymbolHandle symbol,
                                      mx_uint *out_size,
                                      const char*** out_array2);
/*!
 * \brief List arguments in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolListArguments(SymbolHandle symbol,
                                    mx_uint *out_size,
                                    const char ***out_array);
/*!
 * \brief List returns in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolListOutputs(SymbolHandle symbol,
                                  mx_uint *out_size,
                                  const char ***out_array);
/*!
 * \brief Get a symbol that contains all the internals.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are all the internals.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolGetInternals(SymbolHandle symbol,
                                   SymbolHandle *out);
/*!
 * \brief Get a symbol that contains only direct children.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are the direct children.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolGetChildren(SymbolHandle symbol,
                                  SymbolHandle *out);
/*!
 * \brief Get index-th outputs of the symbol.
 * \param symbol The symbol
 * \param index the Index of the output.
 * \param out The output symbol whose outputs are the index-th symbol.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolGetOutput(SymbolHandle symbol,
                                mx_uint index,
                                SymbolHandle *out);
/*!
 * \brief List auxiliary states in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                          mx_uint *out_size,
                                          const char ***out_array);
/*!
 * \brief Compose the symbol on other symbols.
 *
 *  This function will change the sym hanlde.
 *  To achieve function apply behavior, copy the symbol first
 *  before apply.
 *
 * \param sym the symbol to apply
 * \param name the name of symbol
 * \param num_args number of arguments
 * \param keys the key of keyword args (optional)
 * \param args arguments to sym
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolCompose(SymbolHandle sym,
                              const char *name,
                              mx_uint num_args,
                              const char** in,
                              SymbolHandle* in);
/*!
 * \brief Get the gradient graph of the symbol
 *
 * \param sym the symbol to get gradient
 * \param num_wrt number of arguments to get gradient
 * \param wrt the name of the arguments to get gradient
 * \param out the returned symbol that has gradient
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolGrad(SymbolHandle sym,
                           mx_uint num_wrt,
                           const char** in,
                           SymbolHandle* out);
/*!
 * \brief infer shape of unknown input shapes given the known one.
 *  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
 *  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
 *
 * \param sym symbol handle
 * \param num_args numbe of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_ind_ptr the head pointer of the rows in CSR
 * \param arg_shape_data the content of the CSR
 * \param in_shape_size sizeof the returning array of in_shapes
 * \param in_shape_ndim returning array of shape dimensions of eachs input shape.
 * \param in_shape_data returning array of pointers to head of the input shape.
 * \param out_shape_size sizeof the returning array of out_shapes
 * \param out_shape_ndim returning array of shape dimensions of eachs input shape.
 * \param out_shape_data returning array of pointers to head of the input shape.
 * \param aux_shape_size sizeof the returning array of aux_shapes
 * \param aux_shape_ndim returning array of shape dimensions of eachs auxiliary shape.
 * \param aux_shape_data returning array of pointers to head of the auxiliary shape.
 * \param complete whether infer shape completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolInferShape(SymbolHandle sym,
                                 mx_uint num_args,
                                 const char** in,
                                 const mx_uint *in,
                                 const mx_uint *in,
                                 mx_uint *in_shape_size,
                                 const mx_uint **in_shape_ndim,
                                 const mx_uint ***in_shape_data,
                                 mx_uint *out_shape_size,
                                 const mx_uint **out_shape_ndim,
                                 const mx_uint ***out_shape_data,
                                 mx_uint *aux_shape_size,
                                 const mx_uint **aux_shape_ndim,
                                 const mx_uint ***aux_shape_data,
                                 int *out);
/*!
 * \brief partially infer shape of unknown input shapes given the known one.
 *
 *  Return partially inferred results if not all shapes could be inferred.
 *  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
 *  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
 *
 * \param sym symbol handle
 * \param num_args numbe of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_ind_ptr the head pointer of the rows in CSR
 * \param arg_shape_data the content of the CSR
 * \param in_shape_size sizeof the returning array of in_shapes
 * \param in_shape_ndim returning array of shape dimensions of eachs input shape.
 * \param in_shape_data returning array of pointers to head of the input shape.
 * \param out_shape_size sizeof the returning array of out_shapes
 * \param out_shape_ndim returning array of shape dimensions of eachs input shape.
 * \param out_shape_data returning array of pointers to head of the input shape.
 * \param aux_shape_size sizeof the returning array of aux_shapes
 * \param aux_shape_ndim returning array of shape dimensions of eachs auxiliary shape.
 * \param aux_shape_data returning array of pointers to head of the auxiliary shape.
 * \param complete whether infer shape completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolInferShapePartial(SymbolHandle sym,
                                 mx_uint num_args,
                                 const char** in,
                                 const mx_uint *in,
                                 const mx_uint *in,
                                 mx_uint *in_shape_size,
                                 const mx_uint **in_shape_ndim,
                                 const mx_uint ***in_shape_data,
                                 mx_uint *out_shape_size,
                                 const mx_uint **out_shape_ndim,
                                 const mx_uint ***out_shape_data,
                                 mx_uint *aux_shape_size,
                                 const mx_uint **aux_shape_ndim,
                                 const mx_uint ***aux_shape_data,
                                 int *out);

/*!
 * \brief infer type of unknown input types given the known one.
 *  The types are packed into a CSR matrix represented by arg_ind_ptr and arg_type_data
 *  The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
 *
 * \param sym symbol handle
 * \param num_args numbe of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_type_data the content of the CSR
 * \param in_type_size sizeof the returning array of in_types
 * \param in_type_data returning array of pointers to head of the input type.
 * \param out_type_size sizeof the returning array of out_types
 * \param out_type_data returning array of pointers to head of the input type.
 * \param aux_type_size sizeof the returning array of aux_types
 * \param aux_type_data returning array of pointers to head of the auxiliary type.
 * \param complete whether infer type completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
int MXSymbolInferType(SymbolHandle sym,
                                mx_uint num_args,
                                const char** in,
                                const int *in,
                                mx_uint *in_type_size,
                                const int **in_type_data,
                                mx_uint *out_type_size,
                                const int **out_type_data,
                                mx_uint *aux_type_size,
                                const int **aux_type_data,
                                int *out);
//--------------------------------------------
// Part 4: Executor interface
//--------------------------------------------
/*!
 * \brief Delete the executor
 * \param handle the executor.
 * \return 0 when success, -1 when failure happens
 */
int MXExecutorFree(ExecutorHandle handle);
/*!
 * \brief Print the content of execution plan, used for debug.
 * \param handle the executor.
 * \param out_str pointer to hold the output string of the printing.
 * \return 0 when success, -1 when failure happens
 */
int MXExecutorPrint(ExecutorHandle handle, const char **out);
/*!
 * \brief Executor forward method
 *
 * \param handle executor handle
 * \param is_train bool value to indicate whether the forward pass is for evaluation
 * \return 0 when success, -1 when failure happens
 */
int MXExecutorForward(ExecutorHandle handle, int is_train);
/*!
 * \brief Excecutor run backward
 *
 * \param handle execute handle
 * \param len lenth
 * \param head_grads NDArray handle for heads' gradient
 *
 * \return 0 when success, -1 when failure happens
 */
int MXExecutorBackward(ExecutorHandle handle,
                                 mx_uint len,
                                 NDArrayHandle *in);

/*!
 * \brief Get executor's head NDArray
 *
 * \param handle executor handle
 * \param out_size output ndarray vector size
 * \param out out put ndarray handles
 * \return 0 when success, -1 when failure happens
 */
int MXExecutorOutputs(ExecutorHandle handle,
                                mx_uint *out_size,
                                NDArrayHandle **out_array);

/*!
 * \brief Generate Executor from symbol
 *
 * \param symbol_handle symbol handle
 * \param dev_type device type
 * \param dev_id device id
 * \param len length
 * \param in_args in args array
 * \param arg_grad_store arg grads handle array
 * \param grad_req_type grad req array
 * \param aux_states_len length of auxiliary states
 * \param aux_states auxiliary states array
 * \param out output executor handle
 * \return 0 when success, -1 when failure happens
 */
int MXExecutorBind(SymbolHandle symbol_handle,
                             int dev_type,
                             int dev_id,
                             mx_uint len,
                             NDArrayHandle *in,
                             NDArrayHandle *in,
                             mx_uint *in,
                             mx_uint aux_states_len,
                             NDArrayHandle *in,
                             ExecutorHandle *out);
/*!
 * \brief Generate Executor from symbol,
 *  This is advanced function, allow specify group2ctx map.
 *  The user can annotate "ctx_group" attribute to name each group.
 *
 * \param symbol_handle symbol handle
 * \param dev_type device type of default context
 * \param dev_id device id of default context
 * \param num_map_keys size of group2ctx map
 * \param map_keys keys of group2ctx map
 * \param map_dev_types device type of group2ctx map
 * \param map_dev_ids device id of group2ctx map
 * \param len length
 * \param in_args in args array
 * \param arg_grad_store arg grads handle array
 * \param grad_req_type grad req array
 * \param aux_states_len length of auxiliary states
 * \param aux_states auxiliary states array
 * \param out output executor handle
 * \return 0 when success, -1 when failure happens
 */
int MXExecutorBindX(SymbolHandle symbol_handle,
                              int dev_type,
                              int dev_id,
                              mx_uint num_map_keys,
                              const char** in,
                              const int* in,
                              const int* in,
                              mx_uint len,
                              NDArrayHandle *in,
                              NDArrayHandle *in,
                              mx_uint *in,
                              mx_uint aux_states_len,
                              NDArrayHandle *in,
                              ExecutorHandle *out);
/*!
 * \brief Generate Executor from symbol,
 *  This is advanced function, allow specify group2ctx map.
 *  The user can annotate "ctx_group" attribute to name each group.
 *
 * \param symbol_handle symbol handle
 * \param dev_type device type of default context
 * \param dev_id device id of default context
 * \param num_map_keys size of group2ctx map
 * \param map_keys keys of group2ctx map
 * \param map_dev_types device type of group2ctx map
 * \param map_dev_ids device id of group2ctx map
 * \param len length
 * \param in_args in args array
 * \param arg_grad_store arg grads handle array
 * \param grad_req_type grad req array
 * \param aux_states_len length of auxiliary states
 * \param aux_states auxiliary states array
 * \param shared_exec input executor handle for memory sharing
 * \param out output executor handle
 * \return 0 when success, -1 when failure happens
 */
int MXExecutorBindEX(SymbolHandle symbol_handle,
                               int dev_type,
                               int dev_id,
                               mx_uint num_map_keys,
                               const char** in,
                               const int* in,
                               const int* in,
                               mx_uint len,
                               NDArrayHandle *in,
                               NDArrayHandle *in,
                               mx_uint *in,
                               mx_uint aux_states_len,
                               NDArrayHandle *in,
                               ExecutorHandle shared_exec,
                               ExecutorHandle *out);

int MXExecutorSimpleBind(SymbolHandle symbol_handle,
                         int dev_type,
                         int dev_id,
                         const mx_uint num_g2c_keys,
                         const char** in, // g2c_keys,
                         const int* in, // g2c_dev_types,
                         const int* in, // g2c_dev_ids,
                         const mx_uint provided_grad_req_list_len,
                         const char** in, // provided_grad_req_names,
                         const char** in, // provided_grad_req_types,
                         const mx_uint num_provided_arg_shapes,
                         const char** in, // provided_arg_shape_names,
                         const mx_uint* in, // provided_arg_shape_data,
                         const mx_uint* in, // provided_arg_shape_idx,
                         const mx_uint num_provided_arg_dtypes,
                         const char** in, // provided_arg_dtype_names,
                         const int* in, // provided_arg_dtypes,
                         const mx_uint num_shared_arg_names,
                         const char** in, // shared_arg_name_list,
//------------
                         int* shared_buffer_len,
                         const char** shared_buffer_name_list,
                         NDArrayHandle* shared_buffer_handle_list,
                         const char*** updated_shared_buffer_name_list,
                         NDArrayHandle** updated_shared_buffer_handle_list,
//------------------

                         mx_uint* num_in_args,
                         NDArrayHandle** in_args,
                         NDArrayHandle** arg_grads,
//-----------------
                         mx_uint* num_aux_states,
                         NDArrayHandle** aux_states,
//----------
                         ExecutorHandle shared_exec_handle,
                         ExecutorHandle* out
);

/*!
 * \brief set a call back to notify the completion of operation
 */
int MXExecutorSetMonitorCallback(ExecutorHandle handle,
                                           ExecutorMonitorCallback callback,
                                           void* callback_handle);
//--------------------------------------------
// Part 5: IO Interface
//--------------------------------------------
/*!
 * \brief List all the available iterator entries
 * \param out_size the size of returned iterators
 * \param out_array the output iteratos entries
 * \return 0 when success, -1 when failure happens
 */
int MXListDataIters(mx_uint *out_size,
                              DataIterCreator **out_array);
/*!
 * \brief Init an iterator, init with parameters
 * the array size of passed in arguments
 * \param handle of the iterator creator
 * \param num_param number of parameter
 * \param keys parameter keys
 * \param vals parameter values
 * \param out resulting iterator
 * \return 0 when success, -1 when failure happens
 */
int MXDataIterCreateIter(DataIterCreator handle,
                                   mx_uint num_param,
                                   const char **keys,
                                   const char **vals,
                                   DataIterHandle *out);
/*!
 * \brief Get the detailed information about data iterator.
 * \param creator the DataIterCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \return 0 when success, -1 when failure happens
 */
int MXDataIterGetIterInfo(DataIterCreator creator,
                                    const char **name,
                                    const char **description,
                                    mx_uint *num_args,
                                    const char ***arg_names,
                                    const char ***arg_type_infos,
                                    const char ***arg_descriptions);
/*!
 * \brief Free the handle to the IO module
 * \param handle the handle pointer to the data iterator
 * \return 0 when success, -1 when failure happens
 */
int MXDataIterFree(DataIterHandle handle);
/*!
 * \brief Move iterator to next position
 * \param handle the handle to iterator
 * \param out return value of next
 * \return 0 when success, -1 when failure happens
 */
int MXDataIterNext(DataIterHandle handle,
                             int *out);
/*!
 * \brief Call iterator.Reset
 * \param handle the handle to iterator
 * \return 0 when success, -1 when failure happens
 */
int MXDataIterBeforeFirst(DataIterHandle handle);

/*!
 * \brief Get the handle to the NDArray of underlying data
 * \param handle the handle pointer to the data iterator
 * \param out handle to underlying data NDArray
 * \return 0 when success, -1 when failure happens
 */
int MXDataIterGetData(DataIterHandle handle,
                                NDArrayHandle *out);
/*!
 * \brief Get the image index by array.
 * \param handle the handle pointer to the data iterator
 * \param out_index output index of the array.
 * \param out_size output size of the array.
 * \return 0 when success, -1 when failure happens
 */
int MXDataIterGetIndex(DataIterHandle handle,
                                 uint64_t **out_index,
                                 uint64_t *out_size);
/*!
 * \brief Get the padding number in current data batch
 * \param handle the handle pointer to the data iterator
 * \param pad pad number ptr
 * \return 0 when success, -1 when failure happens
 */
int MXDataIterGetPadNum(DataIterHandle handle,
                                  int *out);

/*!
 * \brief Get the handle to the NDArray of underlying label
 * \param handle the handle pointer to the data iterator
 * \param out the handle to underlying label NDArray
 * \return 0 when success, -1 when failure happens
 */
int MXDataIterGetLabel(DataIterHandle handle,
                                 NDArrayHandle *out);
//--------------------------------------------
// Part 6: basic KVStore interface
//--------------------------------------------
/*!
 * \brief Initialized ps-lite environment variables
 * \param num_vars number of variables to initialize
 * \param keys environment keys
 * \param vals environment values
 */
int MXInitPSEnv(mx_uint num_vars,
                          const char **keys,
                          const char **vals);


/*!
 * \brief Create a kvstore
 * \param type the type of KVStore
 * \param out The output type of KVStore
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreCreate(const char *type,
                              KVStoreHandle *out);
/*!
 * \brief Delete a KVStore handle.
 * \param handle handle to the kvstore
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreFree(KVStoreHandle handle);
/*!
 * \brief Init a list of (key,value) pairs in kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreInit(KVStoreHandle handle,
                            mx_uint num,
                            const int* in,
                            NDArrayHandle* in);

/*!
 * \brief Push a list of (key,value) pairs to kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
int MXKVStorePush(KVStoreHandle handle,
                            mx_uint num,
                            const int* in,
                            NDArrayHandle* in,
                            int priority);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
int MXKVStorePull(KVStoreHandle handle,
                            mx_uint num,
                            const int* in,
                            NDArrayHandle* in,
                            int priority);
/*!
 * \brief user-defined updater for the kvstore
 * It's this updater's responsibility to delete \a recv and \a local
 * \param the key
 * \param recv the pushed value on this key
 * \param local the value stored on local on this key
 * \param handle The additional handle to the updater
 */
typedef void (MXKVStoreUpdater)(int key,
                                NDArrayHandle recv,
                                NDArrayHandle local,
                                void *handle);
/*!
 * \brief register an push updater
 * \param handle handle to the KVStore
 * \param updater udpater function
 * \param updater_handle The additional handle used to invoke the updater
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreSetUpdater(KVStoreHandle handle,
                                  MXKVStoreUpdater updater,
                                  void *callback_handle);
/*!
 * \brief get the type of the kvstore
 * \param handle handle to the KVStore
 * \param type a string type
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreGetType(KVStoreHandle handle,
                               const char** out);
//--------------------------------------------
// Part 6: advanced KVStore for multi-machines
//--------------------------------------------

/**
 * \brief return The rank of this node in its group, which is in [0, GroupSize).
 *
 * \param handle handle to the KVStore
 * \param ret the node rank
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreGetRank(KVStoreHandle handle,
                               int *out);

/**
 * \brief return The number of nodes in this group, which is
 * - number of workers if if `IsWorkerNode() == true`,
 * - number of servers if if `IsServerNode() == true`,
 * - 1 if `IsSchedulerNode() == true`,
 * \param handle handle to the KVStore
 * \param ret the group size
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreGetGroupSize(KVStoreHandle handle,
                                    int *out);

/**
 * \brief return whether or not this process is a worker node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreIsWorkerNode(int *out);


/**
 * \brief return whether or not this process is a server node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreIsServerNode(int *out);


/**
 * \brief return whether or not this process is a scheduler node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreIsSchedulerNode(int *out);

/**
 * \brief global barrier among all worker machines
 *
 * \param handle handle to the KVStore
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreBarrier(KVStoreHandle handle);

/**
 * \brief whether to do barrier when finalize
 *
 * \param handle handle to the KVStore
 * \param barrier_before_exit whether to do barrier when kvstore finalize
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreSetBarrierBeforeExit(KVStoreHandle handle,
                                            const int barrier_before_exit);

/**
 * \brief the prototype of a server controller
 * \param head the head of the command
 * \param body the body of the command
 * \param controller_handle helper handle for implementing controller
 */
typedef void (MXKVStoreServerController)(int head,
                                         const char *body,
                                         void *controller_handle);

/**
 * \return Run as server (or scheduler)
 *
 * \param handle handle to the KVStore
 * \param controller the user-defined server controller
 * \param controller_handle helper handle for implementing controller
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreRunServer(KVStoreHandle handle,
                                 MXKVStoreServerController controller,
                                 void *callback_handle);

/**
 * \return Send a command to all server nodes
 *
 * \param handle handle to the KVStore
 * \param cmd_id the head of the command
 * \param cmd_body the body of the command
 * \return 0 when success, -1 when failure happens
 */
int MXKVStoreSendCommmandToServers(KVStoreHandle handle,
                                             int cmd_id,
                                             const char* cmd_body);

/**
 * \brief Get the number of ps dead node(s) specified by {node_id}
 *
 * \param handle handle to the KVStore
 * \param node_id Can be a node group or a single node.
 *                kScheduler = 1, kServerGroup = 2, kWorkerGroup = 4
 * \param number Ouptut number of dead nodes
 * \param timeout_sec A node fails to send heartbeart in {timeout_sec} seconds
 *                    will be presumed as 'dead'
 */
int MXKVStoreGetNumDeadNode(KVStoreHandle handle,
                                      const int node_id,
                                      int *out,
                                      const int timeout_sec = 60);

/**
 * \brief Create a RecordIO writer object
 * \param uri path to file
 * \param out handle pointer to the created object
 * \return 0 when success, -1 when failure happens
*/
int MXRecordIOWriterCreate(const char *uri, RecordIOHandle *out);

/**
 * \brief Delete a RecordIO writer object
 * \param handle handle to RecordIO object
 * \return 0 when success, -1 when failure happens
*/
int MXRecordIOWriterFree(RecordIOHandle handle);

/**
 * \brief Write a record to a RecordIO object
 * \param handle handle to RecordIO object
 * \param buf buffer to write
 * \param size size of buffer
 * \return 0 when success, -1 when failure happens
*/
int MXRecordIOWriterWriteRecord(RecordIOHandle handle,
                                          const char *buf, size_t size);

/**
 * \brief Get the current writer pointer position
 * \param handle handle to RecordIO object
 * \param pos handle to output position
 * \return 0 when success, -1 when failure happens
*/
int MXRecordIOWriterTell(RecordIOHandle handle, size_t *out);

/**
 * \brief Create a RecordIO reader object
 * \param uri path to file
 * \param out handle pointer to the created object
 * \return 0 when success, -1 when failure happens
*/
int MXRecordIOReaderCreate(const char *uri, RecordIOHandle *out);

/**
 * \brief Delete a RecordIO reader object
 * \param handle handle to RecordIO object
 * \return 0 when success, -1 when failure happens
*/
int MXRecordIOReaderFree(RecordIOHandle handle);

/**
 * \brief Write a record to a RecordIO object
 * \param handle handle to RecordIO object
 * \param buf pointer to return buffer
 * \param size point to size of buffer
 * \return 0 when success, -1 when failure happens
*/
int MXRecordIOReaderReadRecord(RecordIOHandle handle,
                                        char const **out_array, size_t *out_size);

/**
 * \brief Set the current reader pointer position
 * \param handle handle to RecordIO object
 * \param pos target position
 * \return 0 when success, -1 when failure happens
*/
int MXRecordIOReaderSeek(RecordIOHandle handle, size_t pos);

/**
 * \brief Create a MXRtc object
*/
int MXRtcCreate(char* name, mx_uint num_input, mx_uint num_output,
                          char** in, char** in,
                          NDArrayHandle* in, NDArrayHandle* in,
                          char* kernel, RtcHandle *out);

/**
 * \brief Run cuda kernel
*/
int MXRtcPush(RtcHandle handle, mx_uint num_input, mx_uint num_output,
                        NDArrayHandle* in, NDArrayHandle* in,
                        mx_uint gridDimX,
                        mx_uint gridDimY,
                        mx_uint gridDimZ,
                        mx_uint blockDimX,
                        mx_uint blockDimY,
                        mx_uint blockDimZ);

/**
 * \brief Delete a MXRtc object
*/
int MXRtcFree(RtcHandle handle);

int MXCustomOpRegister(const char* op_type, CustomOpPropCreator creator);
