/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api.h
 * \brief C API of mxnet
 */
#ifndef MXNET_C_API_H_
#define MXNET_C_API_H_

#ifdef __cplusplus
#define MXNET_EXTERN_C extern "C"
#endif

/*! \brief MXNET_DLL prefix for windows" */
#ifdef _MSC_VER
#define MXNET_DLL MXNET_EXTERN_C __declspec(dllexport)
#else
#define MXNET_DLL MXNET_EXTERN_C
#endif

/*! \brief manually define unsigned int */
typedef unsigned int mx_uint;
/*! \brief manually define unsigned int */
typedef float mx_float;
// all the handles are simply void *
// will be casted internally to specific pointers types
// these typedefs are mainly used for readablity reasons
/*! \brief handle to NArray */
typedef void *NArrayHandle;
/*! \brief handle to a mxnet narray function that changes NArray */
typedef const void *FunctionHandle;
/*! \brief handle to a function that takes param and creates symbol */
typedef void *AtomicSymbolCreator;
/*! \brief handle to a symbol that can be bind as operator */
typedef void *SymbolHandle;
/*! \brief handle to a AtomicSymbol */
typedef void *AtomicSymbolHandle;
/*! \brief handle to an Executor */
typedef void *ExecutorHandle;
/*! \brief handle to a DataIterator */
typedef void *DataIterHandle;
/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  MXGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
MXNET_DLL const char *MXGetLastError();
//-------------------------------------
// Part 1: NArray creation and deletion
//-------------------------------------
/*!
 * \brief create a NArray handle that is not initialized
 *  can be used to pass in as mutate variables
 *  to hold the result of NArray
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNArrayCreateNone(NArrayHandle *out);
/*!
 * \brief create a NArray that shares the memory content with data
 *   NOTE: use this with caution, specifically, do not directly operate
 *   on original memory content unless you think you are confident to do so
 *   note that NArray operations are asynchronize and ONLY dependency between
 *   NArrays can be captured, when you are done with NArray and want to
 *   see the data content inside, call MXNArrayWait
 *   the caller must also keep the data content alive and not being gc
 *   during the liveness of NArray, usually by keep a ref to the data content obj
 *
 * \param data floating point pointer to the head of memory
 * \param shape the shape of the memory
 * \param ndim number of dimension of the shape
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNArrayCreateShareMem(mx_float *data,
                                     mx_uint *shape,
                                     mx_uint ndim,
                                     NArrayHandle *out);
/*!
 * \brief create a NArray with specified shape
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_mask device mask, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *    the narray is first mutated
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNArrayCreate(const mx_uint *shape,
                             mx_uint ndim,
                             int dev_mask,
                             int dev_id,
                             int delay_alloc,
                             NArrayHandle *out);
/*!
 * \brief wait until all the operation with respect NArray
 *  to this NArray is finished, always call this before fetching data out
 * \param handle the NArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNArrayWait(NArrayHandle handle);
/*!
 * \brief wait until all delayed operations in
 *   the system is completed
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNArrayWaitAll();
/*!
 * \brief free the narray handle
 * \param handle the handle to be freed
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNArrayFree(NArrayHandle handle);
/*!
 * \brief get the shape of the array
 * \param handle the handle to the narray
 * \param out_dim the output dimension
 * \param out_pdata pointer holder to get data pointer of the shape
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNArrayGetShape(NArrayHandle handle,
                               mx_uint *out_dim,
                               const mx_uint **out_pdata);
/*!
 * \brief get the content of the data in NArray
 * \param handle the handle to the narray
 * \param out_pdata pointer holder to get pointer of data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNArrayGetData(NArrayHandle handle,
                              mx_float **out_pdata);
/*!
 * \brief get the context of the NArray
 * \param handle the handle to the narray
 * \param out_dev_mask the output device mask
 * \param out_dev_id the output device id
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNArrayGetContext(NArrayHandle handle,
                                 int *out_dev_mask,
                                 int *out_dev_id);

//--------------------------------
// Part 2: functions on NArray
//--------------------------------
/*!
 * \brief list all the available functions handles
 *   most user can use it to list all the needed functions
 * \param out_size the size of returned array
 * \param out_array the output function array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXListFunctions(mx_uint *out_size,
                              FunctionHandle **out_array);
/*!
 * \brief get the function handle by name
 * \param name the name of the function
 * \param out the corresponding function handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXGetFunction(const char *name,
                            FunctionHandle *out);
/*!
 * \brief Get the information of the function handle.
 * \param fun The function handle.
 * \param name The returned name of the function.
 * \param description The returned description of the function.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXFuncGetInfo(FunctionHandle fun,
                            const char **name,
                            const char **description,
                            mx_uint *num_args,
                            const char ***arg_names,
                            const char ***arg_type_infos,
                            const char ***arg_descriptions);
/*!
 * \brief get the argument requirements of the function
 * \param fun input function handle
 * \param num_use_vars how many NArrays to be passed in as used_vars
 * \param num_scalars scalar variable is needed
 * \param num_mutate_vars how many NArrays to be passed in as mutate_vars
 * \param type_mask the type mask of this function
 * \return 0 when success, -1 when failure happens
 * \sa MXFuncInvoke
 */
MXNET_DLL int MXFuncDescribe(FunctionHandle fun,
                             mx_uint *num_use_vars,
                             mx_uint *num_scalars,
                             mx_uint *num_mutate_vars,
                             int *type_mask);
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
MXNET_DLL int MXFuncInvoke(FunctionHandle fun,
                           NArrayHandle *use_vars,
                           mx_float *scalar_args,
                           NArrayHandle *mutate_vars);

//--------------------------------------------
// Part 3: symbolic configuration generation
//--------------------------------------------
/*!
 * \brief list all the available AtomicSymbolEntry
 * \param out_size the size of returned array
 * \param out_array the output AtomicSymbolCreator array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                               AtomicSymbolCreator **out_array);
/*!
 * \brief Get the detailed information about atomic symbol.
 * \param creator the AtomicSymbolCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                          const char **name,
                                          const char **description,
                                          mx_uint *num_args,
                                          const char ***arg_names,
                                          const char ***arg_type_infos,
                                          const char ***arg_descriptions);
/*!
 * \brief Get the docstring of AtomicSymbol.
 * \param creator the AtomicSymbolCreator
 * \param out the returned name of the creator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetAtomicSymbolDoc(AtomicSymbolCreator creator,
                                         const char **out);
/*!
 * \brief Create an AtomicSymbol.
 * \param creator the AtomicSymbolCreator
 * \param num_param the number of parameters
 * \param keys the keys to the params
 * \param vals the vals of the params
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                                         int num_param,
                                         const char **keys,
                                         const char **vals,
                                         SymbolHandle *out);
/*!
 * \brief Create a Variable Symbol.
 * \param name name of the variable
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateVariable(const char *name, SymbolHandle *out);
/*!
 * \brief Create a Symbol by grouping list of symbols together
 * \param num_symbols number of symbols to be grouped
 * \param symbols array of symbol handles
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateGroup(mx_uint num_symbols,
                                  SymbolHandle *symbols,
                                  SymbolHandle *out);
/*!
 * \brief Create symbol from config.
 * \param cfg configuration string
 * \param out created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateFromConfig(const char *cfg,
                                       SymbolHandle *out);
/*!
 * \brief Free the symbol handle.
 * \param symbol the symbol
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolFree(SymbolHandle symbol);
/*!
 * \brief Copy the symbol to another handle
 * \param symbol the source symbol
 * \param out used to hold the result of copy
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCopy(SymbolHandle symbol, SymbolHandle *out);
/*!
 * \brief Print the content of symbol, used for debug.
 * \param symbol the symbol
 * \param out_str pointer to hold the output string of the printing.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolPrint(SymbolHandle symbol, const char **out_str);
/*!
 * \brief List arguments in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListArguments(SymbolHandle symbol,
                                    mx_uint *out_size,
                                    const char ***out_str_array);
/*!
 * \brief List returns in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListReturns(SymbolHandle symbol,
                                  mx_uint *out_size,
                                  const char ***out_str_array);
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
MXNET_DLL int MXSymbolCompose(SymbolHandle sym,
                              const char *name,
                              mx_uint num_args,
                              const char** keys,
                              SymbolHandle* args);
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
 * \param complete whether infer shape completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolInferShape(SymbolHandle sym,
                                 mx_uint num_args,
                                 const char** keys,
                                 const mx_uint *arg_ind_ptr,
                                 const mx_uint *arg_shape_data,
                                 mx_uint *in_shape_size,
                                 const mx_uint **in_shape_ndim,
                                 const mx_uint ***in_shape_data,
                                 mx_uint *out_shape_size,
                                 const mx_uint **out_shape_ndim,
                                 const mx_uint ***out_shape_data,
                                 int *complete);
//--------------------------------------------
// Part 4: Executor interface
//--------------------------------------------
/*!
 * \brief Executor forward method
 *
 * \param handle executor handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorForward(ExecutorHandle handle);
/*!
 * \brief Excecutor run backward
 *
 * \param handle execute handle
 * \param len lenth
 * \param head_grads NArray handle for heads' gradient
 *
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorBackward(ExecutorHandle handle,
                                 mx_uint len,
                                 NArrayHandle *head_grads);

/*!
 * \brief Get executor's head NArray
 *
 * \param handle executor handle
 * \param out_size output narray vector size
 * \param out out put narray handles
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorHeads(ExecutorHandle handle,
                              mx_uint *out_size,
                              NArrayHandle **out);

/*!
 * \brief Generate Executor from symbol
 *
 * \param symbol_handle symbol handle
 * \param dev_mask device mask
 * \param dev_id device id
 * \param len length
 * \param in_args in args array
 * \param arg_grad_store arg grads handle array
 * \param grad_req_type grad req array
 * \param out output executor handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorBind(SymbolHandle symbol_handle,
                             int dev_mask,
                             int dev_id,
                             mx_uint len,
                             NArrayHandle *in_args,
                             NArrayHandle *arg_grad_store,
                             mx_uint *grad_req_type,
                             ExecutorHandle *out);

//--------------------------------------------
// Part 5: IO Interface
//--------------------------------------------
/*!
 * \brief create an data iterator from configs string
 * \param cfg config string that contains the
 *    configuration about the iterator
 * \param out the handle to the iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXIOCreateFromConfig(const char *cfg,
                                   DataIterHandle *out);
/*!
 * \brief move iterator to next position
 * \param handle the handle to iterator
 * \param out return value of next
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXIONext(DataIterHandle handle,
                       int *out);
/*!
 * \brief call iterator.BeforeFirst
 * \param handle the handle to iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXIOBeforeFirst(DataIterHandle handle);
/*!
 * \brief free the handle to the IO module
 * \param handle the handle pointer to the data iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXIOFree(DataIterHandle handle);
/*!
 * \brief get the handle to the NArray of underlying data
 * \param handle the handle pointer to the data iterator
 * \param out handle to underlying data NArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXIOGetData(DataIterHandle handle,
                          NArrayHandle *out);
/*!
 * \brief get the handle to the NArray of underlying label
 * \param handle the handle pointer to the data iterator
 * \param out the handle to underlying label NArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXIOGetLabel(DataIterHandle handle,
                           NArrayHandle *out);

#endif  // MXNET_C_API_H_
