/*!
 *  Copyright (c) 2015 by Contributors
 * \file mxnet_api.h
 * \brief C API of mxnet
 */
#ifndef MXNET_API_H_
#define MXNET_API_H_

#ifdef __cplusplus
#define MXNET_EXTERN_C extern "C"
#endif

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
/*! \brief handle to a symbol that can be bind as operator */
typedef void *SymbolHandle;
/*! \brief handle to a NArrayOperator */
typedef void *OperatorHandle;
/*! \brief handle to a DataIterator */
typedef void *DataIterHandle;

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  MXGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 */
MXNET_DLL const char *MXGetLastError();

//--------------------------------
// Part 1: NArray creation and deletion
//--------------------------------
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
 * \param out_size the output function array
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
 * \brief get the name of function handle 
 * \param fun the function handle
 * \param out_name the name of the function
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXFuncGetName(FunctionHandle fun,
                            const char **out_name);
/*!
 * \brief get the argument requirements of the function
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
 * \brief create symbol from config
 * \param cfg configuration string
 * \param out created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymCreateFromConfig(const char *cfg,
                                    SymbolHandle *out);
/*!
 * \brief free the symbol handle
 * \param sym the symbol
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymFree(SymbolHandle *sym);
/*!
 * \brief set the parameter in to current symbol
 * \param sym the symbol
 * \param name name of the parameter
 * \param val value of the parameter
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymSetParam(SymbolHandle sym,
                            const char *name,
                            const char *val);
//--------------------------------------------
// Part 4: operator interface on NArray
//--------------------------------------------
/*!
 * \brief create operator from symbol
 * \param sym the symbol to create operator from
 * \param dev_mask device mask to indicate the device type
 * \param dev_id the device id we want to bind the symbol to
 * \param out the corresponding function handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXOpCreate(SymbolHandle sym,
                         int dev_mask,
                         int dev_id,
                         OperatorHandle *out);
/*!
 * \brief free the operator handle
 * \param op the handle to be freed
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXOpFree(OperatorHandle op);
/*!
 * \brief return an array to describe the arguments
 *  of this operator
 * \param out_size the size of output array
 * \oaram out_array the array of parameter requirments
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXOpDescribeArgs(mx_uint *out_size,
                               int **out_array);
/*!
 * \brief infer shape of unknown input shapes given the known one
 *  this function do not return the shape of output
 *  the shapes are packed into a CSR matrix represened by ind_ptr and shape_array
 *
 *  When the function returns, it return a new CSR matrix by updating ind_ptr,
 *  and return the content in the return value
 *
 * \param ind_ptr the head pointer of the rows in CSR
 * \param shape_array the content of the CSR
 * \param out_nout number of output arguments of this operation
 * \param out_array another content of CSR with infered shape
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXOpInferShape(mx_uint *ind_ptr,
                             mx_uint *shape_array,
                             mx_uint *out_nout,
                             mx_uint *out_array);
/*!
 * \brief call forward on the operator
 * \param op the operator handle
 * \param in_data array of input narray to the operator
 * \param out_data array of output NArray to hold the result
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXOpForward(OperatorHandle op,
                          NArrayHandle *in_data,
                          NArrayHandle *out_data);
/*!
 * \brief call backward on the operator
 * \param op the operator handle
 * \param grad_next array of output gradients
 * \param in_data array of input narray to the operator
 * \param out_grad array to holds the gradient on these input
 *    can be NULL if that position request is kNullOp
 * \param reqs gradient request type
 * \return 0 when success, -1 when failure happens
 * \sa mxnet::Operator::GradReqType
 */
MXNET_DLL int MXOpBackward(OperatorHandle op,
                           NArrayHandle *grad_next,
                           NArrayHandle *in_data,
                           NArrayHandle *out_grad,
                           mx_uint *reqs);

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

#endif  // MXNET_WRAPPER_H_
