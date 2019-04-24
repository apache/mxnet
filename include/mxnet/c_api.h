/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api.h
 * \brief C API of mxnet
 */
#ifndef MXNET_C_API_H_
#define MXNET_C_API_H_

/*! \brief Inhibit C++ name-mangling for MXNet functions. */
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/*! \brief Keep the default value in C++ */
#ifdef __cplusplus
#define DEFAULT(x) = x
#else
#define DEFAULT(x)
#endif  // __cplusplus

#include <stdint.h>

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*! \brief MXNET_DLL prefix for windows */
#ifdef _WIN32
#ifdef MXNET_EXPORTS
#define MXNET_DLL __declspec(dllexport)
#else
#define MXNET_DLL __declspec(dllimport)
#endif
#else
#define MXNET_DLL
#endif

/*! \brief manually define unsigned int */
typedef unsigned int mx_uint;
/*! \brief manually define float */
typedef float mx_float;
/*! \brief data type to store dim size */
typedef int64_t dim_t;
// all the handles are simply void *
// will be casted internally to specific pointers types
// these typedefs are mainly used for readablity reasons
/*! \brief handle to NDArray */
typedef void *NDArrayHandle;
/*! \brief handle to a mxnet narray function that changes NDArray */
typedef const void *FunctionHandle;
/*! \brief handle to a function that takes param and creates symbol */
typedef void *AtomicSymbolCreator;
/*! \brief handle to cached operator */
typedef void *CachedOpHandle;
/*! \brief handle to a symbol that can be bind as operator */
typedef void *SymbolHandle;
/*! \brief handle to a AtomicSymbol */
typedef void *AtomicSymbolHandle;
/*! \brief handle to an Executor */
typedef void *ExecutorHandle;
/*! \brief handle a dataiter creator */
typedef void *DataIterCreator;
/*! \brief handle to a DataIterator */
typedef void *DataIterHandle;
/*! \brief handle to KVStore */
typedef void *KVStoreHandle;
/*! \brief handle to RecordIO */
typedef void *RecordIOHandle;
/*! \brief handle to MXRtc*/
typedef void *RtcHandle;
/*! \brief handle to rtc cuda module*/
typedef void *CudaModuleHandle;
/*! \brief handle to rtc cuda kernel*/
typedef void *CudaKernelHandle;
/*! \brief handle to a Profile object (domain, duration, counter, etc.) */
typedef void *ProfileHandle;
/*! \brief handle to DLManagedTensor*/
typedef void *DLManagedTensorHandle;
/*! \brief handle to Context */
typedef const void *ContextHandle;
/*! \brief handle to Engine FnProperty */
typedef const void *EngineFnPropertyHandle;
/*! \brief handle to Engine VarHandle */
typedef void *EngineVarHandle;

/*! \brief Engine asynchronous operation */
typedef void (*EngineAsyncFunc)(void*, void*, void*);
/*! \brief Engine synchronous operation */
typedef void (*EngineSyncFunc)(void*, void*);
/*! \brief Callback to free the param for EngineAsyncFunc/EngineSyncFunc */
typedef void (*EngineFuncParamDeleter)(void*);
typedef void (*ExecutorMonitorCallback)(const char*,
                                        NDArrayHandle,
                                        void*);

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

typedef int (*MXGenericCallback)(void);

struct MXCallbackList {
  int num_callbacks;
  int (**callbacks)(void);
  void **contexts;
};

enum CustomOpCallbacks {
  kCustomOpDelete,
  kCustomOpForward,
  kCustomOpBackward
};

enum CustomOpPropCallbacks {
  kCustomOpPropDelete,
  kCustomOpPropListArguments,
  kCustomOpPropListOutputs,
  kCustomOpPropListAuxiliaryStates,
  kCustomOpPropInferShape,
  kCustomOpPropDeclareBackwardDependency,
  kCustomOpPropCreateOperator,
  kCustomOpPropInferType,
  kCustomOpPropInferStorageType,
  kCustomOpPropBackwardInferStorageType
};


typedef int (*CustomOpFBFunc)(int /*size*/, void** /*ptrs*/, int* /*tags*/,
                              const int* /*reqs*/, const int /*is_train*/,
                              void* /*state*/);
typedef int (*CustomOpDelFunc)(void* /*state*/);
typedef int (*CustomOpListFunc)(char*** /*args*/, void* /*state*/);
typedef int (*CustomOpInferShapeFunc)(int /*num_input*/, int* /*ndims*/,
                                      unsigned** /*shapes*/, void* /*state*/);
typedef int (*CustomOpInferStorageTypeFunc)(int /*num_input*/, int* /*stypes*/, void* /*state*/);
typedef int (*CustomOpBackwardInferStorageTypeFunc)(int /*num_input*/,
                                                    int * /*stypes*/,
                                                    int * /*tags*/,
                                                    void * /*state*/);
typedef int (*CustomOpInferTypeFunc)(int /*num_input*/, int* /*types*/, void* /*state*/);
typedef int (*CustomOpBwdDepFunc)(const int* /*out_grad*/, const int* /*in_data*/,
                                  const int* /*out_data*/, int* /*num_deps*/,
                                  int** /*rdeps*/, void* /*state*/);
typedef int (*CustomOpCreateFunc)(const char* /*ctx*/, int /*num_inputs*/,
                                  unsigned** /*shapes*/, const int* /*ndims*/,
                                  const int* /*dtypes*/, struct MXCallbackList* /*ret*/,
                                  void* /*state*/);
typedef int (*CustomOpPropCreator)(const char* /*op_type*/, const int /*num_kwargs*/,
                                   const char** /*keys*/, const char** /*values*/,
                                   struct MXCallbackList* /*ret*/);


enum CustomFunctionCallbacks {
  kCustomFunctionBackward,
  kCustomFunctionDelete
};

typedef int (*CustomFunctionBwdFunc)(int /*num_ograds*/, int /*num_igrads*/, void** /*ptrs*/,
                                     const int* /*reqs*/, const int /*is_train*/,
                                     void* /*state*/);
typedef int (*CustomFunctionDelFunc)(void* /*state*/);

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
// Part 0: Global State setups
//-------------------------------------
/*!
 * \brief Seed all global random number generators in mxnet.
 * \param seed the random number seed.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXRandomSeed(int seed);

/*!
 * \brief Seed the global random number generator of the given device.
 * \param seed the random number seed.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXRandomSeedContext(int seed, int dev_type, int dev_id);

/*!
 * \brief Notify the engine about a shutdown,
 *  This can help engine to print less messages into display.
 *
 *  User do not have to call this function.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXNotifyShutdown();

/*!
 * \brief Set up configuration of profiler for the process passed as profile_process in keys
 * \param num_params Number of parameters
 * \param keys array of parameter keys
 * \param vals array of parameter values
 * \param kvstoreHandle handle to kvstore
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetProcessProfilerConfig(int num_params, const char* const* keys,
                                         const char* const* vals,
                                         KVStoreHandle kvstoreHandle);

/*!
 * \brief Set up configuration of profiler for worker/current process
 * \param num_params Number of parameters
 * \param keys array of parameter keys
 * \param vals array of parameter values
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetProfilerConfig(int num_params, const char* const* keys, const char* const* vals);

/*!
 * \brief Set up state of profiler for either worker or server process
 * \param state indicate the working state of profiler,
 *  profiler not running when state == 0,
 *  profiler running when state == 1
 * \param profile_process an int,
 * when 0 command is for worker/current process,
 * when 1 command is for server process
 * \param kvstoreHandle handle to kvstore, needed for server process profiling
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetProcessProfilerState(int state, int profile_process,
                                        KVStoreHandle kvStoreHandle);

/*!
 * \brief Set up state of profiler for current process
 * \param state indicate the working state of profiler,
 *  profiler not running when state == 0,
 *  profiler running when state == 1
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetProfilerState(int state);

/*!
 * \brief Save profile and stop profiler
 * \param finished true if stat output should stop after this point
 * \param profile_process an int,
 * when 0 command is for worker/current process,
 * when 1 command is for server process
 * \param kvstoreHandle handle to kvstore
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXDumpProcessProfile(int finished, int profile_process, KVStoreHandle kvStoreHandle);


/*!
 * \brief Save profile and stop profiler for worker/current process
 * \param finished true if stat output should stop after this point
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXDumpProfile(int finished);

/*!
 * \brief Print aggregate stats to the a string
 * \param out_str Will receive a pointer to the output string
 * \param reset Clear the aggregate stats after printing
 * \return 0 when success, -1 when failure happens.
 * \note
 */
MXNET_DLL int MXAggregateProfileStatsPrint(const char **out_str, int reset);

/*!
 * \brief Pause profiler tuning collection
 * \param paused If nonzero, profiling pauses. Otherwise, profiling resumes/continues
 * \param profile_process integer which denotes whether to process worker or server process
 * \param kvstoreHandle handle to kvstore
 * \return 0 when success, -1 when failure happens.
 * \note pausing and resuming is global and not recursive
 */
MXNET_DLL int MXProcessProfilePause(int paused, int profile_process, KVStoreHandle kvStoreHandle);

/*!
 * \brief Pause profiler tuning collection for worker/current process
 * \param paused If nonzero, profiling pauses. Otherwise, profiling resumes/continues
 * \return 0 when success, -1 when failure happens.
 * \note pausing and resuming is global and not recursive
 */
MXNET_DLL int MXProfilePause(int paused);

/*!
 * \brief Create profiling domain
 * \param domain String representing the domain name to create
 * \param out Return domain object
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateDomain(const char *domain, ProfileHandle *out);

/*!
 * \brief Create profile task
 * \param name Name of the task
 * \param domain Domain of the task
 * \param out Output handle
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateTask(ProfileHandle domain,
                                  const char *task_name,
                                  ProfileHandle *out);

/*!
 * \brief Create profile frame
 * \param name Name of the frame
 * \param domain Domain of the frame
 * \param out Output handle
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateFrame(ProfileHandle domain,
                                   const char *frame_name,
                                   ProfileHandle *out);

/*!
 * \brief Create profile event
 * \param name Name of the event
 * \param out Output handle
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateEvent(const char *event_name, ProfileHandle *out);

/*!
 * \brief Create profile counter
 * \param name Name of the counter
 * \param domain Domain of the counter
 * \param out Output handle
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateCounter(ProfileHandle domain,
                                     const char *counter_name,
                                     ProfileHandle *out);

/*!
 * \brief Destroy a frame
 * \param frame_handle Handle to frame to destroy
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileDestroyHandle(ProfileHandle frame_handle);

/*!
 * \brief Start timing the duration of a profile duration object such as an event, task or frame
 * \param duration_handle handle to the duration object
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileDurationStart(ProfileHandle duration_handle);

/*!
 * \brief Stop timing the duration of a profile duration object such as an event, task or frame
 * \param duration_handle handle to the duration object
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileDurationStop(ProfileHandle duration_handle);

/*!
 * \brief Set a counter, given its handle
 * \param counter_handle Handle to counter to set
 * \param value Value to set the counter to (64-bit unsigned integer)
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileSetCounter(ProfileHandle counter_handle, uint64_t value);

/*!
 * \brief Adjust a counter by the given amount, given its handle
 * \param counter_handle Handle to counter to adjust
 * \param value Value to adjust the counter by (64-bit signed integer)
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileAdjustCounter(ProfileHandle counter_handle, int64_t value);

/*!
 * \brief Mark a single instant in time
 * \param domain Domain of the marker
 * \param instant_marker_name Name of the marker
 * \param scope Scope of marker ('global', 'process', 'thread', 'task', 'marker')
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileSetMarker(ProfileHandle domain,
                                 const char *instant_marker_name,
                                 const char *scope);

/*!
 * \brief Set the number of OMP threads to use
 * \param thread_num Number of OMP threads desired
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetNumOMPThreads(int thread_num);

/*!
 * \brief set bulk execution limit
 * \param bulk_size new bulk_size
 * \param prev_bulk_size previous bulk_size
 */
MXNET_DLL int MXEngineSetBulkSize(int bulk_size, int* prev_bulk_size);

/*!
 * \brief Get the number of GPUs.
 * \param pointer to int that will hold the number of GPUs available.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXGetGPUCount(int* out);

/*!
 * \brief get the free and total available memory on a GPU
 *  Note: Deprecated, use MXGetGPUMemoryInformation64 instead.
 * \param dev the GPU number to query
 * \param free_mem pointer to the integer holding free GPU memory
 * \param total_mem pointer to the integer holding total GPU memory
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXGetGPUMemoryInformation(int dev, int *free_mem, int *total_mem);

/*!
 * \brief get the free and total available memory on a GPU
 * \param dev the GPU number to query
 * \param free_mem pointer to the uint64_t holding free GPU memory
 * \param total_mem pointer to the uint64_t holding total GPU memory
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXGetGPUMemoryInformation64(int dev, uint64_t *free_mem, uint64_t *total_mem);

/*!
 * \brief get the MXNet library version as an integer
 * \param pointer to the integer holding the version number
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXGetVersion(int *out);

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
MXNET_DLL int MXNDArrayCreateNone(NDArrayHandle *out);
/*!
 * \brief create a NDArray with specified shape
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *    the narray is first mutated
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreate(const mx_uint *shape,
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
 *    the narray is first mutated
 * \param dtype data type of created array
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreateEx(const mx_uint *shape,
                              mx_uint ndim,
                              int dev_type,
                              int dev_id,
                              int delay_alloc,
                              int dtype,
                              NDArrayHandle *out);


/*!
 * \brief create an empty sparse NDArray with specified shape and data type
 * \param storage_type the storage type of the ndarray
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *        the narray is first mutated
 * \param dtype data type of created array
 * \param num_aux the number of aux data to support this ndarray
 * \param aux_type data type of the aux data for the created array
 * \param aux_ndims the dimension of the shapes of aux data
 * \param aux_shape the shapes of aux data
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreateSparseEx(int storage_type,
                                      const mx_uint *shape,
                                      mx_uint ndim,
                                      int dev_type,
                                      int dev_id,
                                      int delay_alloc,
                                      int dtype,
                                      mx_uint num_aux,
                                      int *aux_type,
                                      mx_uint *aux_ndims,
                                      const mx_uint *aux_shape,
                                      NDArrayHandle *out);

/*!
 * \brief create a NDArray handle that is loaded from raw bytes.
 * \param buf the head of the raw bytes
 * \param size size of the raw bytes
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayLoadFromRawBytes(const void *buf,
                                        size_t size,
                                        NDArrayHandle *out);
/*!
 * \brief save the NDArray into raw bytes.
 * \param handle the NDArray handle
 * \param out_size size of the raw bytes
 * \param out_buf the head of returning memory bytes.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySaveRawBytes(NDArrayHandle handle,
                                    size_t *out_size,
                                    const char **out_buf);
/*!
 * \brief Save list of narray into the file.
 * \param fname name of the file.
 * \param num_args number of arguments to save.
 * \param args the array of NDArrayHandles to be saved.
 * \param keys the name of the NDArray, optional, can be NULL
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySave(const char* fname,
                            mx_uint num_args,
                            NDArrayHandle* args,
                            const char** keys);
/*!
 * \brief Load list of narray from the file.
 * \param fname name of the file.
 * \param out_size number of narray loaded.
 * \param out_arr head of the returning narray handles.
 * \param out_name_size size of output name arrray.
 * \param out_names the names of returning NDArrays, can be NULL
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayLoad(const char* fname,
                            mx_uint *out_size,
                            NDArrayHandle** out_arr,
                            mx_uint *out_name_size,
                            const char*** out_names);

/*!
 * \brief Load list / dictionary of narrays from file content loaded into memory.
 * This will load a list of ndarrays in a similar
 * manner to MXNDArrayLoad, however, it loads from
 * buffer containing the contents of a file, rather than
 * from a specified file.
 * \param ndarray_buffer pointer to the start of the ndarray file content
 * \param size size of the file
 * \param out_size number of narray loaded.
 * \param out_arr head of the returning narray handles.
 * \param out_name_size size of output name arrray.
 * \param out_names the names of returning NDArrays, can be NULL
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayLoadFromBuffer(const void *ndarray_buffer,
                            size_t size,
                            mx_uint *out_size,
                            NDArrayHandle** out_arr,
                            mx_uint *out_name_size,
                            const char*** out_names);

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
MXNET_DLL int MXNDArraySyncCopyFromCPU(NDArrayHandle handle,
                                       const void *data,
                                       size_t size);
/*!
 * \brief Perform a synchronize copyto a continugous CPU memory region.
 *
 *  This function will call WaitToRead before the copy is performed.
 *  This is useful to copy data from existing memory region that are
 *  not wrapped by NDArray(thus dependency not being tracked).
 *
 * \param handle the NDArray handle
 * \param data the data source to copy into.
 * \param size the memory size we want to copy into.
 */
MXNET_DLL int MXNDArraySyncCopyToCPU(NDArrayHandle handle,
                                     void *data,
                                     size_t size);
/*!
 * \brief Copy src.data() to dst.data() if i = -1, else dst.aux_data(i) if i >= 0
 * This function blocks. Do not use it in performance critical code.
 * \param handle_dst handle of a dst ndarray whose data/aux_data has been allocated
 * \param handle_src handle of a src ndarray which has default storage type
 * \param i dst data blob indicator
 */
MXNET_DLL int MXNDArraySyncCopyFromNDArray(NDArrayHandle handle_dst,
                                           const NDArrayHandle handle_src,
                                           const int i);

/*!
 * \brief check whether the NDArray format is valid
 * \param full_check if `True`, rigorous check, O(N) operations
 *    Otherwise basic check, O(1) operations
 */
MXNET_DLL int MXNDArraySyncCheckFormat(NDArrayHandle handle, const bool full_check);
/*!
 * \brief Wait until all the pending writes with respect NDArray are finished.
 *  Always call this before read data out synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayWaitToRead(NDArrayHandle handle);
/*!
 * \brief Wait until all the pending read/write with respect NDArray are finished.
 *  Always call this before write data into NDArray synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayWaitToWrite(NDArrayHandle handle);
/*!
 * \brief wait until all delayed operations in
 *   the system is completed
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayWaitAll();
/*!
 * \brief free the narray handle
 * \param handle the handle to be freed
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayFree(NDArrayHandle handle);
/*!
 * \brief Slice the NDArray along axis 0.
 * \param handle the handle to the NDArray
 * \param slice_begin The beginning index of slice
 * \param slice_end The ending index of slice
 * \param out The NDArrayHandle of sliced NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySlice(NDArrayHandle handle,
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
MXNET_DLL int MXNDArrayAt(NDArrayHandle handle,
                          mx_uint idx,
                          NDArrayHandle *out);

/*!
 * \brief get the storage type of the array
 */
MXNET_DLL int MXNDArrayGetStorageType(NDArrayHandle handle,
                                      int *out_storage_type);

/*!
 * \brief Reshape the NDArray.
 * \param handle the handle to the narray
 * \param ndim number of dimensions of new shape
 * \param dims new shape
 * \param out the NDArrayHandle of reshaped NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayReshape(NDArrayHandle handle,
                               int ndim,
                               int *dims,
                               NDArrayHandle *out);

/*!
 * \brief Reshape the NDArray.
 * \param handle the handle to the narray
 * \param ndim number of dimensions of new shape
 * \param dims new shape
 * \param out the NDArrayHandle of reshaped NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayReshape64(NDArrayHandle handle,
                                 int ndim,
                                 dim_t *dims,
                                 bool reverse,
                                 NDArrayHandle *out);
/*!
 * \brief get the shape of the array
 * \param handle the handle to the narray
 * \param out_dim the output dimension
 * \param out_pdata pointer holder to get data pointer of the shape
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetShape(NDArrayHandle handle,
                                mx_uint *out_dim,
                                const mx_uint **out_pdata);
/*!
 * \brief get the content of the data in NDArray
 * \param handle the handle to the ndarray
 * \param out_pdata pointer holder to get pointer of data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetData(NDArrayHandle handle,
                               void **out_pdata);
/*!
* \brief Create a reference view of NDArray that
*  represents as DLManagedTensor
*  Notice: MXNet uses asynchronous execution. Please call MXNDArrayWaitToRead or
*          MXNDArrayWaitToWrite before calling MXNDArrayToDLPack.
* \param handle the handle to the ndarray
* \param out_dlpack pointer holder to get pointer of DLManagedTensor
* \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXNDArrayToDLPack(NDArrayHandle handle,
                                       DLManagedTensorHandle *out_dlpack);

/*!
* \brief Create a NDArray backed by a dlpack tensor.
*
* This allows us to create a NDArray using the memory
* allocated by an external deep learning framework
* that is DLPack compatible.
*
* The memory is retained until the NDArray went out of scope.
*
* \param dlpack the pointer of the input DLManagedTensor
* \param out_handle pointer holder to get pointer of NDArray
* \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXNDArrayFromDLPack(DLManagedTensorHandle dlpack,
                                  NDArrayHandle *out_handle);
/*!
 * \brief Delete a dlpack tensor
 * \param dlpack the pointer of the input DLManagedTensor
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCallDLPackDeleter(DLManagedTensorHandle dlpack);

/*!
 * \brief get the type of the data in NDArray
 * \param handle the handle to the narray
 * \param out_dtype pointer holder to get type of data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetDType(NDArrayHandle handle,
                               int *out_dtype);

/*!
 * \brief get the type of the ith aux data in NDArray
 * \param handle the handle to the narray
 * \param i the index of the aux data
 * \param out_type pointer holder to get type of aux data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetAuxType(NDArrayHandle handle,
                                  mx_uint i,
                                  int *out_type);

/*!
 * \brief Get a deep copy of the ith aux data blob
 * in the form of an NDArray of default storage type.
 * This function blocks. Do not use it in performance critical code.
 */
MXNET_DLL int MXNDArrayGetAuxNDArray(NDArrayHandle handle,
                                     mx_uint i,
                                     NDArrayHandle *out);

/*!
 * \brief Get a deep copy of the data blob
 * in the form of an NDArray of default storage type.
 * This function blocks. Do not use it in performance critical code.
 */
MXNET_DLL int MXNDArrayGetDataNDArray(NDArrayHandle handle,
                                      NDArrayHandle *out);
/*!
 * \brief get the context of the NDArray
 * \param handle the handle to the narray
 * \param out_dev_type the output device type
 * \param out_dev_id the output device id
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetContext(NDArrayHandle handle,
                                  int *out_dev_type,
                                  int *out_dev_id);
/*!
 * \brief return gradient buffer attached to this NDArray
 * \param handle NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetGrad(NDArrayHandle handle, NDArrayHandle *out);
/*!
 * \brief detach and ndarray from computation graph by clearing entry_
 * \param handle NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayDetach(NDArrayHandle handle, NDArrayHandle *out);
/*!
 * \brief set the flag for gradient array state.
 * \param handle NDArray handle
 * \param state the new state.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySetGradState(NDArrayHandle handle, int state);
/*!
 * \brief set the flag for gradient array state.
 * \param handle NDArray handle
 * \param state the new state.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetGradState(NDArrayHandle handle, int *out);
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
 * \param arg_type_infos Type information about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \param return_type Return type of the function.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXFuncGetInfo(FunctionHandle fun,
                            const char **name,
                            const char **description,
                            mx_uint *num_args,
                            const char ***arg_names,
                            const char ***arg_type_infos,
                            const char ***arg_descriptions,
                            const char **return_type DEFAULT(NULL));
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
                           NDArrayHandle *use_vars,
                           mx_float *scalar_args,
                           NDArrayHandle *mutate_vars);
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
MXNET_DLL int MXFuncInvokeEx(FunctionHandle fun,
                             NDArrayHandle *use_vars,
                             mx_float *scalar_args,
                             NDArrayHandle *mutate_vars,
                             int num_params,
                             char **param_keys,
                             char **param_vals);
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
MXNET_DLL int MXImperativeInvoke(AtomicSymbolCreator creator,
                                 int num_inputs,
                                 NDArrayHandle *inputs,
                                 int *num_outputs,
                                 NDArrayHandle **outputs,
                                 int num_params,
                                 const char **param_keys,
                                 const char **param_vals);
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
 * \param out_stypes output ndarrays' stypes
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXImperativeInvokeEx(AtomicSymbolCreator creator,
                                   int num_inputs,
                                   NDArrayHandle *inputs,
                                   int *num_outputs,
                                   NDArrayHandle **outputs,
                                   int num_params,
                                   const char **param_keys,
                                   const char **param_vals,
                                   const int **out_stypes);
/*!
 * \brief set whether to record operator for autograd
 * \param is_recording 1 when recording, 0 when not recording.
 * \param prev returns the previous status before this set.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradSetIsRecording(int is_recording, int* prev);
/*!
 * \brief set whether to record operator for autograd
 * \param is_training 1 when training, 0 when testing
 * \param prev returns the previous status before this set.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradSetIsTraining(int is_training, int* prev);
/*!
 * \brief get whether autograd recording is on
 * \param curr returns the current status.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradIsRecording(bool* curr);
/*!
 * \brief get whether training mode is on
 * \param curr returns the current status.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradIsTraining(bool* curr);
/*!
 * \brief mark NDArrays as variables to compute gradient for autograd
 * \param num_var number of variable NDArrays
 * \param var_handles variable NDArrays
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradMarkVariables(mx_uint num_var,
                                      NDArrayHandle *var_handles,
                                      mx_uint *reqs_array,
                                      NDArrayHandle *grad_handles);
/*!
 * \brief compute the gradient of outputs w.r.t variabels
 * \param num_output number of output NDArray
 * \param output_handles output NDArrays
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradComputeGradient(mx_uint num_output,
                                        NDArrayHandle* output_handles);
/*!
 * \brief compute the gradient of outputs w.r.t variabels
 * \param num_output number of output NDArray
 * \param output_handles output NDArrays
 * \param ograd_handles head gradient for NDArrays
 * \param retain_graph whether to keep the graph after backward
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradBackward(mx_uint num_output,
                                 NDArrayHandle* output_handles,
                                 NDArrayHandle* ograd_handles,
                                 int retain_graph);
/*!
 * \brief compute the gradient of outputs w.r.t variabels
 * \param num_output number of output NDArray
 * \param output_handles output NDArrays
 * \param ograd_handles head gradient for NDArrays
 * \param num_variables number of variables
 * \param
 * \param retain_graph whether to keep the graph after backward
 * \param is_train whether to do backward for training or inference
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradBackwardEx(mx_uint num_output,
                                   NDArrayHandle *output_handles,
                                   NDArrayHandle *ograd_handles,
                                   mx_uint num_variables,
                                   NDArrayHandle *var_handles,
                                   int retain_graph,
                                   int create_graph,
                                   int is_train,
                                   NDArrayHandle **grad_handles,
                                   int **grad_stypes);
/*
 * \brief get the graph constructed by autograd.
 * \param handle ndarray handle
 * \param out output symbol handle
 */
MXNET_DLL int MXAutogradGetSymbol(NDArrayHandle handle, SymbolHandle *out);
/*!
 * \brief create cached operator
 */
MXNET_DLL int MXCreateCachedOp(SymbolHandle handle, CachedOpHandle *out);
/*!
 * \brief create cached operator
 */
MXNET_DLL int MXCreateCachedOpEx(SymbolHandle handle,
                                 int num_flags,
                                 const char** keys,
                                 const char** vals,
                                 CachedOpHandle *out);
/*!
 * \brief free cached operator
 */
MXNET_DLL int MXFreeCachedOp(CachedOpHandle handle);
/*!
 * \brief invoke cached operator
 */
MXNET_DLL int MXInvokeCachedOp(CachedOpHandle handle,
                               int num_inputs,
                               NDArrayHandle *inputs,
                               int *num_outputs,
                               NDArrayHandle **outputs);
/*!
 * \brief invoke a cached op
 * \param handle the handle to the cached op
 * \param num_inputs number of input NDArrays
 * \param inputs input NDArrays
 * \param num_outputs number of output NDArrays
 * \param outputs output NDArrays
 * \param out_stypes output ndarrays' stypes
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXInvokeCachedOpEx(CachedOpHandle handle,
                                 int num_inputs,
                                 NDArrayHandle *inputs,
                                 int *num_outputs,
                                 NDArrayHandle **outputs,
                                 const int** out_stypes);

//--------------------------------------------
// Part 3: symbolic configuration generation
//--------------------------------------------
/*!
 * \brief list all the available operator names, include entries.
 * \param out_size the size of returned array
 * \param out_array the output operator name array.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXListAllOpNames(mx_uint *out_size,
                               const char ***out_array);
/*!
 * \brief list all the available AtomicSymbolEntry
 * \param out_size the size of returned array
 * \param out_array the output AtomicSymbolCreator array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAtomicSymbolCreators(mx_uint *out_size,
                                               AtomicSymbolCreator **out_array);

/*!
 * \brief Get the name of an atomic symbol.
 * \param creator the AtomicSymbolCreator.
 * \param name The returned name of the creator.
 */
MXNET_DLL int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator creator,
                                          const char **name);

/*!
 * \brief Get the input symbols of the graph.
 * \param sym The graph.
 * \param inputs The input symbols of the graph.
 * \param input_size the number of input symbols returned.
 */
MXNET_DLL int MXSymbolGetInputSymbols(SymbolHandle sym, SymbolHandle **inputs,
                                      int *input_size);

/*!
 * \brief Cut a subgraph whose nodes are marked with a subgraph attribute.
 * The input graph will be modified. A variable node will be created for each
 * edge that connects to nodes outside the subgraph. The outside nodes that
 * connect to the subgraph will be returned.
 * \param sym The graph.
 * \param inputs The nodes that connect to the subgraph.
 * \param input_size The number of such nodes.
 */
MXNET_DLL int MXSymbolCutSubgraph(SymbolHandle sym, SymbolHandle **inputs,
                                  int *input_size);

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
MXNET_DLL int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                          const char **name,
                                          const char **description,
                                          mx_uint *num_args,
                                          const char ***arg_names,
                                          const char ***arg_type_infos,
                                          const char ***arg_descriptions,
                                          const char **key_var_num_args,
                                          const char **return_type DEFAULT(NULL));
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
 * \brief Load a symbol from a json file.
 * \param fname the file name.
 * \param out the output symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateFromFile(const char *fname, SymbolHandle *out);
/*!
 * \brief Load a symbol from a json string.
 * \param json the json string.
 * \param out the output symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateFromJSON(const char *json, SymbolHandle *out);
/*!
 * \brief Save a symbol into a json file.
 * \param symbol the input symbol.
 * \param fname the file name.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolSaveToFile(SymbolHandle symbol, const char *fname);
/*!
 * \brief Save a symbol into a json string
 * \param symbol the input symbol.
 * \param out_json output json string.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolSaveToJSON(SymbolHandle symbol, const char **out_json);
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
 * \brief Get string name from symbol
 * \param symbol the source symbol
 * \param out The result name.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetName(SymbolHandle symbol,
                              const char** out,
                              int *success);
/*!
 * \brief Get string attribute from symbol
 * \param symbol the source symbol
 * \param key The key of the symbol.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetAttr(SymbolHandle symbol,
                              const char* key,
                              const char** out,
                              int *success);
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
MXNET_DLL int MXSymbolSetAttr(SymbolHandle symbol,
                              const char* key,
                              const char* value);
/*!
 * \brief Get all attributes from symbol, including all descendents.
 * \param symbol the source symbol
 * \param out_size The number of output attributes
 * \param out 2*out_size strings representing key value pairs.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAttr(SymbolHandle symbol,
                               mx_uint *out_size,
                               const char*** out);
/*!
 * \brief Get all attributes from symbol, excluding descendents.
 * \param symbol the source symbol
 * \param out_size The number of output attributes
 * \param out 2*out_size strings representing key value pairs.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAttrShallow(SymbolHandle symbol,
                                      mx_uint *out_size,
                                      const char*** out);
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
MXNET_DLL int MXSymbolListOutputs(SymbolHandle symbol,
                                  mx_uint *out_size,
                                  const char ***out_str_array);

/*!
 * \brief Get number of outputs of the symbol.
 * \param symbol The symbol
 * \param out_size number of outputs
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetNumOutputs(SymbolHandle symbol,
                                     mx_uint *output_count);

/*!
 * \brief Get a symbol that contains all the internals.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are all the internals.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetInternals(SymbolHandle symbol,
                                   SymbolHandle *out);
/*!
 * \brief Get a symbol that contains only direct children.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are the direct children.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetChildren(SymbolHandle symbol,
                                  SymbolHandle *out);
/*!
 * \brief Get index-th outputs of the symbol.
 * \param symbol The symbol
 * \param index the Index of the output.
 * \param out The output symbol whose outputs are the index-th symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetOutput(SymbolHandle symbol,
                                mx_uint index,
                                SymbolHandle *out);

/*!
 * \brief List auxiliary states in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
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
 * \brief Get the gradient graph of the symbol
 *
 * \param sym the symbol to get gradient
 * \param num_wrt number of arguments to get gradient
 * \param wrt the name of the arguments to get gradient
 * \param out the returned symbol that has gradient
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGrad(SymbolHandle sym,
                           mx_uint num_wrt,
                           const char** wrt,
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
                                 mx_uint *aux_shape_size,
                                 const mx_uint **aux_shape_ndim,
                                 const mx_uint ***aux_shape_data,
                                 int *complete);
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
MXNET_DLL int MXSymbolInferShapePartial(SymbolHandle sym,
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
                                        mx_uint *aux_shape_size,
                                        const mx_uint **aux_shape_ndim,
                                        const mx_uint ***aux_shape_data,
                                        int *complete);

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
MXNET_DLL int MXSymbolInferType(SymbolHandle sym,
                                mx_uint num_args,
                                const char** keys,
                                const int *arg_type_data,
                                mx_uint *in_type_size,
                                const int **in_type_data,
                                mx_uint *out_type_size,
                                const int **out_type_data,
                                mx_uint *aux_type_size,
                                const int **aux_type_data,
                                int *complete);

/*!
 * \brief Convert a symbol into a quantized symbol where FP32 operators are replaced with INT8
 * \param sym_handle symbol to be converted
 * \param ret_sym_handle quantized symbol result
 * \param num_excluded_symbols number of layers excluded from being quantized in the input symbol
 * \param excluded_symbols op names to be excluded from being quantized
 * \param num_offline number of parameters that are quantized offline
 * \param offline_params array of c strings representing the names of params quantized offline
 * \param quantized_dtype the quantized destination type for input data.
 * \param calib_quantize whether calibrate quantize op with offline calibration data.
 */
MXNET_DLL int MXQuantizeSymbol(SymbolHandle sym_handle, SymbolHandle *ret_sym_handle,
                               const mx_uint num_excluded_symbols,
                               const char **excluded_symbols,
                               const mx_uint num_offline, const char **offline_params,
                               const char *quantized_dtype, const bool calib_quantize);

/*!
 * \brief Set calibration table to node attributes in the sym
 * \param sym_handle symbol whose node attributes are to be set by calibration table
 * \param num_layers number of layers in the calibration table
 * \param layer names stored as keys in the calibration table
 * \param low_quantiles low quantiles of layers stored in the calibration table
 * \param high_quantiles high quantiles of layers stored in the calibration table
 * \param ret_sym_handle returned symbol
 */
MXNET_DLL int MXSetCalibTableToQuantizedSymbol(SymbolHandle qsym_handle,
                                               const mx_uint num_layers,
                                               const char** layer_names,
                                               const float* low_quantiles,
                                               const float* high_quantiles,
                                               SymbolHandle* ret_sym_handle);

/*!
 * \brief Run subgraph pass based on the backend provided
 * \param sym_handle symbol to be converted
 * \param backend backend names for subgraph pass
 * \param ret_sym_handle returned symbol
 */
MXNET_DLL int MXGenBackendSubgraph(SymbolHandle sym_handle, const char *backend,
                                   SymbolHandle *ret_sym_handle);

//--------------------------------------------
// Part 4: Executor interface
//--------------------------------------------
/*!
 * \brief Delete the executor
 * \param handle the executor.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorFree(ExecutorHandle handle);
/*!
 * \brief Print the content of execution plan, used for debug.
 * \param handle the executor.
 * \param out_str pointer to hold the output string of the printing.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorPrint(ExecutorHandle handle, const char **out_str);
/*!
 * \brief Executor forward method
 *
 * \param handle executor handle
 * \param is_train int value to indicate whether the forward pass is for evaluation
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorForward(ExecutorHandle handle, int is_train);
/*!
 * \brief Excecutor run backward
 *
 * \param handle execute handle
 * \param len lenth
 * \param head_grads NDArray handle for heads' gradient
 *
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorBackward(ExecutorHandle handle,
                                 mx_uint len,
                                 NDArrayHandle *head_grads);
/*!
 * \brief Excecutor run backward
 *
 * \param handle execute handle
 * \param len lenth
 * \param head_grads NDArray handle for heads' gradient
 * \param is_train int value to indicate whether the backward pass is for evaluation
 *
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorBackwardEx(ExecutorHandle handle,
                                   mx_uint len,
                                   NDArrayHandle *head_grads,
                                   int is_train);
/*!
 * \brief Get executor's head NDArray
 *
 * \param handle executor handle
 * \param out_size output narray vector size
 * \param out out put narray handles
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorOutputs(ExecutorHandle handle,
                                mx_uint *out_size,
                                NDArrayHandle **out);

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
MXNET_DLL int MXExecutorBind(SymbolHandle symbol_handle,
                             int dev_type,
                             int dev_id,
                             mx_uint len,
                             NDArrayHandle *in_args,
                             NDArrayHandle *arg_grad_store,
                             mx_uint *grad_req_type,
                             mx_uint aux_states_len,
                             NDArrayHandle *aux_states,
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
MXNET_DLL int MXExecutorBindX(SymbolHandle symbol_handle,
                              int dev_type,
                              int dev_id,
                              mx_uint num_map_keys,
                              const char** map_keys,
                              const int* map_dev_types,
                              const int* map_dev_ids,
                              mx_uint len,
                              NDArrayHandle *in_args,
                              NDArrayHandle *arg_grad_store,
                              mx_uint *grad_req_type,
                              mx_uint aux_states_len,
                              NDArrayHandle *aux_states,
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
MXNET_DLL int MXExecutorBindEX(SymbolHandle symbol_handle,
                               int dev_type,
                               int dev_id,
                               mx_uint num_map_keys,
                               const char** map_keys,
                               const int* map_dev_types,
                               const int* map_dev_ids,
                               mx_uint len,
                               NDArrayHandle *in_args,
                               NDArrayHandle *arg_grad_store,
                               mx_uint *grad_req_type,
                               mx_uint aux_states_len,
                               NDArrayHandle *aux_states,
                               ExecutorHandle shared_exec,
                               ExecutorHandle *out);

MXNET_DLL int MXExecutorSimpleBind(SymbolHandle symbol_handle,
                                   int dev_type,
                                   int dev_id,
                                   const mx_uint num_g2c_keys,
                                   const char** g2c_keys,
                                   const int* g2c_dev_types,
                                   const int* g2c_dev_ids,
                                   const mx_uint provided_grad_req_list_len,
                                   const char** provided_grad_req_names,
                                   const char** provided_grad_req_types,
                                   const mx_uint num_provided_arg_shapes,
                                   const char** provided_arg_shape_names,
                                   const mx_uint* provided_arg_shape_data,
                                   const mx_uint* provided_arg_shape_idx,
                                   const mx_uint num_provided_arg_dtypes,
                                   const char** provided_arg_dtype_names,
                                   const int* provided_arg_dtypes,
                                   const mx_uint num_provided_arg_stypes,
                                   const char** provided_arg_stype_names,
                                   const int* provided_arg_stypes,
                                   const mx_uint num_shared_arg_names,
                                   const char** shared_arg_name_list,
                                   int* shared_buffer_len,
                                   const char** shared_buffer_name_list,
                                   NDArrayHandle* shared_buffer_handle_list,
                                   const char*** updated_shared_buffer_name_list,
                                   NDArrayHandle** updated_shared_buffer_handle_list,
                                   mx_uint* num_in_args,
                                   NDArrayHandle** in_args,
                                   NDArrayHandle** arg_grads,
                                   mx_uint* num_aux_states,
                                   NDArrayHandle** aux_states,
                                   ExecutorHandle shared_exec_handle,
                                   ExecutorHandle* out);

/*!
 * \brief Return a new executor with the same symbol and shared memory,
 * but different input/output shapes.
 *
 * \param partial_shaping Whether to allow changing the shape of unspecified arguments.
 * \param allow_up_sizing Whether to allow allocating new ndarrays that's larger than the original.
 * \param dev_type device type of default context
 * \param dev_id device id of default context
 * \param num_map_keys size of group2ctx map
 * \param map_keys keys of group2ctx map
 * \param map_dev_types device type of group2ctx map
 * \param map_dev_ids device id of group2ctx map
 * \param num_in_args length of in_args
 * \param in_args in args array
 * \param arg_grads arg grads handle array
 * \param num_aux_states length of auxiliary states
 * \param aux_states auxiliary states array
 * \param shared_exec input executor handle for memory sharing
 * \param out output executor handle
 * \return a new executor
 */
MXNET_DLL int MXExecutorReshape(int partial_shaping,
                                int allow_up_sizing,
                                int dev_type,
                                int dev_id,
                                mx_uint num_map_keys,
                                const char** map_keys,
                                const int* map_dev_types,
                                const int* map_dev_ids,
                                const mx_uint num_provided_arg_shapes,
                                const char** provided_arg_shape_names,
                                const mx_uint* provided_arg_shape_data,
                                const mx_uint* provided_arg_shape_idx,
                                mx_uint* num_in_args,
                                NDArrayHandle** in_args,
                                NDArrayHandle** arg_grads,
                                mx_uint* num_aux_states,
                                NDArrayHandle** aux_states,
                                ExecutorHandle shared_exec,
                                ExecutorHandle *out);

/*!
 * \brief get optimized graph from graph executor
 */
MXNET_DLL int MXExecutorGetOptimizedSymbol(ExecutorHandle handle,
                                           SymbolHandle *out);

/*!
 * \brief set a call back to notify the completion of operation
 */
MXNET_DLL int MXExecutorSetMonitorCallback(ExecutorHandle handle,
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
MXNET_DLL int MXListDataIters(mx_uint *out_size,
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
MXNET_DLL int MXDataIterCreateIter(DataIterCreator handle,
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
MXNET_DLL int MXDataIterGetIterInfo(DataIterCreator creator,
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
MXNET_DLL int MXDataIterFree(DataIterHandle handle);
/*!
 * \brief Move iterator to next position
 * \param handle the handle to iterator
 * \param out return value of next
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterNext(DataIterHandle handle,
                             int *out);
/*!
 * \brief Call iterator.Reset
 * \param handle the handle to iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterBeforeFirst(DataIterHandle handle);

/*!
 * \brief Get the handle to the NDArray of underlying data
 * \param handle the handle pointer to the data iterator
 * \param out handle to underlying data NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetData(DataIterHandle handle,
                                NDArrayHandle *out);
/*!
 * \brief Get the image index by array.
 * \param handle the handle pointer to the data iterator
 * \param out_index output index of the array.
 * \param out_size output size of the array.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetIndex(DataIterHandle handle,
                                 uint64_t **out_index,
                                 uint64_t *out_size);
/*!
 * \brief Get the padding number in current data batch
 * \param handle the handle pointer to the data iterator
 * \param pad pad number ptr
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetPadNum(DataIterHandle handle,
                                  int *pad);

/*!
 * \brief Get the handle to the NDArray of underlying label
 * \param handle the handle pointer to the data iterator
 * \param out the handle to underlying label NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetLabel(DataIterHandle handle,
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
MXNET_DLL int MXInitPSEnv(mx_uint num_vars,
                          const char **keys,
                          const char **vals);


/*!
 * \brief Create a kvstore
 * \param type the type of KVStore
 * \param out The output type of KVStore
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreCreate(const char *type,
                              KVStoreHandle *out);

/*!
 * \brief Set parameters to use low-bit compressed gradients
 * \param handle handle to the kvstore
 * \param keys keys for compression parameters
 * \param vals values for compression parameters
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSetGradientCompression(KVStoreHandle handle,
                                              mx_uint num_params,
                                              const char** keys,
                                              const char** vals);

/*!
 * \brief Delete a KVStore handle.
 * \param handle handle to the kvstore
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreFree(KVStoreHandle handle);
/*!
 * \brief Init a list of (key,value) pairs in kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreInit(KVStoreHandle handle,
                            mx_uint num,
                            const int* keys,
                            NDArrayHandle* vals);

/*!
 * \brief Init a list of (key,value) pairs in kvstore, where each key is a string
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreInitEx(KVStoreHandle handle,
                              mx_uint num,
                              const char** keys,
                              NDArrayHandle* vals);

/*!
 * \brief Push a list of (key,value) pairs to kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePush(KVStoreHandle handle,
                            mx_uint num,
                            const int* keys,
                            NDArrayHandle* vals,
                            int priority);
/*!
 * \brief Push a list of (key,value) pairs to kvstore, where each key is a string
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePushEx(KVStoreHandle handle,
                              mx_uint num,
                              const char** keys,
                              NDArrayHandle* vals,
                              int priority);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \param ignore_sparse whether to ignore sparse arrays in the request
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullWithSparse(KVStoreHandle handle,
                                      mx_uint num,
                                      const int* keys,
                                      NDArrayHandle* vals,
                                      int priority,
                                      bool ignore_sparse);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore, where each key is a string
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \param ignore_sparse whether to ignore sparse arrays in the request
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullWithSparseEx(KVStoreHandle handle,
                                        mx_uint num,
                                        const char** keys,
                                        NDArrayHandle* vals,
                                        int priority,
                                        bool ignore_sparse);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePull(KVStoreHandle handle,
                            mx_uint num,
                            const int* keys,
                            NDArrayHandle* vals,
                            int priority);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore, where each key is a string
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullEx(KVStoreHandle handle,
                              mx_uint num,
                              const char** keys,
                              NDArrayHandle* vals,
                              int priority);

/*!
 * \brief pull a list of (key, value) pairs from the kvstore, where each key is an integer.
 *        The NDArray pulled back will be in row_sparse storage with only the specified
 *        row_ids present based row_ids (others rows are zeros).
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param row_ids the list of row_id NDArrays
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullRowSparse(KVStoreHandle handle,
                                     mx_uint num,
                                     const int* keys,
                                     NDArrayHandle* vals,
                                     const NDArrayHandle* row_ids,
                                     int priority);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore, where each key is a string.
 *        The NDArray pulled back will be in row_sparse storage with only the specified
 *        row_ids present based row_ids (others rows are zeros).
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param row_ids the list of row_id NDArrays
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullRowSparseEx(KVStoreHandle handle,
                                       mx_uint num,
                                       const char** keys,
                                       NDArrayHandle* vals,
                                       const NDArrayHandle* row_ids,
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
 * \brief user-defined updater for the kvstore with string keys
 * It's this updater's responsibility to delete \a recv and \a local
 * \param the key
 * \param recv the pushed value on this key
 * \param local the value stored on local on this key
 * \param handle The additional handle to the updater
 */
typedef void (MXKVStoreStrUpdater)(const char* key,
                                   NDArrayHandle recv,
                                   NDArrayHandle local,
                                   void *handle);
/*!
 * \brief register a push updater
 * \param handle handle to the KVStore
 * \param updater udpater function
 * \param updater_handle The additional handle used to invoke the updater
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSetUpdater(KVStoreHandle handle,
                                  MXKVStoreUpdater updater,
                                  void *updater_handle);
/*!
 * \brief register a push updater with int keys and one with string keys
 * \param handle handle to the KVStore
 * \param updater updater function with int keys
 * \param str_updater updater function with string keys
 * \param updater_handle The additional handle used to invoke the updater
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSetUpdaterEx(KVStoreHandle handle,
                                    MXKVStoreUpdater updater,
                                    MXKVStoreStrUpdater str_updater,
                                    void *updater_handle);
/*!
 * \brief get the type of the kvstore
 * \param handle handle to the KVStore
 * \param type a string type
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreGetType(KVStoreHandle handle,
                               const char** type);
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
MXNET_DLL int MXKVStoreGetRank(KVStoreHandle handle,
                               int *ret);

/**
 * \brief return The number of nodes in this group, which is
 * - number of workers if if `IsWorkerNode() == true`,
 * - number of servers if if `IsServerNode() == true`,
 * - 1 if `IsSchedulerNode() == true`,
 * \param handle handle to the KVStore
 * \param ret the group size
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreGetGroupSize(KVStoreHandle handle,
                                    int *ret);

/**
 * \brief return whether or not this process is a worker node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreIsWorkerNode(int *ret);


/**
 * \brief return whether or not this process is a server node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreIsServerNode(int *ret);


/**
 * \brief return whether or not this process is a scheduler node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreIsSchedulerNode(int *ret);

/**
 * \brief global barrier among all worker machines
 *
 * \param handle handle to the KVStore
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreBarrier(KVStoreHandle handle);

/**
 * \brief whether to do barrier when finalize
 *
 * \param handle handle to the KVStore
 * \param barrier_before_exit whether to do barrier when kvstore finalize
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSetBarrierBeforeExit(KVStoreHandle handle,
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
 * \brief Run as server (or scheduler)
 * \param handle handle to the KVStore
 * \param controller the user-defined server controller
 * \param controller_handle helper handle for implementing controller
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreRunServer(KVStoreHandle handle,
                                 MXKVStoreServerController controller,
                                 void *controller_handle);

/**
 * \brief Send a command to all server nodes
 * \param handle handle to the KVStore
 * \param cmd_id the head of the command
 * \param cmd_body the body of the command
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSendCommmandToServers(KVStoreHandle handle,
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
MXNET_DLL int MXKVStoreGetNumDeadNode(KVStoreHandle handle,
                                      const int node_id,
                                      int *number,
                                      const int timeout_sec DEFAULT(60));

/**
 * \brief Create a RecordIO writer object
 * \param uri path to file
 * \param out handle pointer to the created object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterCreate(const char *uri, RecordIOHandle *out);

/**
 * \brief Delete a RecordIO writer object
 * \param handle handle to RecordIO object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterFree(RecordIOHandle handle);

/**
 * \brief Write a record to a RecordIO object
 * \param handle handle to RecordIO object
 * \param buf buffer to write
 * \param size size of buffer
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterWriteRecord(RecordIOHandle handle,
                                          const char *buf, size_t size);

/**
 * \brief Get the current writer pointer position
 * \param handle handle to RecordIO object
 * \param pos handle to output position
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterTell(RecordIOHandle handle, size_t *pos);

/**
 * \brief Create a RecordIO reader object
 * \param uri path to file
 * \param out handle pointer to the created object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderCreate(const char *uri, RecordIOHandle *out);

/**
 * \brief Delete a RecordIO reader object
 * \param handle handle to RecordIO object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderFree(RecordIOHandle handle);

/**
 * \brief Write a record to a RecordIO object
 * \param handle handle to RecordIO object
 * \param buf pointer to return buffer
 * \param size point to size of buffer
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderReadRecord(RecordIOHandle handle,
                                        char const **buf, size_t *size);

/**
 * \brief Set the current reader pointer position
 * \param handle handle to RecordIO object
 * \param pos target position
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderSeek(RecordIOHandle handle, size_t pos);

/**
 * \brief Get the current writer pointer position
 * \param handle handle to RecordIO object
 * \param pos handle to output position
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderTell(RecordIOHandle handle, size_t *pos);

/**
 * \brief Create a MXRtc object
*/
MXNET_DLL int MXRtcCreate(char* name, mx_uint num_input, mx_uint num_output,
                          char** input_names, char** output_names,
                          NDArrayHandle* inputs, NDArrayHandle* outputs,
                          char* kernel, RtcHandle *out);

/**
 * \brief Run cuda kernel
*/
MXNET_DLL int MXRtcPush(RtcHandle handle, mx_uint num_input, mx_uint num_output,
                        NDArrayHandle* inputs, NDArrayHandle* outputs,
                        mx_uint gridDimX,
                        mx_uint gridDimY,
                        mx_uint gridDimZ,
                        mx_uint blockDimX,
                        mx_uint blockDimY,
                        mx_uint blockDimZ);

/**
 * \brief Delete a MXRtc object
*/
MXNET_DLL int MXRtcFree(RtcHandle handle);
/*
 * \brief register custom operators from frontend.
 * \param op_type name of custom op
 * \param creator
 */
MXNET_DLL int MXCustomOpRegister(const char* op_type, CustomOpPropCreator creator);
/*
 * \brief record custom function for backward later.
 * \param num_inputs number of input NDArrays.
 * \param inputs handle to input NDArrays.
 * \param num_outputs number of output NDArrays.
 * \param outputs handle to output NDArrays.
 * \param callbacks callbacks for backward function.
 */
MXNET_DLL int MXCustomFunctionRecord(int num_inputs, NDArrayHandle *inputs,
                                     int num_outputs, NDArrayHandle *outputs,
                                     struct MXCallbackList *callbacks);
/*
 * \brief create cuda rtc module
 * \param source cuda source code
 * \param num_options number of compiler flags
 * \param options compiler flags
 * \param num_exports number of exported function names
 * \param exported function names
 * \param out handle to created module
 */
MXNET_DLL int MXRtcCudaModuleCreate(const char* source, int num_options,
                                    const char** options, int num_exports,
                                    const char** exports, CudaModuleHandle *out);
/*
 * \brief delete cuda rtc module
 * \param handle handle to cuda module
 */
MXNET_DLL int MXRtcCudaModuleFree(CudaModuleHandle handle);
/*
 * \brief get kernel from module
 * \param handle handle to cuda module
 * \param name name of kernel function
 * \param num_args number of arguments
 * \param is_ndarray whether argument is ndarray
 * \param is_const whether argument is constant
 * \param arg_types data type of arguments
 * \param out created kernel
 */
MXNET_DLL int MXRtcCudaKernelCreate(CudaModuleHandle handle, const char* name,
                                    int num_args, int* is_ndarray, int* is_const,
                                    int* arg_types, CudaKernelHandle *out);
/*
 * \brief delete kernel
 * \param handle handle to previously created kernel
 */
MXNET_DLL int MXRtcCudaKernelFree(CudaKernelHandle handle);
/*
 * \brief launch cuda kernel
 * \param handle handle to kernel
 * \param dev_id (GPU) device id
 * \param args pointer to arguments
 * \param grid_dim_x grid dimension x
 * \param grid_dim_y grid dimension y
 * \param grid_dim_z grid dimension z
 * \param block_dim_x block dimension x
 * \param block_dim_y block dimension y
 * \param block_dim_z block dimension z
 * \param shared_mem size of dynamically allocated shared memory
 */
MXNET_DLL int MXRtcCudaKernelCall(CudaKernelHandle handle, int dev_id, void** args,
                                  mx_uint grid_dim_x, mx_uint grid_dim_y,
                                  mx_uint grid_dim_z, mx_uint block_dim_x,
                                  mx_uint block_dim_y, mx_uint block_dim_z,
                                  mx_uint shared_mem);
/*!
 * \brief Get shared memory handle from NDArray
 * \param handle NDArray handle.
 * \param shared_pid output PID
 * \param shared_id output shared memory id.
 */
MXNET_DLL int MXNDArrayGetSharedMemHandle(NDArrayHandle handle, int* shared_pid,
                                          int* shared_id);
/*!
 * \brief Reconstruct NDArray from shared memory handle
 * \param shared_pid shared PID
 * \param shared_id shared memory id
 * \param shape pointer to NDArray dimensions
 * \param ndim number of NDArray dimensions
 * \param dtype data type of NDArray
 * \param out constructed NDArray
 */
MXNET_DLL int MXNDArrayCreateFromSharedMem(int shared_pid, int shared_id, const mx_uint *shape,
                                           mx_uint ndim, int dtype, NDArrayHandle *out);

/*!
  * \brief Push an asynchronous operation to the engine.
  * \param async_func Execution function whici takes a parameter on_complete
  *                   that must be called when the execution ompletes.
  * \param func_param The parameter set on calling async_func, can be NULL.
  * \param deleter The callback to free func_param, can be NULL.
  * \param ctx_handle Execution context.
  * \param const_vars_handle The variables that current operation will use
  *                          but not mutate.
  * \param num_const_vars The number of const_vars.
  * \param mutable_vars_handle The variables that current operation will mutate.
  * \param num_mutable_vars The number of mutable_vars.
  * \param prop_handle Property of the function.
  * \param priority Priority of the action, as hint to the engine.
  * \param opr_name The operation name.
  * \param wait Whether this is a WaitForVar operation.
  */
MXNET_DLL int MXEnginePushAsync(EngineAsyncFunc async_func, void* func_param,
                                EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                                EngineVarHandle const_vars_handle, int num_const_vars,
                                EngineVarHandle mutable_vars_handle, int num_mutable_vars,
                                EngineFnPropertyHandle prop_handle DEFAULT(NULL),
                                int priority DEFAULT(0), const char* opr_name DEFAULT(NULL),
                                bool wait DEFAULT(false));

/*!
  * \brief Push a synchronous operation to the engine.
  * \param sync_func Execution function that executes the operation.
  * \param func_param The parameter set on calling sync_func, can be NULL.
  * \param deleter The callback to free func_param, can be NULL.
  * \param ctx_handle Execution context.
  * \param const_vars_handle The variables that current operation will use
  *                          but not mutate.
  * \param num_const_vars The number of const_vars.
  * \param mutable_vars_handle The variables that current operation will mutate.
  * \param num_mutable_vars The number of mutable_vars.
  * \param prop_handle Property of the function.
  * \param priority Priority of the action, as hint to the engine.
  * \param opr_name The operation name.
  */
MXNET_DLL int MXEnginePushSync(EngineSyncFunc sync_func, void* func_param,
                               EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                               EngineVarHandle const_vars_handle, int num_const_vars,
                               EngineVarHandle mutable_vars_handle, int num_mutable_vars,
                               EngineFnPropertyHandle prop_handle DEFAULT(NULL),
                               int priority DEFAULT(0), const char* opr_name DEFAULT(NULL));

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // MXNET_C_API_H_
