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

/*
 * \file c_runtime_api.h
 * \brief MXNet runtime library.
 */
// Acknowledgement: This file originates from incubator-tvm
#ifndef MXNET_RUNTIME_C_RUNTIME_API_H_
#define MXNET_RUNTIME_C_RUNTIME_API_H_

#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <mxnet/c_api.h>
#include <stdint.h>
#include <stddef.h>


/*!
 * \brief The type code in MXNetType
 * \note MXNetType is used in two places.
 */
typedef enum {
  // The type code of other types are compatible with DLPack.
  // The next few fields are extension types
  // that is used by MXNet API calls.
  kHandle = 3U,
  kNull = 4U,
  kMXNetType = 5U,
  kMXNetContext = 6U,
  kArrayHandle = 7U,
  kObjectHandle = 8U,
  kModuleHandle = 9U,
  kFuncHandle = 10U,
  kStr = 11U,
  kBytes = 12U,
  kNDArrayContainer = 13U,
  kNDArrayHandle = 14U,
  // Extension codes for other frameworks to integrate MXNet PackedFunc.
  // To make sure each framework's id do not conflict, use first and
  // last sections to mark ranges.
  // Open an issue at the repo if you need a section of code.
  kExtBegin = 15U,
  kNNVMFirst = 16U,
  kNNVMLast = 20U,
  // The following section of code is used for non-reserved types.
  kExtReserveEnd = 64U,
  kExtEnd = 128U,
  // The rest of the space is used for custom, user-supplied datatypes
  kCustomBegin = 129U,
} MXNetTypeCode;

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  DLDataType v_type;
} MXNetValue;

/*!
 * \brief Byte array type used to pass in byte array
 *  When kBytes is used as data type.
 */
typedef struct {
  const char* data;
  size_t size;
} MXNetByteArray;

/*! \brief Handle to packed function handle. */
typedef void* MXNetFunctionHandle;
/*! \brief Handle to Object. */
typedef void* MXNetObjectHandle;

/*!
 * \brief Free the function when it is no longer needed.
 * \param func The function handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNetFuncFree(MXNetFunctionHandle func);

/*!
 * \brief Call a Packed MXNet Function.
 *
 * \param func node handle of the function.
 * \param arg_values The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 *
 * \param ret_val The return value.
 * \param ret_type_code the type code of return value.
 *
 * \return 0 when success, -1 when failure happens
 * \note MXNet calls always exchanges with type bits=64, lanes=1
 *
 * \note API calls always exchanges with type bits=64, lanes=1
 *   If API call returns container handles (e.g. FunctionHandle)
 *   these handles should be managed by the front-end.
 *   The front-end need to call free function (e.g. MXNetFuncFree)
 *   to free these handles.
 */
MXNET_DLL int MXNetFuncCall(MXNetFunctionHandle func,
                            MXNetValue* arg_values,
                            int* type_codes,
                            int num_args,
                            MXNetValue* ret_val,
                            int* ret_type_code);

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer, NULL if it does not exist.
 *
 * \note The function handle of global function is managed by MXNet runtime,
 *  So MXNetFuncFree is should not be called when it get deleted.
 */
MXNET_DLL int MXNetFuncGetGlobal(const char* name, MXNetFunctionHandle* out);

/*!
 * \brief List all the globally registered function name
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNetFuncListGlobalNames(int* out_size,
                                       const char*** out_array);

/*!
 * \brief Free the object.
 *
 * \param obj The object handle.
 * \note Internally we decrease the reference counter of the object.
 *       The object will be freed when every reference to the object are removed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNetObjectFree(MXNetObjectHandle obj);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // MXNET_RUNTIME_C_RUNTIME_API_H_
