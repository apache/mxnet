/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef ONEAPI_DNNL_DNNL_OCL_H
#define ONEAPI_DNNL_DNNL_OCL_H

#include "oneapi/dnnl/dnnl.h"

/// @cond DO_NOT_DOCUMENT_THIS
// Set target version for OpenCL explicitly to suppress a compiler warning.
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include <CL/cl.h>
/// @endcond

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_ocl_interop
/// @{

/// Returns an OpenCL memory object associated with a memory object.
///
/// @param memory Memory object.
/// @param mem_object Output OpenCL memory object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_memory_get_mem_object(
        const_dnnl_memory_t memory, cl_mem *mem_object);

/// Sets OpenCL memory object associated with a memory object.
///
/// For behavioral details, see dnnl_memory_set_data_handle().
///
/// @param memory Memory object.
/// @param mem_object OpenCL memory object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_memory_set_mem_object(
        dnnl_memory_t memory, cl_mem mem_object);

/// Creates an engine associated with an OpenCL device and an OpenCL context.
///
/// @param engine Output engine.
/// @param device Underlying OpenCL device to use for the engine.
/// @param context Underlying OpenCL context to use for the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_engine_create(
        dnnl_engine_t *engine, cl_device_id device, cl_context context);

/// Returns the OpenCL context associated with an engine.
///
/// @param engine Engine to query.
/// @param context Output underlying OpenCL context of the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_engine_get_context(
        dnnl_engine_t engine, cl_context *context);

/// Returns the OpenCL device associated with an engine.
///
/// @param engine Engine to query.
/// @param device Output underlying OpenCL device of the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_get_device(
        dnnl_engine_t engine, cl_device_id *device);

/// Creates an execution stream for a given engine associated with
/// an OpenCL command queue.
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param queue OpenCL command queue to use.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_stream_create(
        dnnl_stream_t *stream, dnnl_engine_t engine, cl_command_queue queue);

/// Returns the OpenCL command queue associated with an execution stream.
///
/// @param stream Execution stream to query.
/// @param queue Output OpenCL command queue.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_stream_get_command_queue(
        dnnl_stream_t stream, cl_command_queue *queue);

/// @} dnnl_api_ocl_interop

/// @} dnnl_api_interop

/// @} dnnl_api

#ifdef __cplusplus
}
#endif

#endif
