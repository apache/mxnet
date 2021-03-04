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

#ifndef ONEAPI_DNNL_DNNL_OCL_HPP
#define ONEAPI_DNNL_DNNL_OCL_HPP

#include "oneapi/dnnl/dnnl.hpp"

/// @cond DO_NOT_DOCUMENT_THIS
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_ocl.h"

#include <CL/cl.h>
/// @endcond

/// @addtogroup dnnl_api
/// @{

namespace dnnl {

/// @addtogroup dnnl_api_interop Runtime interoperability API
/// API extensions to interact with the underlying run-time.
/// @{

/// @addtogroup dnnl_api_ocl_interop OpenCL interoperability API
/// API extensions to interact with the underlying OpenCL run-time.
///
/// @sa @ref dev_guide_opencl_interoperability in developer guide
/// @{

/// OpenCL interoperability namespace
namespace ocl_interop {

/// Constructs an engine from OpenCL device and context objects.
///
/// @param device The OpenCL device that this engine will encapsulate.
/// @param context The OpenCL context (containing the device) that this
///     engine will use for all operations.
/// @returns An engine.
inline engine make_engine(cl_device_id device, cl_context context) {
    dnnl_engine_t c_engine;
    error::wrap_c_api(
            dnnl_ocl_interop_engine_create(&c_engine, device, context),
            "could not create an engine");
    return engine(c_engine);
}

/// Returns OpenCL context associated with the engine.
///
/// @param aengine An engine.
/// @returns Underlying OpenCL context.
inline cl_context get_context(const engine &aengine) {
    cl_context context = nullptr;
    error::wrap_c_api(
            dnnl_ocl_interop_engine_get_context(aengine.get(), &context),
            "could not get an OpenCL context from an engine");
    return context;
}

/// Returns OpenCL device associated with the engine.
///
/// @param aengine An engine.
/// @returns Underlying OpenCL device.
inline cl_device_id get_device(const engine &aengine) {
    cl_device_id device = nullptr;
    error::wrap_c_api(dnnl_ocl_interop_get_device(aengine.get(), &device),
            "could not get an OpenCL device from an engine");
    return device;
}

/// Constructs an execution stream for the specified engine and OpenCL queue.
///
/// @param aengine Engine to create the stream on.
/// @param queue OpenCL queue to use for the stream.
/// @returns An execution stream.
inline stream make_stream(const engine &aengine, cl_command_queue queue) {
    dnnl_stream_t c_stream;
    error::wrap_c_api(
            dnnl_ocl_interop_stream_create(&c_stream, aengine.get(), queue),
            "could not create a stream");
    return stream(c_stream);
}

/// Returns OpenCL queue object associated with the execution stream.
///
/// @param astream An execution stream.
/// @returns Underlying OpenCL queue.
inline cl_command_queue get_command_queue(const stream &astream) {
    cl_command_queue queue = nullptr;
    error::wrap_c_api(
            dnnl_ocl_interop_stream_get_command_queue(astream.get(), &queue),
            "could not get an OpenCL command queue from a stream");
    return queue;
}

/// Returns the OpenCL memory object associated with the memory object.
///
/// @param amemory A memory object.
/// @returns Underlying OpenCL memory object.
inline cl_mem get_mem_object(const memory &amemory) {
    cl_mem mem_object;
    error::wrap_c_api(
            dnnl_ocl_interop_memory_get_mem_object(amemory.get(), &mem_object),
            "could not get OpenCL buffer object from a memory object");
    return mem_object;
}

/// Sets the OpenCL memory object associated with the memory object.
///
/// For behavioral details see memory::set_data_handle().
///
/// @param amemory A memory object.
/// @param mem_object OpenCL cl_mem object to use as the underlying
///     storage. It must have at least get_desc().get_size() bytes
///     allocated.
inline void set_mem_object(memory &amemory, cl_mem mem_object) {
    error::wrap_c_api(
            dnnl_ocl_interop_memory_set_mem_object(amemory.get(), mem_object),
            "could not set OpenCL buffer object from a memory object");
}

} // namespace ocl_interop

/// @} dnnl_api_ocl_interop

/// @} dnnl_api_interop

} // namespace dnnl

/// @} dnnl_api

#endif
