/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <CL/cl.h>

#include "oneapi/dnnl/dnnl_ocl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_stream.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::gpu::ocl;

status_t dnnl_ocl_interop_stream_create(
        stream_t **stream, engine_t *engine, cl_command_queue queue) {
    bool args_ok = !utils::any_null(stream, engine, queue)
            && engine->runtime_kind() == runtime_kind::ocl;

    if (!args_ok) return status::invalid_arguments;

    auto *ocl_engine = utils::downcast<ocl_gpu_engine_t *>(engine);
    return ocl_engine->create_stream(stream, queue);
}

status_t dnnl_ocl_interop_stream_get_command_queue(
        stream_t *stream, cl_command_queue *queue) {
    bool args_ok = !utils::any_null(queue, stream)
            && stream->engine()->runtime_kind() == runtime_kind::ocl;

    if (!args_ok) return status::invalid_arguments;

    auto *ocl_stream = utils::downcast<ocl_stream_t *>(stream);
    *queue = ocl_stream->queue();
    return status::success;
}
