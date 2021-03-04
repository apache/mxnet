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
#include "gpu/ocl/ocl_engine.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::gpu::ocl;

status_t dnnl_ocl_interop_engine_create(
        engine_t **engine, cl_device_id device, cl_context context) {
    bool args_ok = !utils::any_null(engine, device, context);
    if (!args_ok) return status::invalid_arguments;

    ocl_engine_factory_t f(engine_kind::gpu);
    return f.engine_create(engine, device, context);
}

status_t dnnl_ocl_interop_engine_get_context(
        engine_t *engine, cl_context *context) {
    bool args_ok = !utils::any_null(engine, context)
            && (engine->runtime_kind() == runtime_kind::ocl);

    if (!args_ok) return status::invalid_arguments;

    auto *ocl_engine = utils::downcast<ocl_gpu_engine_t *>(engine);
    *context = ocl_engine->context();
    return status::success;
}

status_t dnnl_ocl_interop_get_device(engine_t *engine, cl_device_id *device) {
    bool args_ok = !utils::any_null(engine, device)
            && (engine->runtime_kind() == runtime_kind::ocl);

    if (!args_ok) return status::invalid_arguments;

    auto *ocl_engine = utils::downcast<ocl_gpu_engine_t *>(engine);
    *device = ocl_engine->device();
    return status::success;
}
