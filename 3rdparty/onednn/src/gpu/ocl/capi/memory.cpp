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
#include "common/memory.hpp"
#include "common/utils.hpp"

using namespace dnnl::impl;

status_t dnnl_ocl_interop_memory_get_mem_object(
        const memory_t *memory, cl_mem *mem_object) {
    if (utils::any_null(mem_object)) return status::invalid_arguments;

    if (!memory) {
        *mem_object = nullptr;
        return status::success;
    }
    bool args_ok = (memory->engine()->runtime_kind() == runtime_kind::ocl);
    if (!args_ok) return status::invalid_arguments;

    void *handle;
    status_t status = memory->get_data_handle(&handle);
    if (status == status::success) *mem_object = static_cast<cl_mem>(handle);

    return status;
}

status_t dnnl_ocl_interop_memory_set_mem_object(
        memory_t *memory, cl_mem mem_object) {
    bool args_ok = (memory->engine()->runtime_kind() == runtime_kind::ocl);
    if (!args_ok) return status::invalid_arguments;

    return memory->set_data_handle(static_cast<void *>(mem_object), nullptr);
}
