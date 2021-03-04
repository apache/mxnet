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

#ifndef GPU_OCL_OCL_ENGINE_HPP
#define GPU_OCL_OCL_ENGINE_HPP

#include "gpu/ocl/ocl_gpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_engine_factory_t : public engine_factory_t {
public:
    ocl_engine_factory_t(engine_kind_t engine_kind) {
        assert(engine_kind == engine_kind::gpu);
        MAYBE_UNUSED(engine_kind);
    }

    size_t count() const override {
        std::vector<cl_device_id> ocl_devices;
        status_t status = get_ocl_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);
        if (status != status::success) return status;
        return ocl_devices.size();
    }

    status_t engine_create(engine_t **engine, size_t index) const override {
        status_t status;
        std::vector<cl_device_id> ocl_devices;

        status = get_ocl_devices(&ocl_devices, CL_DEVICE_TYPE_GPU);
        if (status != status::success) return status;

        if (index >= ocl_devices.size()) return status::invalid_arguments;

        auto *ocl_engine = new ocl_gpu_engine_t(ocl_devices[index]);
        if (!ocl_engine) return status::out_of_memory;

        status = ocl_engine->init();
        if (status != status::success) {
            delete ocl_engine;
            return status;
        }
        *engine = ocl_engine;
        return status::success;
    }

    status_t engine_create(
            engine_t **engine, cl_device_id device, cl_context context) {
        auto *ocl_engine = new ocl_gpu_engine_t(device, context);
        if (!ocl_engine) return status::out_of_memory;

        status_t status = ocl_engine->init();
        if (status != status::success) {
            delete ocl_engine;
            return status;
        }
        *engine = ocl_engine;
        return status::success;
    }
};
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_OCL_ENGINE_HPP
