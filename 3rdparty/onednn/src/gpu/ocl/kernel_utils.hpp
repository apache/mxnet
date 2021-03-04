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

#ifndef GPU_OCL_KERNEL_UTILS_HPP
#define GPU_OCL_KERNEL_UTILS_HPP

#include <vector>
#include <unordered_map>

#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

const char **get_kernel_source(const char *name);

template <typename GetKernelSourceFunc>
status_t create_kernels(const compute::compute_engine_t *engine,
        compute::kernel_list_t &kernel_list,
        const compute::kernel_ctx_t &kernel_ctx,
        const GetKernelSourceFunc &get_kernel_source_func) {
    auto *ocl_engine = utils::downcast<const ocl::ocl_gpu_engine_t *>(engine);

    // Group kernels by their source.
    std::unordered_map<const char **, std::vector<const char *>>
            source_to_names;
    for (auto &kv : kernel_list.kernels()) {
        auto &name = kv.first;
        const char **source = get_kernel_source_func(name.c_str());
        source_to_names[source].push_back(name.c_str());
    }

    // Iterate through sources, create all kernels for the current source.
    for (auto &kv : source_to_names) {
        std::vector<compute::kernel_t> kernels;
        CHECK(ocl_engine->create_kernels_from_ocl_source(
                &kernels, kv.second, kv.first, kernel_ctx));

        // Update kernel list with created kernels.
        for (size_t i = 0; i < kv.second.size(); ++i) {
            kernel_list.set(kv.second[i], kernels[i]);
        }
    }
    return status::success;
}

inline status_t create_kernels(const compute::compute_engine_t *engine,
        compute::kernel_list_t &kernel_list,
        const compute::kernel_ctx_t &kernel_ctx) {
    return create_kernels(
            engine, kernel_list, kernel_ctx, ocl::get_kernel_source);
}

inline compute::kernel_t create_kernel(const compute::compute_engine_t *engine,
        const std::string &name, const compute::kernel_ctx_t &kernel_ctx) {
    compute::kernel_t kernel;
    compute::kernel_list_t kernel_list;
    kernel_list.add(name.c_str(), &kernel);
    create_kernels(engine, kernel_list, kernel_ctx);
    return kernel;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
