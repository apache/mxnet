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

#ifndef GPU_GEMM_GPU_GEMM_UTILS_HPP
#define GPU_GEMM_GPU_GEMM_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/primitive_attr.hpp"
#include "gpu/gemm/gpu_gemm.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace gemm_utils {

inline status_t prepare_scales(const primitive_attr_t *attr, engine_t *engine,
        std::unique_ptr<memory_storage_t> &mem_storage) {
    mem_storage.reset();
    status_t s = status::success;
    const bool is_defined = attr->output_scales_.defined();

    if (!is_defined) return status::success;

    const dim_t count = attr->output_scales_.count_;
    const float *s_data = attr->output_scales_.scales_;

    const size_t size = count * sizeof(float);
    memory_storage_t *mem_storage_ptr = nullptr;
    s = engine->create_memory_storage(&mem_storage_ptr, size);
    if (s != status::success) return s;
    mem_storage.reset(mem_storage_ptr);

    float *mapped_mem_storage = nullptr;
    s = mem_storage->map_data(
            (void **)&mapped_mem_storage, nullptr, sizeof(*s_data) * count);
    if (s != status::success) return s;
    utils::array_copy(mapped_mem_storage, s_data, count);
    s = mem_storage->unmap_data((void *)mapped_mem_storage, nullptr);
    if (s != status::success) return s;

    return s;
}

inline status_t prepare_zero_points(const primitive_attr_t *attr,
        engine_t *engine, int arg,
        std::unique_ptr<memory_storage_t> &mem_storage) {
    mem_storage.reset();
    status_t s = status::success;
    const bool is_defined = attr->zero_points_.defined(arg);

    if (!is_defined) return status::success;

    dim_t count = 0;
    const int *zp_data = nullptr;
    s = attr->zero_points_.get(arg, &count, nullptr, &zp_data);
    if (s != status::success) return s;

    const size_t size = count * sizeof(int);
    memory_storage_t *mem_storage_ptr = nullptr;
    s = engine->create_memory_storage(&mem_storage_ptr, size);
    if (s != status::success) return s;
    mem_storage.reset(mem_storage_ptr);

    int *mapped_mem_storage = nullptr;
    s = mem_storage->map_data(
            (void **)&mapped_mem_storage, nullptr, sizeof(*zp_data) * count);
    if (s != status::success) return s;
    utils::array_copy(mapped_mem_storage, zp_data, count);
    s = mem_storage->unmap_data((void *)mapped_mem_storage, nullptr);
    if (s != status::success) return s;

    return s;
}

inline const gpu_gemm_t *gpu_gemm(const std::shared_ptr<primitive_t> &p) {
    return utils::downcast<gpu_gemm_t *>(p.get());
}

} // namespace gemm_utils
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
