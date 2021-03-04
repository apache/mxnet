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

#ifndef GPU_OCL_OCL_GPU_DEVICE_INFO_HPP
#define GPU_OCL_OCL_GPU_DEVICE_INFO_HPP

#include <string>
#include <vector>
#include <CL/cl.h>

#include "gpu/compute/device_info.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_gpu_device_info_t : public compute::device_info_t {
public:
    ocl_gpu_device_info_t(cl_device_id device) : device_(device) {}

    bool has(compute::device_ext_t ext) const override {
        return this->extensions_ & (uint64_t)ext;
    }

    compute::gpu_arch_t gpu_arch() const override { return gpu_arch_; }

    int eu_count() const override { return eu_count_; }
    int hw_threads() const override { return hw_threads_; }
    size_t llc_cache_size() const override { return llc_cache_size_; }

private:
    status_t init_arch() override;
    status_t init_device_name() override;
    status_t init_runtime_version() override;
    status_t init_extensions() override;
    status_t init_attributes() override;

    size_t get_llc_cache_size() const;

    cl_device_id device_ = nullptr;

    int32_t hw_threads_ = 0;
    int32_t eu_count_ = 0;
    size_t llc_cache_size_ = 0;

    // extensions_ and gpu_arch_ describe effective extensions and GPU architecture.
    uint64_t extensions_ = 0;
    compute::gpu_arch_t gpu_arch_ = compute::gpu_arch_t::unknown;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_OCL_GPU_DEVICE_INFO_HPP
