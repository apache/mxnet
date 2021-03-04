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

#ifndef SYCL_DEVICE_INFO_HPP
#define SYCL_DEVICE_INFO_HPP

#include <vector>
#include <CL/sycl.hpp>

#include "gpu/compute/device_info.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_device_info_t : public gpu::compute::device_info_t {
public:
    sycl_device_info_t(const cl::sycl::device &device) : device_(device) {}

    bool has(gpu::compute::device_ext_t ext) const override {
        return this->extensions_ & (uint64_t)ext;
    }

    gpu::compute::gpu_arch_t gpu_arch() const override { return gpu_arch_; }

    int eu_count() const override { return eu_count_; }
    int hw_threads() const override { return hw_threads_; }
    size_t llc_cache_size() const override { return llc_cache_size_; }

private:
    status_t init_arch() override;
    status_t init_device_name() override;
    status_t init_runtime_version() override;
    status_t init_extensions() override;
    status_t init_attributes() override;

    cl::sycl::device device_;

    int32_t hw_threads_ = 0;
    int32_t eu_count_ = 0;
    size_t llc_cache_size_ = 0;

    uint64_t extensions_ = 0;
    gpu::compute::gpu_arch_t gpu_arch_ = gpu::compute::gpu_arch_t::unknown;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_DEVICE_INFO_HPP
