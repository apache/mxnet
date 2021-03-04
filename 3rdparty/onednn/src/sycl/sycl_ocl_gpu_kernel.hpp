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

#ifndef SYCL_OCL_GPU_KERNEL_HPP
#define SYCL_OCL_GPU_KERNEL_HPP

#include <CL/cl.h>

#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_gpu_kernel.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_ocl_gpu_kernel_t : public gpu::ocl::ocl_gpu_kernel_t {
public:
    using ocl_gpu_kernel_t::ocl_gpu_kernel_t;

    ~sycl_ocl_gpu_kernel_t() override = default;

    status_t parallel_for(stream_t &stream,
            const gpu::compute::nd_range_t &range,
            const gpu::compute::kernel_arg_list_t &arg_list) const override;

    status_t realize(
            gpu::compute::kernel_t *kernel, engine_t *engine) const override;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_OCL_GPU_KERNEL_HPP
