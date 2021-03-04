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

#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

backend_t get_sycl_gpu_backend() {
    // Create default GPU device and query its backend (assumed as default)
    static backend_t default_backend = []() {
        const backend_t fallback = backend_t::opencl;

        const auto gpu_type = cl::sycl::info::device_type::gpu;
        if (cl::sycl::device::get_devices(gpu_type).empty()) return fallback;

        cl::sycl::device dev {cl::sycl::gpu_selector {}};
        backend_t backend = get_sycl_backend(dev);

#if !defined(DNNL_WITH_LEVEL_ZERO)
        if (backend == backend_t::level0) backend = fallback;
#endif

        return backend;
    }();

    return default_backend;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
