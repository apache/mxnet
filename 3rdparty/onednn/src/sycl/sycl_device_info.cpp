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

#include "sycl/sycl_device_info.hpp"
#include "sycl/sycl_utils.hpp"

#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_gpu_detect.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

status_t sycl_device_info_t::init_arch() {
    // skip cpu engines
    if (!device_.is_gpu()) return status::success;

    // skip other vendors
    const int intel_vendor_id = 0x8086;
    auto vendor_id = device_.get_info<cl::sycl::info::device::vendor_id>();
    if (vendor_id != intel_vendor_id) return status::success;

    // try to detect gpu by device name first
    gpu_arch_ = gpu::ocl::detect_gpu_arch_by_device_name(name());
    if (gpu_arch_ != gpu::compute::gpu_arch_t::unknown) return status::success;

    // if failed, use slower method
    backend_t be = get_sycl_backend(device_);
    if (be == backend_t::opencl) {
        cl_int err = CL_SUCCESS;

        cl_device_id device = device_.get();
        cl_context context
                = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        OCL_CHECK(err);

        gpu_arch_ = gpu::ocl::detect_gpu_arch(device, context);
        err = clReleaseContext(context);
        OCL_CHECK(err);
    } else if (be == backend_t::level0) {
        // TODO: add support for L0 binary ngen check
        // XXX: query from ocl_engine for now
        gpu::ocl::ocl_engine_factory_t f(engine_kind::gpu);

        engine_t *engine;
        CHECK(f.engine_create(&engine, 0));

        std::unique_ptr<gpu::compute::compute_engine_t> compute_engine(
                utils::downcast<gpu::compute::compute_engine_t *>(engine));

        auto *dev_info = compute_engine->device_info();
        gpu_arch_ = dev_info->gpu_arch();
    } else {
        assert(!"not_expected");
    }

    return status::success;
}

status_t sycl_device_info_t::init_device_name() {
    auto device_name = device_.get_info<cl::sycl::info::device::name>();
    set_name(device_name);
    return status::success;
}

status_t sycl_device_info_t::init_runtime_version() {
    auto driver_version
            = device_.get_info<cl::sycl::info::device::driver_version>();

    gpu::compute::runtime_version_t runtime_version;
    if (runtime_version.set_from_string(driver_version.c_str())
            != status::success) {
        runtime_version.major = 0;
        runtime_version.minor = 0;
        runtime_version.build = 0;
    }

    set_runtime_version(runtime_version);
    return status::success;
}

status_t sycl_device_info_t::init_extensions() {
    using namespace gpu::compute;

    std::string extension_string;
    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        const char *s_ext = ext2cl_str((device_ext_t)i_ext);
        if (s_ext && device_.has_extension(s_ext)) {
            extension_string += std::string(s_ext) + " ";
            extensions_ |= i_ext;
        }
    }

    return status::success;
}

status_t sycl_device_info_t::init_attributes() {
    eu_count_ = device_.get_info<cl::sycl::info::device::max_compute_units>();

    // Assume 7 threads by default
    int32_t threads_per_eu = 7;
    switch (gpu_arch_) {
        case gpu::compute::gpu_arch_t::gen9: threads_per_eu = 7; break;
        case gpu::compute::gpu_arch_t::gen12lp: threads_per_eu = 7; break;
        default: break;
    }

    hw_threads_ = eu_count_ * threads_per_eu;

    // TODO: Fix for discrete GPUs. The code below is written for
    // integrated GPUs assuming that last-level cache for GPU is shared
    // with CPU.
    llc_cache_size_ = cpu::platform::get_per_core_cache_size(3)
            * cpu::platform::get_num_cores();

    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
