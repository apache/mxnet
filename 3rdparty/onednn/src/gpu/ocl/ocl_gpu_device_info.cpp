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

#include "gpu/ocl/ocl_gpu_device_info.hpp"
#include "gpu/ocl/ocl_gpu_detect.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ocl_gpu_device_info_t::init_arch() {
    cl_int err = CL_SUCCESS;

    // skip other vendors
    const cl_uint intel_vendor_id = 0x8086;
    cl_uint vendor_id;
    err = clGetDeviceInfo(
            device_, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, nullptr);
    OCL_CHECK(err);
    if (vendor_id != intel_vendor_id) return status::success;

    // try to detect gpu by device name first
    gpu_arch_ = detect_gpu_arch_by_device_name(name());
    if (gpu_arch_ != compute::gpu_arch_t::unknown) return status::success;

    // if failed, use slower method
    cl_context context
            = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    OCL_CHECK(err);

    gpu_arch_ = detect_gpu_arch(device_, context);
    err = clReleaseContext(context);
    OCL_CHECK(err);

    return status::success;
}

status_t ocl_gpu_device_info_t::init_device_name() {
    cl_int err = CL_SUCCESS;

    size_t param_size = 0;
    err = clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, nullptr, &param_size);
    OCL_CHECK(err);

    std::string device_name(param_size, '\0');
    err = clGetDeviceInfo(
            device_, CL_DEVICE_NAME, param_size, &device_name[0], &param_size);
    OCL_CHECK(err);

    set_name(device_name);
    return status::success;
}

status_t ocl_gpu_device_info_t::init_runtime_version() {
    cl_int err = CL_SUCCESS;

    size_t param_size = 0;
    err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, 0, nullptr, &param_size);
    OCL_CHECK(err);

    std::string driver_version(param_size, '\0');
    err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, param_size,
            &driver_version[0], nullptr);
    OCL_CHECK(err);

    compute::runtime_version_t runtime_version;
    if (runtime_version.set_from_string(&driver_version[0])
            != status::success) {
        runtime_version.major = 0;
        runtime_version.minor = 0;
        runtime_version.build = 0;
    }

    set_runtime_version(runtime_version);
    return status::success;
}

static uint64_t get_future_extensions(compute::gpu_arch_t gpu_arch) {
    using namespace compute;

    uint64_t extensions = 0;
    switch (gpu_arch) {
        case gpu_arch_t::gen12lp:
            extensions |= (uint64_t)device_ext_t::intel_dot_accumulate;
            break;
        default: break;
    }
    return extensions;
}

status_t ocl_gpu_device_info_t::init_extensions() {
    cl_int err = CL_SUCCESS;

    // query device for extensions
    size_t param_size = 0;
    err = clGetDeviceInfo(
            device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &param_size);
    OCL_CHECK(err);

    std::string extension_string(param_size, '\0');
    err = clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, param_size,
            &extension_string[0], &param_size);
    OCL_CHECK(err);

    // convert to ours
    using namespace compute;
    for (uint64_t i_ext = 1; i_ext < (uint64_t)device_ext_t::last;
            i_ext <<= 1) {
        const char *s_ext = ext2cl_str((device_ext_t)i_ext);
        if (s_ext && extension_string.find(s_ext) != std::string::npos) {
            extensions_ |= i_ext;
        }
    }

    // Handle future extensions, not yet supported by the OpenCL API
    extensions_ |= (uint64_t)get_future_extensions(gpu_arch());

    return status::success;
}

status_t ocl_gpu_device_info_t::init_attributes() {
    cl_int err = CL_SUCCESS;

    cl_uint eu_count = 0;
    err = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
            &eu_count, nullptr);
    OCL_CHECK(err);
    eu_count_ = (int32_t)eu_count;

    // Assume 7 threads by default
    int32_t threads_per_eu = 7;

    switch (gpu_arch_) {
        case compute::gpu_arch_t::gen9:
        case compute::gpu_arch_t::gen12lp: threads_per_eu = 7; break;
        default: break;
    }

    hw_threads_ = eu_count_ * threads_per_eu;

    // TODO: Fix for discrete GPUs. The code below is written for
    // integrated GPUs assuming that last-level cache for GPU is shared
    // with CPU.
    llc_cache_size_ = get_llc_cache_size();

    return status::success;
}

size_t ocl_gpu_device_info_t::get_llc_cache_size() const {
    // Integrated GPUs share LLC with CPU which is L3 cache on CPU.
    size_t cache_size = cpu::platform::get_per_core_cache_size(3)
            * cpu::platform::get_num_cores();
    return cache_size;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
