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

#include <cstring>
#include <CL/cl_ext.h>

#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t get_ocl_devices(
        std::vector<cl_device_id> *devices, cl_device_type device_type) {
    cl_uint num_platforms = 0;

    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    // No platforms - a valid scenario
    if (err == CL_PLATFORM_NOT_FOUND_KHR) return status::success;

    OCL_CHECK(err);

    std::vector<cl_platform_id> platforms(num_platforms);
    OCL_CHECK(clGetPlatformIDs(num_platforms, &platforms[0], nullptr));

    for (size_t i = 0; i < platforms.size(); ++i) {
        cl_uint num_devices = 0;
        cl_int err = clGetDeviceIDs(
                platforms[i], device_type, 0, nullptr, &num_devices);

        if (!utils::one_of(err, CL_SUCCESS, CL_DEVICE_NOT_FOUND)) {
            return status::runtime_error;
        }

        if (num_devices != 0) {
            std::vector<cl_device_id> plat_devices;
            plat_devices.resize(num_devices);
            OCL_CHECK(clGetDeviceIDs(platforms[i], device_type, num_devices,
                    &plat_devices[0], nullptr));

            // Use Intel devices only
            for (size_t j = 0; j < plat_devices.size(); ++j) {
                cl_uint vendor_id;
                clGetDeviceInfo(plat_devices[j], CL_DEVICE_VENDOR_ID,
                        sizeof(cl_uint), &vendor_id, nullptr);
                if (vendor_id == 0x8086) {
                    devices->push_back(plat_devices[j]);
                }
            }
        }
    }
    // No devices found but still return success
    return status::success;
}

status_t get_ocl_kernel_arg_type(
        compute::scalar_type_t *type, cl_kernel ocl_kernel, int idx) {
    char s_type[16];
    OCL_CHECK(clGetKernelArgInfo(ocl_kernel, idx, CL_KERNEL_ARG_TYPE_NAME,
            sizeof(s_type), s_type, nullptr));
#define CASE(x) \
    if (!strcmp(STRINGIFY(x), s_type)) { \
        *type = compute::scalar_type_t::_##x; \
        return status::success; \
    }
    CASE(char)
    CASE(float)
    CASE(half)
    CASE(int)
    CASE(long)
    CASE(short)
    CASE(uchar)
    CASE(uint)
    CASE(ulong)
    CASE(ushort)
    CASE(zero_pad_mask_t)
#undef CASE

    assert(!"Not expected");
    return status::runtime_error;
}

cl_mem clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret) {
    return clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
