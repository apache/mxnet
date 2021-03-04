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

#ifndef DNNL_TEST_COMMON_OCL_HPP
#define DNNL_TEST_COMMON_OCL_HPP

#include "gpu/ocl/ocl_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "gtest/gtest.h"

#include <CL/cl.h>

// Define a separate macro, that does not clash with OCL_CHECK from the library.
#ifdef DNNL_ENABLE_MEM_DEBUG
#define TEST_OCL_CHECK(x) \
    do { \
        dnnl_status_t s = dnnl::impl::gpu::ocl::convert_to_dnnl(x); \
        dnnl::error::wrap_c_api(s, dnnl_status2str(s)); \
    } while (0)
#else
#define TEST_OCL_CHECK(x) \
    do { \
        int s = int(x); \
        EXPECT_EQ(s, CL_SUCCESS) << "OpenCL error: " << s; \
    } while (0)
#endif

static inline cl_device_id find_ocl_device(cl_device_type dev_type) {
    cl_int err;
    const size_t max_platforms = 16;

    cl_uint nplatforms;
    cl_platform_id ocl_platforms[max_platforms];
    err = clGetPlatformIDs(max_platforms, ocl_platforms, &nplatforms);
    if (err != CL_SUCCESS) {
        // OpenCL has no support on the platform.
        return nullptr;
    }

    for (cl_uint i = 0; i < nplatforms; ++i) {
        cl_platform_id ocl_platform = ocl_platforms[i];

        const size_t max_platform_vendor_size = 256;
        std::string platform_vendor(max_platform_vendor_size + 1, 0);
        TEST_OCL_CHECK(clGetPlatformInfo(ocl_platform, CL_PLATFORM_VENDOR,
                max_platform_vendor_size * sizeof(char), &platform_vendor[0],
                nullptr));
        cl_uint ndevices;
        cl_device_id ocl_dev;
        err = clGetDeviceIDs(ocl_platform, dev_type, 1, &ocl_dev, &ndevices);
        if (err == CL_SUCCESS) { return ocl_dev; }
    }
    return nullptr;
}

// Base generic class providing RAII support for OpenCL objects
template <typename T, typename release_t = int32_t(T)>
struct ocl_wrapper_base_t {
    ocl_wrapper_base_t(T t, release_t *release = nullptr)
        : t_(t), release_(release) {}
    ocl_wrapper_base_t(ocl_wrapper_base_t &&other)
        : t_(other.t_), release_(other.release_) {
        other.t_ = nullptr;
    }
    ~ocl_wrapper_base_t() {
        if (release_ && t_) { release_(t_); }
    }

    ocl_wrapper_base_t(const ocl_wrapper_base_t &) = delete;
    ocl_wrapper_base_t &operator=(const ocl_wrapper_base_t &) = delete;

    operator T() const { return t_; }

private:
    T t_;
    release_t *release_;
};

// Auxiliary class providing RAII support for OpenCL objects,
// specialized for specific OpenCL types
template <typename T>
struct ocl_wrapper_t {};

template <>
struct ocl_wrapper_t<cl_device_id> : ocl_wrapper_base_t<cl_device_id> {
    ocl_wrapper_t(cl_device_id dev) : ocl_wrapper_base_t(dev) {}
};

template <>
struct ocl_wrapper_t<cl_context> : ocl_wrapper_base_t<cl_context> {
    ocl_wrapper_t(cl_context ctx)
        : ocl_wrapper_base_t(ctx, &clReleaseContext) {}
};

template <>
struct ocl_wrapper_t<cl_command_queue> : ocl_wrapper_base_t<cl_command_queue> {
    ocl_wrapper_t(cl_command_queue queue)
        : ocl_wrapper_base_t(queue, &clReleaseCommandQueue) {}
};

template <>
struct ocl_wrapper_t<cl_kernel> : ocl_wrapper_base_t<cl_kernel> {
    ocl_wrapper_t(cl_kernel kernel)
        : ocl_wrapper_base_t(kernel, &clReleaseKernel) {}
};

// Constructs an OpenCL wrapper object (providing RAII support)
template <typename T>
ocl_wrapper_t<T> make_ocl_wrapper(T t) {
    return ocl_wrapper_t<T>(t);
}

#endif
