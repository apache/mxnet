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

#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_utils.hpp"

#include <vector>
#include <CL/sycl.hpp>

#include "sycl/level_zero_utils.hpp"

// Intel(R) oneAPI DPC++ Compiler uses reversed global work-item IDs starting
// from 10-24-2019.
// ComputeCpp version >= 1.1.6 uses reversed global work-item IDs.
#if defined(DNNL_SYCL_DPCPP) && (__SYCL_COMPILER_VERSION >= 20191024)
#define DNNL_SYCL_REVERSE_RANGE 1
#elif defined(DNNL_SYCL_COMPUTECPP) \
        && (COMPUTECPP_VERSION_MAJOR > 1 \
                || (COMPUTECPP_VERSION_MAJOR == 1 \
                        && (COMPUTECPP_VERSION_MINOR > 1 \
                                || (COMPUTECPP_VERSION_MINOR == 1 \
                                        && COMPUTECPP_VERSION_PATCH >= 6))))
#define DNNL_SYCL_REVERSE_RANGE 1
#else
#define DNNL_SYCL_REVERSE_RANGE 0
#endif

namespace dnnl {
namespace impl {
namespace sycl {

using buffer_u8_t = cl::sycl::buffer<uint8_t, 1>;

inline cl::sycl::range<3> to_sycl_range(const gpu::compute::nd_range_t &range) {
    auto *global_range = range.global_range();
#if DNNL_SYCL_REVERSE_RANGE
    auto sycl_global_range = cl::sycl::range<3>(
            global_range[2], global_range[1], global_range[0]);
#else
    auto sycl_global_range = cl::sycl::range<3>(
            global_range[0], global_range[1], global_range[2]);
#endif
    return sycl_global_range;
}

inline cl::sycl::nd_range<3> to_sycl_nd_range(
        const gpu::compute::nd_range_t &range) {
    auto *global_range = range.global_range();
    auto *local_range = range.local_range();

    auto sycl_global_range = to_sycl_range(range);

    if (!local_range) {
        assert(!"not expected");
        return cl::sycl::nd_range<3>(
                sycl_global_range, cl::sycl::range<3>(1, 1, 1));
    }

#if DNNL_SYCL_REVERSE_RANGE
    auto sycl_local_range = cl::sycl::range<3>(
            local_range[2], local_range[1], local_range[0]);
#else
    auto sycl_local_range = cl::sycl::range<3>(
            local_range[0], local_range[1], local_range[2]);
#endif
    return cl::sycl::nd_range<3>(sycl_global_range, sycl_local_range);
}

// Automatically use codeplay_host_task if it is supported by compiler,
// otherwise fall back to single_task.
template <typename K, typename H, typename F>
inline auto host_task_impl(H &cgh, F f, int)
        -> decltype(cgh.codeplay_host_task(f)) {
    cgh.codeplay_host_task(f);
}

template <typename K, typename H, typename F>
inline void host_task_impl(H &cgh, F f, long) {
    cgh.template single_task<K>(f);
}

template <typename K, typename H, typename F>
inline void host_task(H &cgh, F f) {
    // Third argument is 0 (int) which prefers the
    // run_on_host_intel option if both are available.
    host_task_impl<K>(cgh, f, 0);
}

enum class backend_t {
    unknown,
    host,
    level0,
    opencl,
};

inline std::string to_string(backend_t backend) {
    switch (backend) {
        case backend_t::host: return "Host";
        case backend_t::level0: return "Level Zero";
        case backend_t::opencl: return "OpenCL";
        default: return "Unknown";
    }
}

backend_t get_sycl_gpu_backend();

inline backend_t get_sycl_backend(const cl::sycl::device &dev) {
    if (dev.is_host()) return backend_t::host;

#ifdef DNNL_SYCL_DPCPP
    auto plat = dev.get_platform();
    std::string plat_name = plat.get_info<cl::sycl::info::platform::name>();
    if (plat_name.find("OpenCL") != std::string::npos) return backend_t::opencl;

    if (plat_name.find("Level-Zero") != std::string::npos)
        return backend_t::level0;

    return backend_t::unknown;
#else
    return backend_t::opencl;
#endif
}

inline bool are_equal(
        const cl::sycl::device &lhs, const cl::sycl::device &rhs) {
    auto lhs_be = get_sycl_backend(lhs);
    auto rhs_be = get_sycl_backend(rhs);
    if (lhs_be != rhs_be) return false;

    // Only one host device exists.
    if (lhs_be == backend_t::host) return true;

    // Compare cl_device_id for OpenCL backend.
    if (lhs_be == backend_t::opencl) {
        // Use wrapper objects to avoid memory leak.
        auto lhs_ocl_dev = gpu::ocl::make_ocl_wrapper(lhs.get());
        auto rhs_ocl_dev = gpu::ocl::make_ocl_wrapper(rhs.get());
        return lhs_ocl_dev == rhs_ocl_dev;
    }

    // Other backends do not retain the returned handles.
    auto lhs_handle = lhs.get();
    auto rhs_handle = rhs.get();

    return lhs_handle == rhs_handle;
}

inline device_id_t sycl_device_id(const cl::sycl::device &dev) {
    if (dev.is_host())
        return std::make_tuple(static_cast<int>(backend_t::host), 0, 0);

    device_id_t device_id
            = device_id_t {static_cast<int>(backend_t::unknown), 0, 0};
#ifdef DNNL_SYCL_DPCPP
    switch (get_sycl_backend(dev)) {
        case backend_t::opencl:
            device_id = std::make_tuple(static_cast<int>(backend_t::opencl),
                    reinterpret_cast<uint64_t>(dev.get()), 0);
            break;
        case backend_t::level0: {
#if defined(DNNL_WITH_LEVEL_ZERO)
            device_id = std::tuple_cat(
                    std::make_tuple(static_cast<int>(backend_t::level0)),
                    get_device_uuid(dev));
#else
            assert(!"unreachable");
#endif
            break;
        }
        case backend_t::unknown: assert(!"unknown backend"); break;
        default: assert(!"unreachable");
    }
#else
    device_id = std::make_tuple(static_cast<int>(backend_t::opencl),
            reinterpret_cast<uint64_t>(dev.get()), 0);
#endif
    assert(std::get<0>(device_id) != static_cast<int>(backend_t::unknown));
    return device_id;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#undef DNNL_SYCL_REVERSE_RANGE

#endif
