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

#ifndef SYCL_STREAM_CPU_THUNK_HPP
#define SYCL_STREAM_CPU_THUNK_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"

#include <stddef.h>
#include <stdint.h>
#include <vector>

namespace dnnl {
namespace impl {
namespace sycl {

struct submit_ctx_t {
    stream_t *stream;
    const primitive_iface_t *prim_iface;
    exec_ctx_t exec_ctx;
    std::vector<const memory_storage_t *> sycl_mem_storages;

    submit_ctx_t(const exec_ctx_t &exec_ctx) : exec_ctx(exec_ctx) {}
};

struct thunk_params_t {
    static constexpr size_t max_size = 32;

    size_t size;
    uintptr_t native_pointers[max_size];
    uintptr_t submit_ctx_ptr;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

// OpenCL for CPU cannot find mangled functions so use
// C linkage for the thunk
extern "C" void DNNL_API dnnl_impl_sycl_cpu_thunk(
        const dnnl::impl::sycl::thunk_params_t *params);

#endif // SYCL_STREAM_CPU_THUNK_HPP
