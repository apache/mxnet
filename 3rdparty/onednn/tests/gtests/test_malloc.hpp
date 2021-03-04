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

#ifndef TEST_MALLOC_HPP
#define TEST_MALLOC_HPP

#ifdef DNNL_ENABLE_MEM_DEBUG
#include "src/common/internal_defs.hpp"

namespace dnnl {
namespace impl {
// Declare the dnnl::impl::malloc symbol as exported (dynamic linking) or
// strong (static linking) in order to redirect calls inside the library to
// the custom one.
void DNNL_STRONG *malloc(size_t size, int alignment);
} // namespace impl
} // namespace dnnl

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include <CL/cl.h>
namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
// Declare the dnnl::impl::gpu::ocl::clCreateBuffer_wrapper symbol as exported
// (dynamic linking) or strong (static linking) in order to redirect calls
// inside the library to the custom one.
cl_mem DNNL_STRONG clCreateBuffer_wrapper(cl_context context,
        cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret);
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

void reset_malloc_counter();
void increment_malloc_counter();
bool test_out_of_memory();
#else
static inline void reset_malloc_counter() {}
static inline void increment_malloc_counter() {}
static inline bool test_out_of_memory() {
    return false;
}
#endif

#endif
