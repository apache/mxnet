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

#ifdef DNNL_ENABLE_MEM_DEBUG

#ifdef _WIN32
#include <malloc.h>
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "src/common/memory_debug.hpp"
#include "tests/gtests/test_malloc.hpp"

// Counter of failed mallocs caught during execution of
// catch_expected_failures(). It is used in combination when building with
// DNNL_ENABLE_MEM_DEBUG=ON, and is useless otherwise.
static size_t malloc_count = 0;

// Index of the current failed malloc. Once a malloc that has not been counted
// by malloc_count has been reached, malloc fails, and the counter is
// incremented. Since there may be allocations inside parallel regions, this
// index is thread_local to avoid race conditions, therefore only the master
// thread will trigger a failed malloc.
static std::atomic<size_t> malloc_idx(0);

// Reset the counter of total memory allocations.
void reset_malloc_counter() {
    malloc_count = 0;
    // Reset malloc index before re-running the test.
    malloc_idx = 0;
}

// Increment the counter of total memory allocations.
void increment_malloc_counter() {
    ++malloc_count;
    // Reset malloc index before re-running the test.
    malloc_idx = 0;
}

bool test_out_of_memory() {
    return true;
}

namespace dnnl {
namespace impl {

// Custom malloc that replaces the one in the library during dynamical linking.
// If a malloc, that has not been counted by malloc_count, has been reached the
// malloc will return nullptr.
void *malloc(size_t size, int alignment) {
    ++malloc_idx;
    if (malloc_idx > malloc_count) { return nullptr; }

    return memory_debug::malloc(size, alignment);
}
} // namespace impl
} // namespace dnnl

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include <CL/cl.h>

#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "tests/gtests/dnnl_test_common_ocl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Custom clCreateBuffer wrapper that replaces the one in the library during
// dynamical linking. If an allocation, that has not been counted by
// malloc_count, has been reached the wrapper allocator will fail.
cl_mem clCreateBuffer_wrapper(cl_context context, cl_mem_flags flags,
        size_t size, void *host_ptr, cl_int *errcode_ret) {
    ++malloc_idx;
    if (malloc_idx > malloc_count) {
        *errcode_ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;
        return nullptr;
    }

    return clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

#endif
