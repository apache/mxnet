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

#ifndef GPU_JIT_JIT_GENERATOR_HPP
#define GPU_JIT_JIT_GENERATOR_HPP

#include "gpu/jit/jit_generator_base.hpp"
#include "gpu/jit/ngen/ngen_opencl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using gpu_gen_t = ngen::HW;
constexpr gpu_gen_t gpu_gen9 = ngen::HW::Gen9;
constexpr gpu_gen_t gpu_gen11 = ngen::HW::Gen11;
constexpr gpu_gen_t gpu_gen12lp = ngen::HW::Gen12LP;

// nGEN jit generator
//
// The main purpose of this header file is to provide extra features for nGEN
// kernel generator, e.g. additional macros and debugging capabilities.
//
// Jit generator provides additional memory to simplify kernel debugging. This
// memory is allocated using Shared Virtual Memory (SVM) feature in OpenCL 2.0.
// SVM enables the host and device portions of an OpenCL application to
// seamlessly share pointers and complex pointer-containing data-structures.
// This memory can be used to dump state of GPU registers or view GPU memory on
// the host in debugger.
//
// In order to use debug memory:
// 1.  Allocate it using 'void jit_generator::dbg_alloc(cl_context context)'
// 2.  Get memory pointer using 'void* jit_generator::dbg_memory()'
// 3.  Pass it as extra OpenCL kernel argument and define it as new argument in
//     kernel interface at corresponding order.
// 4.  Set a breakpoint after 'dnnl_stream_wait()', memory will be available on
//     the host side after kernel execution.
//
// A short example below demonstrates how to use debug memory:
//
//  ``` c++
//  status_t primitive_impl_t::execute(const exec_ctx_t &ctx) {
//      ...
//      auto gpu_engine = utils::downcast<ocl_gpu_engine*>(engine);
//      jit_generator->dbg_alloc(gpu_engine->context());
//      void* dbg_mem = jit_generator->dbg_memory();
//      ...
//      compute::kernel_arg_list_t arg_list;
//      arg_list.set(0, src);
//      arg_list.set(1, dst);
//      arg_list.set(2, dbg_mem, kernel_arg_t::kind_t::svm);
//      ...
//      compute_stream->parallel_for(nd_range, kernel_, arg_list);
//  }
//
//  ngen_kernel_t() : jit_generator<...>() {
//      externalName("ngen_kernel");
//      newArgument("src", GlobalPtr);
//      newArgument("dst", GlobalPtr);
//      newArgument("dbg_mem", GlobalPtr);
//      finalizeInterface();
//      ...
//      auto header = r32;
//      auto data = r64;
//      mov<uint64_t>(1, r64, getArgument("dbg_mem"));
//      store(1, scattered_dword(), A64, header, data);
//      ...
//  }
//  ```
//

template <gpu_gen_t hw>
class jit_generator : public ngen::OpenCLCodeGenerator<hw>,
                      public jit_generator_base {
private:
#ifdef CL_VERSION_2_0
    struct svm_deleter {
        cl_context context_;

        void operator()(void *ptr) noexcept {
            if (ptr) clSVMFree(context_, ptr);
        }
    };
    std::unique_ptr<void, svm_deleter> dbg_memory_;
#endif

public:
    jit_generator() = default;

    std::vector<unsigned char> get_binary(
            cl_context context, cl_device_id device) override {
        return ngen::OpenCLCodeGenerator<hw>::getBinary(context, device);
    }

    const char *kernel_name() const override {
        return ngen::OpenCLCodeGenerator<hw>::getExternalName().c_str();
    }

#ifdef CL_VERSION_2_0
    void dbg_alloc(cl_context context);
    void *dbg_memory() const { return dbg_memory_.get(); }
#endif
};

#ifdef CL_VERSION_2_0
template <gpu_gen_t hw>
void jit_generator<hw>::dbg_alloc(cl_context context) {
    constexpr size_t size = 1048576;
    void *mem = clSVMAlloc(
            context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, size, 0);
    dbg_memory_ = decltype(dbg_memory_)(mem, svm_deleter {context});
    memset(mem, 0xcd, size);
}
#endif

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_JIT_JIT_GENERATOR_HPP
