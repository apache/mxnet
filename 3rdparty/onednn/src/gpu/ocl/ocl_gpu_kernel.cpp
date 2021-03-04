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

#include <assert.h>
#include <string>
#include <CL/cl.h>

#include "gpu/ocl/ocl_gpu_kernel.hpp"

#include "common/utils.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

ocl_gpu_kernel_t::~ocl_gpu_kernel_t() {
    if (ocl_kernel_) OCL_CHECK_V(clReleaseKernel(ocl_kernel_));
}

status_t ocl_gpu_kernel_t::parallel_for(stream_t &stream,
        const compute::nd_range_t &range,
        const compute::kernel_arg_list_t &arg_list) const {
    assert(state_ == state_t::kernel);

    auto *ocl_stream = utils::downcast<ocl_stream_t *>(&stream);
    cl_command_queue queue = ocl_stream->queue();

    assert(ocl_kernel_ && "kernel is NULL");

    for (int i = 0; i < arg_list.nargs(); ++i) {
        auto &arg = arg_list.get(i);
        cl_int set_err;
        if (arg.is_global()) {
            auto *mem_storage
                    = static_cast<const memory_storage_t *>(arg.value());
            cl_mem ocl_mem = nullptr;
            if (!mem_storage->is_null()) {
                auto *ocl_mem_storage
                        = utils::downcast<const ocl_memory_storage_t *>(
                                mem_storage);

                // Validate that the OpenCL contexts match for execution
                // context and memory.
                auto stream_ocl_ctx
                        = utils::downcast<ocl_gpu_engine_t *>(stream.engine())
                                  ->context();
                auto memory_storage_ocl_ctx
                        = utils::downcast<ocl_gpu_engine_t *>(
                                ocl_mem_storage->engine())
                                  ->context();
                if (stream_ocl_ctx != memory_storage_ocl_ctx) {
                    MAYBE_REPORT_ERROR(
                            "mismatched OpenCL context for primitive/memory");
                    return status::invalid_arguments;
                }

                ocl_mem = ocl_mem_storage->mem_object();
            }
            set_err = clSetKernelArg(ocl_kernel_, i, sizeof(cl_mem), &ocl_mem);
        } else if (arg.is_local()) {
            set_err = clSetKernelArg(ocl_kernel_, i, arg.size(), arg.value());
        } else if (arg.is_svm_pointer()) {
#ifdef CL_VERSION_2_0
            set_err = clSetKernelArgSVMPointer(ocl_kernel_, i, arg.value());
#else
            return status::runtime_error; // SVM is not supported
#endif // CL_VERSION_2_0
        } else {
            compute::scalar_type_t real_arg_type;
            CHECK(get_ocl_kernel_arg_type(&real_arg_type, ocl_kernel_, i));
            // Convert if types do not match.
            typename std::aligned_storage<sizeof(float), sizeof(float)>::type
                    tmp_storage;
            void *cast_storage = &tmp_storage;
            auto cvt_arg = compute::kernel_arg_t::cast(
                    real_arg_type, arg, cast_storage);
            set_err = clSetKernelArg(
                    ocl_kernel_, i, cvt_arg.size(), cvt_arg.value());
        }
        status_t status = convert_to_dnnl(set_err);
        if (status != status::success) return status;
    }

    cl_uint ndims = static_cast<cl_uint>(range.ndims());
    if (range.is_zero()) { return status::success; }
    cl_int err = clEnqueueNDRangeKernel(queue, ocl_kernel_, ndims, nullptr,
            range.global_range(), range.local_range(), 0, nullptr, nullptr);
    status_t status = convert_to_dnnl(err);
    return status;
}

status_t ocl_gpu_kernel_t::realize(
        compute::kernel_t *kernel, engine_t *engine) const {
    assert(state_ == state_t::binary);
    if (binary_.empty()) return status::success;
    auto *compute_engine = utils::downcast<ocl_gpu_engine_t *>(engine);

    cl_int err;
    cl_device_id dev = compute_engine->device();
    const unsigned char *binary_buffer = binary_.data();
    size_t binary_size = binary_.size();
    assert(binary_size > 0);

    auto program = clCreateProgramWithBinary(compute_engine->context(), 1, &dev,
            &binary_size, &binary_buffer, nullptr, &err);
    OCL_CHECK(err);
    err = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
    OCL_CHECK(err);
    cl_kernel ocl_kernel = clCreateKernel(program, name(), &err);
    OCL_CHECK(err);
    (*kernel) = compute::kernel_t(new ocl_gpu_kernel_t(ocl_kernel));
    OCL_CHECK(clReleaseProgram(program));

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
