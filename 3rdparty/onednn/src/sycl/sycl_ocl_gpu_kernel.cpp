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

#include <CL/sycl.hpp>

#include "common/utils.hpp"
#include "gpu/zero_pad_struct.h"
#include "sycl/level_zero_utils.hpp"
#include "sycl/sycl_c_types_map.hpp"
#include "sycl/sycl_ocl_gpu_kernel.hpp"
#include "sycl/sycl_stream.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

static void set_scalar_arg(
        cl::sycl::handler &cgh, int index, size_t size, const void *value) {
    switch (size) {
        case sizeof(uint8_t):
            cgh.set_arg(index, *static_cast<const uint8_t *>(value));
            break;
        case sizeof(uint16_t):
            cgh.set_arg(index, *static_cast<const uint16_t *>(value));
            break;
        case sizeof(uint32_t):
            cgh.set_arg(index, *static_cast<const uint32_t *>(value));
            break;
        case sizeof(uint64_t):
            cgh.set_arg(index, *static_cast<const uint64_t *>(value));
            break;
        case sizeof(zero_pad_mask_t):
            cgh.set_arg(index, *static_cast<const zero_pad_mask_t *>(value));
            break;
        default:
            assert(!"Please add another case");
            throw std::runtime_error("Internal error");
    }
}

status_t sycl_ocl_gpu_kernel_t::realize(
        gpu::compute::kernel_t *kernel, engine_t *engine) const {
    assert(state_ == state_t::binary);
    if (binary_.size() == 0) return status::success;
    auto *compute_engine = utils::downcast<sycl_gpu_engine_t *>(engine);

    cl_device_id ocl_device;
    cl_context ocl_context;
    std::unique_ptr<gpu::ocl::ocl_gpu_engine_t> ocl_engine;

    if (compute_engine->backend() == backend_t::opencl) {
        ocl_device = compute_engine->ocl_device();
        ocl_context = compute_engine->ocl_context();
    } else if (compute_engine->backend() == backend_t::level0) {
        // FIXME: This does not work for multi-GPU systems. OpenCL engine
        // should be created based on the Level0 device to ensure that the
        // program is created for the same physical device that was used
        // to create the binary. However, OpenCL does not provide any API to
        // match its devices with Level0.
        //
        // Currently we always create an OpenCL engine for the 0th device at
        // binary creation time and here.
        gpu::ocl::ocl_engine_factory_t f(engine_kind::gpu);
        engine_t *ocl_engine_ptr;
        CHECK(f.engine_create(&ocl_engine_ptr, 0));
        ocl_engine.reset(
                utils::downcast<gpu::ocl::ocl_gpu_engine_t *>(ocl_engine_ptr));
        ocl_device = ocl_engine->device();
        ocl_context = ocl_engine->context();
    } else {
        assert(!"not expected");
        return status::invalid_arguments;
    }

    cl_int err;
    const unsigned char *binary_buffer = binary_.data();
    size_t binary_size = binary_.size();
    assert(binary_size > 0);

    auto program = clCreateProgramWithBinary(ocl_context, 1, &ocl_device,
            &binary_size, &binary_buffer, nullptr, &err);
    OCL_CHECK(err);
    err = clBuildProgram(program, 1, &ocl_device, nullptr, nullptr, nullptr);
    OCL_CHECK(err);
    cl_kernel ocl_kernel = clCreateKernel(program, name(), &err);
    OCL_CHECK(err);
    (*kernel) = gpu::compute::kernel_t(new sycl_ocl_gpu_kernel_t(ocl_kernel));
    OCL_CHECK(clReleaseProgram(program));

    return status::success;
}

status_t sycl_create_kernel(std::unique_ptr<cl::sycl::kernel> &sycl_kernel,
        const sycl_gpu_engine_t *sycl_engine, cl_kernel ocl_kernel,
        void **handle_to_destroy) {
    cl_program ocl_program;
    OCL_CHECK(clGetKernelInfo(ocl_kernel, CL_KERNEL_PROGRAM,
            sizeof(ocl_program), &ocl_program, nullptr));

    std::string kernel_name(128, '\0');
    OCL_CHECK(clGetKernelInfo(ocl_kernel, CL_KERNEL_FUNCTION_NAME,
            kernel_name.size(), &kernel_name[0], nullptr));

    if (sycl_engine->backend() == backend_t::opencl) {
        cl::sycl::program sycl_program(sycl_engine->context(), ocl_program);
        sycl_kernel.reset(
                new cl::sycl::kernel(sycl_program.get_kernel(kernel_name)));
        return status::success;
    }

#if defined(DNNL_SYCL_DPCPP) && defined(DNNL_WITH_LEVEL_ZERO)
    if (sycl_engine->backend() != backend_t::level0)
        return status::invalid_arguments;

    size_t binary_size = 0;
    OCL_CHECK(clGetProgramInfo(ocl_program, CL_PROGRAM_BINARY_SIZES,
            sizeof(size_t), &binary_size, nullptr));

    std::vector<unsigned char> binary(binary_size);
    auto *binary_ptr = binary.data();
    OCL_CHECK(clGetProgramInfo(ocl_program, CL_PROGRAM_BINARIES, binary_size,
            &binary_ptr, nullptr));

    return sycl_create_kernel_with_level_zero(
            sycl_kernel, sycl_engine, binary, kernel_name, handle_to_destroy);
#else
    return status::invalid_arguments;
#endif
}

status_t sycl_ocl_gpu_kernel_t::parallel_for(stream_t &stream,
        const gpu::compute::nd_range_t &range,
        const gpu::compute::kernel_arg_list_t &arg_list) const {
    if (range.is_zero()) return status::success;
    auto *sycl_stream = utils::downcast<sycl::sycl_stream_t *>(&stream);
    auto *sycl_engine
            = utils::downcast<sycl::sycl_gpu_engine_t *>(sycl_stream->engine());
    auto &queue = sycl_stream->queue();

    std::unique_ptr<cl::sycl::kernel> sycl_kernel;
    void *handle_to_destroy = nullptr;
    CHECK(sycl_create_kernel(
            sycl_kernel, sycl_engine, ocl_kernel_, &handle_to_destroy));

    auto event = queue.submit([&](cl::sycl::handler &cgh) {
#ifdef DNNL_SYCL_DPCPP
        cgh.depends_on(sycl_stream->get_deps());
#endif
        for (int i = 0; i < arg_list.nargs(); ++i) {
            auto &arg = arg_list.get(i);
            if (arg.is_global()) {
                auto *mem_storage
                        = static_cast<const memory_storage_t *>(arg.value());
                if (*mem_storage) {
                    auto *sycl_mem_storage = utils::downcast<
                            const sycl_memory_storage_base_t *>(mem_storage);
                    switch (sycl_mem_storage->memory_kind()) {
                        case memory_kind::buffer: {
                            auto *m = utils::downcast<
                                    const sycl_buffer_memory_storage_t *>(
                                    mem_storage);
                            auto &sycl_buf = m->buffer();
                            cgh.set_arg((int)i,
                                    sycl_buf.get_access<
                                            cl::sycl::access::mode::read_write>(
                                            cgh));
                            break;
                        }
#ifdef DNNL_SYCL_DPCPP
                        case memory_kind::usm: {
                            auto *m = utils::downcast<
                                    const sycl_usm_memory_storage_t *>(
                                    mem_storage);
                            cgh.set_arg((int)i, m->usm_ptr());
                            break;
                        }
#endif
                        default: assert(!"not expected");
                    }
                } else {
                    cgh.set_arg((int)i, nullptr);
                }
            } else if (arg.is_local()) {
                auto acc = cl::sycl::accessor<uint8_t, 1,
                        cl::sycl::access::mode::read_write,
                        cl::sycl::access::target::local>(
                        cl::sycl::range<1>(arg.size()), cgh);
                cgh.set_arg((int)i, acc);
            } else {
                gpu::compute::scalar_type_t real_arg_type;
                gpu::ocl::get_ocl_kernel_arg_type(
                        &real_arg_type, ocl_kernel_, i);
                typename std::aligned_storage<sizeof(float),
                        sizeof(float)>::type tmp_storage;
                void *cast_storage = &tmp_storage;
                auto cvt_arg = gpu::compute::kernel_arg_t::cast(
                        real_arg_type, arg, cast_storage);
                set_scalar_arg(cgh, (int)i, cvt_arg.size(), cvt_arg.value());
            }
        }
        if (range.local_range()) {
            auto sycl_nd_range = to_sycl_nd_range(range);
            cgh.parallel_for(sycl_nd_range, *sycl_kernel);
        } else {
            auto sycl_range = to_sycl_range(range);
            cgh.parallel_for(sycl_range, *sycl_kernel);
        }
    });

#if defined(DNNL_SYCL_DPCPP) && defined(DNNL_WITH_LEVEL_ZERO)
    if (sycl_engine->backend() == backend_t::level0)
        CHECK(sycl_destroy_kernel_with_level_zero(
                sycl_kernel, handle_to_destroy));
#endif

    sycl_stream->set_deps({event});
    return status::success;
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
