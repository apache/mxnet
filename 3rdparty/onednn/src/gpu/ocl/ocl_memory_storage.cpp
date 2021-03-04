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

#include <CL/cl.h>

#include "gpu/ocl/ocl_memory_storage.hpp"

#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ocl_memory_storage_t::init_allocate(size_t size) {
    auto *ocl_engine = utils::downcast<ocl_gpu_engine_t *>(engine());
    cl_int err;
    mem_object_ = clCreateBuffer_wrapper(
            ocl_engine->context(), CL_MEM_READ_WRITE, size, nullptr, &err);
    OCL_CHECK(err);
    return status::success;
}

namespace {
status_t get_map_queue(
        cl_command_queue &queue, engine_t *engine, stream_t *stream) {
    ocl_stream_t *ocl_stream;
    if (stream == nullptr) {
        auto *ocl_engine = utils::downcast<ocl_gpu_engine_t *>(engine);
        status_t status = ocl_engine->get_service_stream(stream);
        if (status != status::success) { return status::runtime_error; }
    }
    ocl_stream = utils::downcast<ocl_stream_t *>(stream);
    queue = ocl_stream->queue();
    return status::success;
}
} // namespace

status_t ocl_memory_storage_t::map_data(
        void **mapped_ptr, stream_t *stream, size_t) const {
    if (!mem_object()) {
        *mapped_ptr = nullptr;
        return status::success;
    }

    cl_mem_flags mem_flags;
    OCL_CHECK(clGetMemObjectInfo(mem_object(), CL_MEM_FLAGS, sizeof(mem_flags),
            &mem_flags, nullptr));

    size_t mem_bytes;
    OCL_CHECK(clGetMemObjectInfo(
            mem_object(), CL_MEM_SIZE, sizeof(mem_bytes), &mem_bytes, nullptr));

    cl_map_flags map_flags = 0;
    if (mem_flags & CL_MEM_READ_WRITE) {
        map_flags |= CL_MAP_READ;
        map_flags |= CL_MAP_WRITE;
    } else if (mem_flags & CL_MEM_READ_ONLY) {
        map_flags |= CL_MAP_READ;
    } else if (mem_flags & CL_MEM_WRITE_ONLY) {
        map_flags |= CL_MAP_WRITE;
    }

    cl_command_queue queue;
    CHECK(get_map_queue(queue, engine(), stream));

    // Use blocking operation to simplify the implementation and API
    cl_int err;
    *mapped_ptr = clEnqueueMapBuffer(queue, mem_object(), CL_TRUE, map_flags, 0,
            mem_bytes, 0, nullptr, nullptr, &err);
    return convert_to_dnnl(err);
}

status_t ocl_memory_storage_t::unmap_data(
        void *mapped_ptr, stream_t *stream) const {
    if (!mapped_ptr) return status::success;
    cl_command_queue queue;
    CHECK(get_map_queue(queue, engine(), stream));
    OCL_CHECK(clEnqueueUnmapMemObject(queue, mem_object_,
            const_cast<void *>(mapped_ptr), 0, nullptr, nullptr));
    OCL_CHECK(clFinish(queue));
    return status::success;
}

std::unique_ptr<memory_storage_t> ocl_memory_storage_t::get_sub_storage(
        size_t offset, size_t size) const {
    // Fast return on size = 0.
    // It also seems clCreateSubBuffer() does not work properly for such case.
    // Assumption: returned sub-storage won't be used for extracting cl_mem.
    if (size == 0) return nullptr;

    cl_mem_flags mem_flags;
    cl_int err;
    err = clGetMemObjectInfo(
            mem_object(), CL_MEM_FLAGS, sizeof(mem_flags), &mem_flags, nullptr);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) return nullptr;

    assert(size != 0);
    cl_buffer_region buffer_region = {base_offset_ + offset, size};
    ocl_wrapper_t<cl_mem> sub_buffer = clCreateSubBuffer(parent_mem_object(),
            mem_flags, CL_BUFFER_CREATE_TYPE_REGION, &buffer_region, &err);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) return nullptr;

    auto sub_storage
            = new ocl_memory_storage_t(this->engine(), parent_storage());
    if (sub_storage) {
        sub_storage->init(memory_flags_t::use_runtime_ptr, size, sub_buffer);
        sub_storage->base_offset_ = base_offset_ + offset;
    }
    return std::unique_ptr<memory_storage_t>(sub_storage);
}

std::unique_ptr<memory_storage_t> ocl_memory_storage_t::clone() const {
    auto storage = new ocl_memory_storage_t(engine());
    if (storage) storage->init(memory_flags_t::use_runtime_ptr, 0, mem_object_);
    return std::unique_ptr<memory_storage_t>(storage);
}

cl_mem ocl_memory_storage_t::parent_mem_object() const {
    return utils::downcast<const ocl_memory_storage_t *>(parent_storage())
            ->mem_object();
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
