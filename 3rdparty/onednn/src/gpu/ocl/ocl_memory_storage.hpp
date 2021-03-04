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

#ifndef GPU_OCL_OCL_MEMORY_STORAGE_HPP
#define GPU_OCL_OCL_MEMORY_STORAGE_HPP

#include <CL/cl.h>

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class ocl_memory_storage_t : public memory_storage_t {
public:
    ocl_memory_storage_t(engine_t *engine)
        : memory_storage_t(engine), mem_object_(nullptr) {}

    ocl_memory_storage_t(
            engine_t *engine, const memory_storage_t *parent_storage)
        : memory_storage_t(engine, parent_storage) {}

    status_t get_data_handle(void **handle) const override {
        *handle = static_cast<void *>(mem_object_.get());
        return status::success;
    }

    status_t set_data_handle(void *handle) override {
        mem_object_ = ocl_wrapper_t<cl_mem>(static_cast<cl_mem>(handle), true);
        return status::success;
    }

    status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t) const override;
    status_t unmap_data(void *mapped_ptr, stream_t *stream) const override;

    cl_mem mem_object() const { return mem_object_.get(); }

    bool is_host_accessible() const override { return false; }

    std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override;

    virtual std::unique_ptr<memory_storage_t> clone() const override;

protected:
    status_t init_allocate(size_t size) override;

private:
    cl_mem parent_mem_object() const;

    ocl_wrapper_t<cl_mem> mem_object_;
    size_t base_offset_ = 0;

    DNNL_DISALLOW_COPY_AND_ASSIGN(ocl_memory_storage_t);
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
