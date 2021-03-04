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

#ifndef SYCL_BUFFER_MEMORY_STORAGE_HPP
#define SYCL_BUFFER_MEMORY_STORAGE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_c_types_map.hpp"
#include "sycl/sycl_memory_storage_base.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_buffer_memory_storage_t : public sycl_memory_storage_base_t {
public:
    sycl_buffer_memory_storage_t(engine_t *engine);

    sycl_buffer_memory_storage_t(
            engine_t *engine, const memory_storage_t *parent_storage);

    buffer_u8_t &buffer() const { return *buffer_; }

    memory_kind_t memory_kind() const override { return memory_kind::buffer; }

    virtual status_t get_data_handle(void **handle) const override {
        *handle = static_cast<void *>(buffer_.get());
        return status::success;
    }

    virtual status_t set_data_handle(void *handle) override {
        if (!handle) return status::success;

        auto *buf_u8_ptr = static_cast<buffer_u8_t *>(handle);
        buffer_.reset(new buffer_u8_t(*buf_u8_ptr));
        return status::success;
    }

    virtual size_t base_offset() const override { return base_offset_; }

    virtual status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t) const override;
    virtual status_t unmap_data(
            void *mapped_ptr, stream_t *stream) const override;

    virtual bool is_host_accessible() const override { return false; }

    virtual std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override;

    virtual std::unique_ptr<memory_storage_t> clone() const override;

protected:
    virtual status_t init_allocate(size_t size) override;

private:
    buffer_u8_t &parent_buffer() const;

    std::shared_ptr<buffer_u8_t> buffer_;
    size_t base_offset_ = 0;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_BUFFER_MEMORY_STORAGE_HPP
