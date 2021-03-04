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

#include "sycl/sycl_buffer_memory_storage.hpp"
#include "sycl/sycl_engine_base.hpp"

#include <CL/sycl.hpp>

#include "common/guard_manager.hpp"
#include "common/memory.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

struct map_buffer_tag;

sycl_buffer_memory_storage_t::sycl_buffer_memory_storage_t(engine_t *engine)
    : sycl_memory_storage_base_t(engine) {}

sycl_buffer_memory_storage_t::sycl_buffer_memory_storage_t(
        engine_t *engine, const memory_storage_t *parent_storage)
    : sycl_memory_storage_base_t(engine, parent_storage) {}

status_t sycl_buffer_memory_storage_t::map_data(
        void **mapped_ptr, stream_t *stream, size_t) const {
    if (!buffer_) {
        *mapped_ptr = nullptr;
        return status::success;
    }

    auto &guard_manager = guard_manager_t<map_buffer_tag>::instance();

    auto acc = buffer_->get_access<cl::sycl::access::mode::read_write>();
    auto *acc_ptr = new decltype(acc)(acc);
    *mapped_ptr = static_cast<void *>(acc_ptr->get_pointer());
    auto unmap_callback = [=]() { delete acc_ptr; };
    return guard_manager.enter(this, unmap_callback);
}

status_t sycl_buffer_memory_storage_t::unmap_data(
        void *mapped_ptr, stream_t *stream) const {
    if (!mapped_ptr) return status::success;

    auto &guard_manager = guard_manager_t<map_buffer_tag>::instance();
    return guard_manager.exit(this);
}

std::unique_ptr<memory_storage_t> sycl_buffer_memory_storage_t::get_sub_storage(
        size_t offset, size_t size) const {
    auto storage = utils::make_unique<sycl_buffer_memory_storage_t>(
            engine(), parent_storage());
    if (!storage) return nullptr;

    status_t status
            = storage->init(memory_flags_t::use_runtime_ptr, 0, nullptr);
    if (status != status::success) return nullptr;

    if (engine()->kind() == engine_kind::cpu) {
        storage->buffer_ = buffer_;
    } else {
#ifdef DNNL_SYCL_DPCPP
        buffer_u8_t *sub_buffer = buffer_
                ? new buffer_u8_t(parent_buffer(), base_offset_ + offset, size)
                : nullptr;
#endif
#ifdef DNNL_SYCL_COMPUTECPP
        // XXX: Workaround for ComputeCpp. Sub-buffers support is broken in
        // ComputeCpp compiler so we either return the existing buffer (if
        // offset is 0) or create a new buffer (assuming the caller doesn't
        // request the memory several times).
        // Apparently this workaround does NOT work in general case but at
        // least covers all the existing cases in the library at the moment.
        auto sub_buffer = (!buffer_ || size == 0) ? nullptr
                                                  : (base_offset_ + offset != 0)
                        ? new buffer_u8_t(cl::sycl::range<1>(size))
                        : new buffer_u8_t(*buffer_);
#endif
        storage->buffer_.reset(sub_buffer);
        storage->base_offset_ = base_offset_ + offset;
    }

    return storage;
}

std::unique_ptr<memory_storage_t> sycl_buffer_memory_storage_t::clone() const {
    auto storage = utils::make_unique<sycl_buffer_memory_storage_t>(engine());
    if (!storage) return nullptr;

    status_t status
            = storage->init(memory_flags_t::use_runtime_ptr, 0, nullptr);
    if (status != status::success) return nullptr;

    storage->buffer_ = buffer_;
    storage->base_offset_ = base_offset_;
    return storage;
}

status_t sycl_buffer_memory_storage_t::init_allocate(size_t size) {
    const auto &device
            = utils::downcast<sycl_engine_base_t *>(engine())->device();
    if (size > device.get_info<cl::sycl::info::device::max_mem_alloc_size>()) {
        return status::out_of_memory;
    }

    buffer_.reset(new buffer_u8_t(cl::sycl::range<1>(size)));
    if (!buffer_) return status::out_of_memory;
    return status::success;
}

buffer_u8_t &sycl_buffer_memory_storage_t::parent_buffer() const {
    return utils::downcast<const sycl_buffer_memory_storage_t *>(
            parent_storage())
            ->buffer();
}

} // namespace sycl
} // namespace impl
} // namespace dnnl
