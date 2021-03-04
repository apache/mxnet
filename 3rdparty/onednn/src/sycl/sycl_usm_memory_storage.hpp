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

#ifndef SYCL_USM_MEMORY_STORAGE_HPP
#define SYCL_USM_MEMORY_STORAGE_HPP

#include "oneapi/dnnl/dnnl_config.h"

#ifdef DNNL_SYCL_DPCPP

#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "sycl/sycl_engine_base.hpp"
#include "sycl/sycl_memory_storage_base.hpp"

#include <memory>
#include <CL/sycl.hpp>

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_usm_memory_storage_t : public sycl_memory_storage_base_t {
public:
    sycl_usm_memory_storage_t(engine_t *engine)
        : sycl_memory_storage_base_t(engine) {}

    void *usm_ptr() const { return usm_ptr_.get(); }

    memory_kind_t memory_kind() const override { return memory_kind::usm; }

    virtual status_t get_data_handle(void **handle) const override {
        *handle = usm_ptr_.get();
        return status::success;
    }

    virtual status_t set_data_handle(void *handle) override {
        auto *sycl_engine = utils::downcast<sycl_engine_base_t *>(engine());
        auto &sycl_ctx = sycl_engine->context();

        usm_ptr_ = decltype(usm_ptr_)(handle, [](void *) {});
        usm_kind_ = cl::sycl::get_pointer_type(handle, sycl_ctx);

        return status::success;
    }

    virtual status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t size) const override;
    virtual status_t unmap_data(
            void *mapped_ptr, stream_t *stream) const override;

    virtual bool is_host_accessible() const override {
        if (engine()->kind() == engine_kind::cpu) return true; // optimism

        /* FIXME: remove the W/A below when possible (fixed driver).
         * Currently, shared USM suffers from synchronization issues between
         * different devices. Specifically, the data migration from GPU to CPU
         * may misbehave if accessed from multiple threads.
         *
         * One of the possible ways to work around the issue is to pretend that
         * shared USM is not accessible on the host, hence map will allocate an
         * extra memory and do a true copy of the data. This is what is done
         * here: the shared USM marked as not-host-accessible.
         *
         * There is another possible work-around: touch the data on the host in
         * the main thread, causing the "sequential" data migration. This W/A
         * is a little bit more preferable as it doesn't allocate extra memory
         * on the host. However it didn't work well for benchdnn, though worked
         * perfectly fine for gtests. As we weren't able to find the cause of
         * this behavior we went with the approach above. Hopefully, the driver
         * will be fixed and we can get rid of W/A altogether. */
        return utils::one_of(usm_kind_, cl::sycl::usm::alloc::host,
                // cl::sycl::usm::alloc::shared, // W/A (see above)
                cl::sycl::usm::alloc::unknown);
    }

    virtual std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override {
        void *sub_ptr = usm_ptr_.get()
                ? reinterpret_cast<uint8_t *>(usm_ptr_.get()) + offset
                : nullptr;
        auto storage = utils::make_unique<sycl_usm_memory_storage_t>(engine());
        if (!storage) return nullptr;
        storage->init(memory_flags_t::use_runtime_ptr, size, sub_ptr);
        return storage;
    }

    virtual std::unique_ptr<memory_storage_t> clone() const override {
        auto storage = utils::make_unique<sycl_usm_memory_storage_t>(engine());
        if (!storage) return nullptr;

        status_t status
                = storage->init(memory_flags_t::use_runtime_ptr, 0, nullptr);
        if (status != status::success) return nullptr;

        storage->usm_ptr_ = decltype(usm_ptr_)(usm_ptr_.get(), [](void *) {});
        storage->usm_kind_ = usm_kind_;

        return storage;
    }

protected:
    virtual status_t init_allocate(size_t size) override {
        auto *sycl_engine = utils::downcast<sycl_engine_base_t *>(engine());
        auto &sycl_dev = sycl_engine->device();
        auto &sycl_ctx = sycl_engine->context();

        usm_kind_ = cl::sycl::usm::alloc::shared;
        void *usm_ptr_alloc = cl::sycl::malloc_shared(size, sycl_dev, sycl_ctx);
        if (!usm_ptr_alloc) return status::out_of_memory;

        usm_ptr_ = decltype(usm_ptr_)(usm_ptr_alloc,
                [&](void *ptr) { cl::sycl::free(ptr, sycl_ctx); });
        return status::success;
    }

private:
    std::unique_ptr<void, std::function<void(void *)>> usm_ptr_;
    cl::sycl::usm::alloc usm_kind_ = cl::sycl::usm::alloc::unknown;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif

#endif // SYCL_USM_MEMORY_STORAGE_HPP
