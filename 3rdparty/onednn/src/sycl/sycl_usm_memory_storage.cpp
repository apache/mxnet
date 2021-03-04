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

#include "sycl/sycl_usm_memory_storage.hpp"

#ifdef DNNL_SYCL_DPCPP

#include <CL/sycl.hpp>

#include "common/guard_manager.hpp"
#include "common/memory.hpp"
#include "common/utils.hpp"

#include "sycl/sycl_engine_base.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

struct map_usm_tag;

status_t sycl_usm_memory_storage_t::map_data(
        void **mapped_ptr, stream_t *stream, size_t size) const {
    void *usm_ptr = this->usm_ptr(); // shadowing is bad

    if (!usm_ptr || size == 0) {
        *mapped_ptr = nullptr;
        return status::success;
    }

    if (is_host_accessible()) {
        *mapped_ptr = usm_ptr;
        return status::success;
    }

    if (!stream) CHECK(engine()->get_service_stream(stream));

    cl::sycl::queue sycl_queue
            = utils::downcast<sycl_stream_t *>(stream)->queue();

    void *host_ptr = cl::sycl::malloc_host(size, sycl_queue.get_context());
    if (!host_ptr) return status::out_of_memory;

    sycl_queue.wait_and_throw();
    sycl_queue.memcpy(host_ptr, usm_ptr, size).wait();

    *mapped_ptr = host_ptr;
    auto unmap_callback = [=]() mutable {
        sycl_queue.wait_and_throw();
        sycl_queue.memcpy(usm_ptr, host_ptr, size).wait();
        cl::sycl::free(host_ptr, sycl_queue.get_context());
    };

    auto &guard_manager = guard_manager_t<map_usm_tag>::instance();
    return guard_manager.enter(this, unmap_callback);
}

status_t sycl_usm_memory_storage_t::unmap_data(
        void *mapped_ptr, stream_t *stream) const {
    if (!mapped_ptr || is_host_accessible()) return status::success;

    auto &guard_manager = guard_manager_t<map_usm_tag>::instance();
    return guard_manager.exit(this);
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif
