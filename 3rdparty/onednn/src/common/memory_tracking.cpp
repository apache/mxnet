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

#include "memory_tracking.hpp"
#include "primitive_exec_types.hpp"

#include "engine.hpp"

namespace dnnl {
namespace impl {
namespace memory_tracking {

const void *registry_t::entry_t::compute_ptr(const void *base_ptr) const {
    if (size == 0) return nullptr;
    assert(base_ptr != nullptr);

    char *ptr = (char *)base_ptr + offset;
    char *aligned_ptr = utils::align_ptr<char>(ptr, get_alignment(alignment));

    if (memory_debug::is_mem_debug_overflow() && size % getpagesize() != 0) {
        // Align to end of page
        size_t page_end_offset = utils::rnd_up(size, alignment) % getpagesize();
        aligned_ptr += getpagesize() - page_end_offset;
        if (aligned_ptr - getpagesize() > ptr) aligned_ptr -= getpagesize();
        assert((size_t)aligned_ptr % alignment == 0);
    }

    assert(aligned_ptr + size <= ptr + capacity - buffer_protect_size());
    return (const void *)aligned_ptr;
}

char *grantor_t::get_host_storage_ptr(const memory_storage_t *storage) const {
    assert(storage != nullptr);
    return (char *)exec_ctx_->host_ptr(storage);
}

bool grantor_t::is_cpu_engine(const memory_storage_t *mem_storage) const {
    if (!mem_storage) return false;
    auto engine = mem_storage->engine();
    assert(engine);
    if (engine->kind() == engine_kind::cpu) return true;
    return false;
}

} // namespace memory_tracking
} // namespace impl
} // namespace dnnl
