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

#include "scratchpad_debug.hpp"

#include "engine.hpp"
#include "memory_debug.hpp"
#include "memory_tracking.hpp"

namespace dnnl {
namespace impl {
namespace scratchpad_debug {

void protect_scratchpad_buffer(void *scratchpad_ptr, engine_kind_t engine_kind,
        const memory_tracking::registry_t &registry) {
    if (scratchpad_ptr == nullptr) return;

    auto end = registry.end(scratchpad_ptr);
    auto curr = registry.begin(scratchpad_ptr);
    for (; curr != end; curr++) {
        std::pair<void *, size_t> data_range = *curr;
        memory_debug::protect_buffer(
                data_range.first, data_range.second, engine_kind);
    }
}
void unprotect_scratchpad_buffer(const void *scratchpad_ptr,
        engine_kind_t engine_kind,
        const memory_tracking::registry_t &registry) {
    if (scratchpad_ptr == nullptr) return;

    auto end = registry.cend(scratchpad_ptr);
    auto curr = registry.cbegin(scratchpad_ptr);
    for (; curr != end; curr++) {
        std::pair<const void *, size_t> data_range = *curr;
        memory_debug::unprotect_buffer(
                data_range.first, data_range.second, engine_kind);
    }
}

void protect_scratchpad_buffer(const memory_storage_t *storage,
        const memory_tracking::registry_t &registry) {
    if (storage != nullptr)
        protect_scratchpad_buffer(
                storage->data_handle(), storage->engine()->kind(), registry);
}
void unprotect_scratchpad_buffer(const memory_storage_t *storage,
        const memory_tracking::registry_t &registry) {
    if (storage != nullptr)
        unprotect_scratchpad_buffer(
                storage->data_handle(), storage->engine()->kind(), registry);
}
} // namespace scratchpad_debug
} // namespace impl
} // namespace dnnl
