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

#ifndef GUARD_MANAGER_HPP
#define GUARD_MANAGER_HPP

#include "common/c_types_map.hpp"

#include <functional>
#include <mutex>
#include <unordered_map>

namespace dnnl {
namespace impl {

// Service class to support RAII semantics with parameterized "finalization".
template <typename tag_type = void>
struct guard_manager_t : public c_compatible {
    guard_manager_t() = default;

    ~guard_manager_t() { assert(registered_callbacks.empty()); }

    static guard_manager_t &instance() {
        static guard_manager_t guard_manager;
        return guard_manager;
    }

    status_t enter(const void *ptr, const std::function<void()> &callback) {
        std::lock_guard<std::mutex> guard(mutex_);

        assert(registered_callbacks.count(ptr) == 0);
        registered_callbacks[ptr] = callback;
        return status::success;
    }

    status_t exit(const void *ptr) {
        std::lock_guard<std::mutex> guard(mutex_);

        assert(registered_callbacks.count(ptr) == 1);

        registered_callbacks[ptr]();
        registered_callbacks.erase(ptr);

        return status::success;
    }

private:
    std::unordered_map<const void *, std::function<void()>>
            registered_callbacks;
    std::mutex mutex_;
};

} // namespace impl
} // namespace dnnl

#endif
