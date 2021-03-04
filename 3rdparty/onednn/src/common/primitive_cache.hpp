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

#ifndef COMMON_PRIMITIVE_CACHE_HPP
#define COMMON_PRIMITIVE_CACHE_HPP

#include <future>
#include <list>
#include <memory>
#include <unordered_map>

#include "c_types_map.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "primitive_hashing.hpp"
#include "rw_mutex.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_t;
struct primitive_cache_t : public c_compatible {
    struct cache_value_t {
        std::shared_ptr<primitive_t> primitive;
        status_t status;
    };
    using key_t = primitive_hashing::key_t;
    using value_t = std::shared_future<cache_value_t>;

    virtual ~primitive_cache_t() = default;

    virtual status_t set_capacity(int capacity) = 0;
    virtual int get_capacity() const = 0;

    virtual value_t get_or_add(
            const key_t &key, const value_t &value, bool need_lock)
            = 0;
    virtual void remove_if_invalidated(const key_t &key, bool need_lock) = 0;

    virtual int get_size() const = 0;

protected:
    static utils::rw_mutex_t &rw_mutex() {
        static utils::rw_mutex_t mutex;
        return mutex;
    }

    void lock_read(bool need_lock) {
        if (need_lock) rw_mutex().lock_read();
    }

    void lock_write(bool need_lock) {
        if (need_lock) rw_mutex().lock_write();
    }

    void unlock_read(bool need_lock) {
        if (need_lock) rw_mutex().unlock_read();
    }

    void unlock_write(bool need_lock) {
        if (need_lock) rw_mutex().unlock_write();
    }
};

// The cache uses LRU replacement policy
struct lru_primitive_cache_t : public primitive_cache_t {
    lru_primitive_cache_t(int capacity) : capacity_(capacity) {}

    ~lru_primitive_cache_t() override = default;

    status_t set_capacity(int capacity) override;
    int get_capacity() const override;

    value_t get_or_add(
            const key_t &key, const value_t &value, bool need_lock) override;
    void remove_if_invalidated(const key_t &key, bool need_lock) override;

    int get_size() const override;

private:
    void evict(size_t n);
    void add(const key_t &key, const value_t &value);
    value_t get(const key_t &key);

    size_t capacity_;
    using cache_list_t = std::list<std::pair<key_t, value_t>>;
    cache_list_t cache_list_;
    std::unordered_map<key_t, cache_list_t::iterator> cache_mapper_;
};

primitive_cache_t &primitive_cache();

status_t DNNL_API get_primitive_cache_size(int *size);

} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
