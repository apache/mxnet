/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef COMMON_VERBOSE_HPP
#define COMMON_VERBOSE_HPP

#include <cinttypes>
#include <mutex>
#include <stdio.h>

#include "c_types_map.hpp"
#include "oneapi/dnnl/dnnl_debug.h"
#include "utils.hpp"
#include "z_magic.hpp"

namespace dnnl {
namespace impl {

struct verbose_t {
    int level;
};

int get_verbose();
double get_msec();

#if !defined(DISABLE_VERBOSE)
#define DNNL_VERBOSE_BUF_LEN 1024
#else
#define DNNL_VERBOSE_BUF_LEN 1
#endif

/// A container for primitive desc verbose string.
struct primitive_desc_t;
struct pd_info_t {
    pd_info_t() = default;
    pd_info_t(const pd_info_t &rhs)
        : str_(rhs.str_), is_initialized_(rhs.is_initialized_) {}
    pd_info_t &operator=(const pd_info_t &rhs) {
        is_initialized_ = rhs.is_initialized_;
        str_ = rhs.str_;
        return *this;
    }

    const char *c_str() const { return str_.c_str(); }
    bool is_initialized() const { return is_initialized_; }

    void init(engine_t *engine, const primitive_desc_t *pd);

private:
    std::string str_;

#if defined(DISABLE_VERBOSE)
    bool is_initialized_ = true; // no verbose -> info is always ready
#else
    bool is_initialized_ = false;
#endif

    // Alas, `std::once_flag` cannot be manually set and/or copied (in terms of
    // its state). Hence, when `pd_info_t` is copied the `initialization_flag_`
    // is always reset. To avoid re-initialization we use an extra
    // `is_initialized_` flag, that should be checked before calling `init()`.
    std::once_flag initialization_flag_;
};

} // namespace impl
} // namespace dnnl

#endif
