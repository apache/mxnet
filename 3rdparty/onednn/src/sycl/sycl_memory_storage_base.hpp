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

#ifndef SYCL_MEMORY_STORAGE_BASE_HPP
#define SYCL_MEMORY_STORAGE_BASE_HPP

#include "common/memory_storage.hpp"
#include "sycl/sycl_c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_memory_storage_base_t : public memory_storage_t {
public:
    using memory_storage_t::memory_storage_t;

    virtual memory_kind_t memory_kind() const = 0;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_MEMORY_STORAGE_BASE_HPP
