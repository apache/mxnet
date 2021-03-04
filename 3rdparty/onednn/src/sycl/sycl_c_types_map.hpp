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

#ifndef SYCL_SYCL_C_TYPES_MAP_HPP
#define SYCL_SYCL_C_TYPES_MAP_HPP

#include "oneapi/dnnl/dnnl_sycl_types.h"

namespace dnnl {
namespace impl {
namespace sycl {

using memory_kind_t = dnnl_sycl_interop_memory_kind_t;
namespace memory_kind {
const memory_kind_t usm = dnnl_sycl_interop_usm;
const memory_kind_t buffer = dnnl_sycl_interop_buffer;
} // namespace memory_kind

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_SYCL_C_TYPES_MAP_HPP
