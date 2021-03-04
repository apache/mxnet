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

#ifndef COMMON_INTERNAL_DESC_TYPES_HPP
#define COMMON_INTERNAL_DESC_TYPES_HPP

#include <vector>
#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {
namespace impl {

// The types are not exposed
struct dnnl_reorder_desc_t {
    dnnl_primitive_kind_t primitive_kind;
    dnnl_memory_desc_t src_md;
    dnnl_memory_desc_t dst_md;
    dnnl_engine_kind_t src_engine_kind;
    dnnl_engine_kind_t dst_engine_kind;
};

struct dnnl_concat_desc_t {
    dnnl_primitive_kind_t primitive_kind;
    dnnl_memory_desc_t dst_md;
    dnnl_dim_t n;
    dnnl_dim_t concat_dimension;
    std::vector<dnnl_memory_desc_t> src_mds;
};

struct dnnl_sum_desc_t {
    dnnl_primitive_kind_t primitive_kind;
    dnnl_memory_desc_t dst_md;
    dnnl_dim_t n;
    std::vector<float> scales;
    std::vector<dnnl_memory_desc_t> src_mds;
};

struct dnnl_zero_pad_desc_t {
    dnnl_primitive_kind_t primitive_kind;
};

} // namespace impl
} // namespace dnnl

#endif // INTERNAL_DESC_TYPES
