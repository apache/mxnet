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

#include <assert.h>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

status_t dnnl_binary_desc_init(binary_desc_t *binary_desc, alg_kind_t alg_kind,
        const memory_desc_t *src0_md, const memory_desc_t *src1_md,
        const memory_desc_t *dst_md) {
    bool args_ok = true && !any_null(binary_desc, src0_md, src1_md, dst_md)
            && one_of(alg_kind, binary_add, binary_mul, binary_max, binary_min,
                    binary_div);
    if (!args_ok) return invalid_arguments;

    auto bod = binary_desc_t();
    bod.primitive_kind = primitive_kind::binary;
    bod.alg_kind = alg_kind;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src0_md).has_runtime_dims_or_strides()
            || memory_desc_wrapper(src1_md).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_md).has_runtime_dims_or_strides();
    if (runtime_dims_or_strides) return unimplemented;

    bod.src_desc[0] = *src0_md;
    bod.src_desc[1] = *src1_md;
    bod.dst_desc = *dst_md;

    const int ndims = src0_md->ndims;
    const dims_t &dims = src0_md->dims;

    if (src1_md->ndims != ndims) return invalid_arguments;
    for (int d = 0; d < ndims; ++d) {
        if (!(src1_md->dims[d] == dims[d] || src1_md->dims[d] == 1))
            return invalid_arguments;
    }

    // check dst
    if (dst_md->format_kind == format_kind::blocked) {
        if (!memory_desc_wrapper(dst_md).similar_to(
                    memory_desc_wrapper(src0_md), true, false))
            return invalid_arguments;
    } else {
        if (dst_md->ndims != ndims) return invalid_arguments;
        for (int d = 0; d < ndims; ++d) {
            if (dst_md->dims[d] != dims[d]) return invalid_arguments;
        }
    }

    *binary_desc = bod;
    return success;
}
