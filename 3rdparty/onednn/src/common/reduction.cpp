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

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "utils.hpp"

dnnl_status_t dnnl_reduction_desc_init(dnnl_reduction_desc_t *desc,
        dnnl_alg_kind_t alg_kind, const dnnl_memory_desc_t *src_desc,
        const dnnl_memory_desc_t *dst_desc, float p, float eps) {
    using namespace dnnl::impl;
    using namespace dnnl::impl::status;
    using namespace dnnl::impl::utils;
    using namespace dnnl::impl::alg_kind;

    bool args_ok = !any_null(desc, src_desc, dst_desc)
            && src_desc->format_kind != format_kind::any
            && one_of(alg_kind, reduction_max, reduction_min, reduction_sum,
                    reduction_mul, reduction_mean, reduction_norm_lp_max,
                    reduction_norm_lp_sum, reduction_norm_lp_power_p_max,
                    reduction_norm_lp_power_p_sum)
            && IMPLICATION(one_of(alg_kind, reduction_norm_lp_max,
                                   reduction_norm_lp_sum,
                                   reduction_norm_lp_power_p_max,
                                   reduction_norm_lp_power_p_sum),
                    p >= 1.0f)
            && IMPLICATION(one_of(alg_kind, reduction_mean,
                                   reduction_norm_lp_max, reduction_norm_lp_sum,
                                   reduction_norm_lp_power_p_max,
                                   reduction_norm_lp_power_p_sum),
                    one_of(src_desc->data_type, data_type::f32,
                            data_type::bf16));
    if (!args_ok) return invalid_arguments;

    if (src_desc->ndims != dst_desc->ndims) return invalid_arguments;

    for (auto d = 0; d < src_desc->ndims; ++d) {
        const auto dst_dim_d = dst_desc->dims[d];
        if (!one_of(dst_dim_d, 1, src_desc->dims[d])) return invalid_arguments;
    }

    // reduction primitive doesn't support identity operation
    if (array_cmp(src_desc->dims, dst_desc->dims, src_desc->ndims))
        return invalid_arguments;

    if (src_desc->format_kind != format_kind::blocked
            || !one_of(dst_desc->format_kind, format_kind::blocked,
                    format_kind::any))
        return invalid_arguments;

    if (src_desc->extra.flags != 0
            || !IMPLICATION(dst_desc->format_kind == format_kind::blocked,
                    dst_desc->extra.flags == 0))
        return invalid_arguments;

    auto rd = reduction_desc_t();
    rd.primitive_kind = primitive_kind::reduction;
    rd.alg_kind = alg_kind;

    rd.src_desc = *src_desc;
    rd.dst_desc = *dst_desc;

    rd.p = p;
    rd.eps = eps;

    *desc = rd;
    return success;
}
