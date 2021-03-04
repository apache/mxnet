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
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

namespace {
status_t resampling_desc_init(resampling_desc_t *resampling_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind, const float *factors,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc) {
    bool args_ok = true
            && one_of(alg_kind, resampling_nearest, resampling_linear)
            && src_desc && IMPLICATION(dst_desc == nullptr, factors)
            && utils::one_of(src_desc->ndims, 3, 4, 5);
    if (!args_ok) return invalid_arguments;

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    if (is_fwd) {
        args_ok = args_ok && src_desc->format_kind != format_kind::any;
        if (!args_ok) return invalid_arguments;
    }

    auto rd = resampling_desc_t();
    rd.primitive_kind = primitive_kind::resampling;
    rd.prop_kind = prop_kind;
    rd.alg_kind = alg_kind;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || (dst_desc
                    && memory_desc_wrapper(dst_desc)
                               .has_runtime_dims_or_strides());
    if (runtime_dims_or_strides) return unimplemented;

    auto fill_dst_md = [](const memory_desc_t *i_md, const float *factors,
                               memory_desc_t *o_md) {
        o_md->ndims = i_md->ndims;
        o_md->data_type = i_md->data_type;
        utils::array_copy(o_md->dims, i_md->dims, 2);
        for (int i = 0; i < o_md->ndims - 2; i++)
            o_md->dims[2 + i] = (dim_t)(i_md->dims[2 + i] * factors[i]);
        o_md->format_kind = format_kind::any;
    };

    (prop_kind == backward_data ? rd.diff_src_desc : rd.src_desc) = *src_desc;
    if (dst_desc)
        (is_fwd ? rd.dst_desc : rd.diff_dst_desc) = *dst_desc;
    else {
        dst_desc = (is_fwd ? &rd.dst_desc : &rd.diff_dst_desc);
        fill_dst_md(
                src_desc, factors, (is_fwd ? &rd.dst_desc : &rd.diff_dst_desc));
    }

    /* User provided factors are used only to compute destination dimensions.
     Implementation uses true scaling factors from source to destination */
    for (int i = 0; i < src_desc->ndims - 2; i++)
        rd.factors[i] = (float)((double)dst_desc->dims[2 + i]
                / src_desc->dims[2 + i]);

    bool consistency = src_desc->ndims == dst_desc->ndims
            && src_desc->dims[0] == dst_desc->dims[0]
            && src_desc->dims[1] == dst_desc->dims[1];

    if (!consistency) return invalid_arguments;

    *resampling_desc = rd;
    return success;
}
} // namespace

status_t dnnl_resampling_forward_desc_init(resampling_desc_t *resampling_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind, const float *factors,
        const memory_desc_t *src_desc, const memory_desc_t *dst_desc) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return resampling_desc_init(
            resampling_desc, prop_kind, alg_kind, factors, src_desc, dst_desc);
}

status_t dnnl_resampling_backward_desc_init(resampling_desc_t *resampling_desc,
        alg_kind_t alg_kind, const float *factors,
        const memory_desc_t *diff_src_desc,
        const memory_desc_t *diff_dst_desc) {
    return resampling_desc_init(resampling_desc, backward_data, alg_kind,
            factors, diff_src_desc, diff_dst_desc);
}
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
