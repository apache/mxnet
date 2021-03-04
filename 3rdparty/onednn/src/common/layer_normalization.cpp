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
using namespace dnnl::impl::types;

namespace {
status_t lnorm_desc_init(layer_normalization_desc_t *lnorm_desc,
        prop_kind_t prop_kind, const memory_desc_t *data_desc,
        const memory_desc_t *stat_desc, const memory_desc_t *diff_data_desc,
        float epsilon, unsigned flags) {
    bool args_ok = true && !any_null(lnorm_desc, data_desc)
            && one_of(prop_kind, forward_training, forward_inference,
                    backward_data, backward)
            && 2 <= data_desc->ndims && data_desc->ndims <= 5
            && IMPLICATION(prop_kind & backward, diff_data_desc != nullptr)
            && (flags & ~(dnnl_use_global_stats | dnnl_use_scaleshift)) == 0;
    if (!args_ok) return invalid_arguments;

    auto ld = layer_normalization_desc_t();
    ld.primitive_kind = primitive_kind::layer_normalization;
    ld.prop_kind = prop_kind;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(data_desc).has_runtime_dims_or_strides()
            || (stat_desc
                    && memory_desc_wrapper(stat_desc)
                               .has_runtime_dims_or_strides());
    if (one_of(prop_kind, backward_data, backward))
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_data_desc)
                           .has_runtime_dims_or_strides();
    if (runtime_dims_or_strides) return unimplemented;

    ld.data_desc = *data_desc;
    ld.stat_desc = zero_md();
    ld.diff_data_desc = zero_md();
    if (one_of(ld.prop_kind, backward_data, backward))
        ld.diff_data_desc = *diff_data_desc;

    if (stat_desc)
        ld.stat_desc = *stat_desc;
    else
        CHECK(dnnl_memory_desc_init_by_tag(&ld.stat_desc,
                ld.data_desc.ndims - 1, ld.data_desc.dims, data_type::f32,
                format_tag::any));

    int ndims = data_desc->ndims;
    dims_t scaleshift_dims = {2, data_desc->dims[ndims - 1]};
    dnnl_memory_desc_init_by_tag(&ld.data_scaleshift_desc, 2, scaleshift_dims,
            data_type::f32, dnnl_nc);
    ld.diff_data_scaleshift_desc = zero_md();
    if (ld.prop_kind == backward) {
        ld.diff_data_scaleshift_desc = ld.data_scaleshift_desc;
    }

    ld.layer_norm_epsilon = epsilon;
    ld.flags = flags;

    if (ld.prop_kind == backward_data) {
        bool consistency = ld.diff_data_desc.ndims == ld.data_desc.ndims
                && array_cmp(ld.diff_data_desc.dims, ld.data_desc.dims,
                        ld.diff_data_desc.ndims)
                && ld.data_desc.ndims == ld.stat_desc.ndims + 1
                && array_cmp(ld.stat_desc.dims, ld.data_desc.dims,
                        ld.stat_desc.ndims);
        if (!consistency) return invalid_arguments;
    }

    *lnorm_desc = ld;
    return success;
}
} // namespace

status_t dnnl_layer_normalization_forward_desc_init(
        layer_normalization_desc_t *lnorm_desc, prop_kind_t prop_kind,
        const memory_desc_t *data_desc, const memory_desc_t *stat_desc,
        float epsilon, unsigned flags) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return lnorm_desc_init(lnorm_desc, prop_kind, data_desc, stat_desc, nullptr,
            epsilon, flags);
}

status_t dnnl_layer_normalization_backward_desc_init(
        layer_normalization_desc_t *lnorm_desc, prop_kind_t prop_kind,
        const memory_desc_t *diff_data_desc, const memory_desc_t *data_desc,
        const memory_desc_t *stat_desc, float epsilon, unsigned flags) {
    if (!one_of(prop_kind, backward, backward_data)) return invalid_arguments;
    return lnorm_desc_init(lnorm_desc, prop_kind, data_desc, stat_desc,
            diff_data_desc, epsilon, flags);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
