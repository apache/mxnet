/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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
#include "math_utils.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

namespace {
status_t eltwise_desc_init(eltwise_desc_t *eltwise_desc, prop_kind_t prop_kind,
        alg_kind_t alg_kind, const memory_desc_t *data_desc,
        const memory_desc_t *diff_data_desc, float alpha, float beta) {
    bool args_ok = true && !any_null(eltwise_desc, data_desc)
            && one_of(prop_kind, forward_training, forward_inference,
                    backward_data)
            && IMPLICATION(
                    prop_kind == backward_data, diff_data_desc != nullptr)
            && IMPLICATION(alg_kind == eltwise_round,
                    one_of(prop_kind, forward_training, forward_inference))
            && math::is_eltwise_ok(data_desc->data_type, alg_kind, alpha, beta);
    if (!args_ok) return invalid_arguments;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(data_desc).has_runtime_dims_or_strides();
    if (prop_kind == backward_data)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(diff_data_desc)
                           .has_runtime_dims_or_strides();
    if (runtime_dims_or_strides) return unimplemented;

    auto ed = eltwise_desc_t();
    ed.primitive_kind = primitive_kind::eltwise;
    ed.prop_kind = prop_kind;
    ed.alg_kind = alg_kind;

    ed.data_desc = *data_desc;
    if (ed.prop_kind == backward_data) ed.diff_data_desc = *diff_data_desc;

    ed.alpha = alpha;
    ed.beta = beta;

    bool consistency = true
            && IMPLICATION(ed.prop_kind == backward_data,
                    array_cmp(ed.diff_data_desc.dims, ed.data_desc.dims,
                            ed.diff_data_desc.ndims));
    if (!consistency) return invalid_arguments;

    *eltwise_desc = ed;
    return success;
}
} // namespace

status_t dnnl_eltwise_forward_desc_init(eltwise_desc_t *eltwise_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *data_desc, float alpha, float beta) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;
    return eltwise_desc_init(
            eltwise_desc, prop_kind, alg_kind, data_desc, nullptr, alpha, beta);
}

status_t dnnl_eltwise_backward_desc_init(eltwise_desc_t *eltwise_desc,
        alg_kind_t alg_kind, const memory_desc_t *diff_data_desc,
        const memory_desc_t *data_desc, float alpha, float beta) {
    return eltwise_desc_init(eltwise_desc, backward_data, alg_kind, data_desc,
            diff_data_desc, alpha, beta);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
