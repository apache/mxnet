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

#ifndef CPU_DW_CONVOLUTION_UTILS_HPP
#define CPU_DW_CONVOLUTION_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/primitive_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

inline status_t get_depthwise_conv_desc(convolution_desc_t &cd_dw,
        const memory_desc_t &src_dw_md, const primitive_attr_t &attr_1x1,
        primitive_attr_t &attr_dw, int dw_po_index) {

    const memory_desc_wrapper src_dw_d(src_dw_md);
    const int ndims = src_dw_d.ndims();
    if (ndims != 4) return status::unimplemented;

    if (dw_po_index == -1 || dw_po_index >= attr_1x1.post_ops_.len()
            || !attr_1x1.post_ops_.entry_[dw_po_index].is_convolution())
        return status::invalid_arguments;

    // Create new attributes with scales from depthwise post-op and copy
    // post-ops after depthwise post-op.
    auto &dw_po = attr_1x1.post_ops_.entry_[dw_po_index].depthwise_conv;
    if (utils::one_of(
                dw_po.dst_dt, data_type::u8, data_type::s8, data_type::s32)
            && dw_po.count) {
        CHECK(attr_dw.output_scales_.set(
                dw_po.count, dw_po.mask, dw_po.scales));
    }

    auto dw_po_len = attr_1x1.post_ops_.len() - (dw_po_index + 1);
    attr_dw.post_ops_.entry_.resize(dw_po_len);
    for (int i = 0; i < dw_po_len; ++i) {
        CHECK(attr_dw.post_ops_.entry_[i].copy_from(
                attr_1x1.post_ops_.entry_[i + dw_po_index + 1]));
    }

    attr_dw.scratchpad_mode_ = attr_1x1.scratchpad_mode_;

    const bool with_bias = dw_po.bias_dt != data_type::undef;

    const auto n = src_dw_d.dims()[0];
    const auto oc = src_dw_d.dims()[1];
    const auto g = src_dw_d.dims()[1];
    const auto ih = src_dw_d.dims()[ndims - 2];
    const auto iw = src_dw_d.dims()[ndims - 1];
    const auto stride = dw_po.stride;

    const dims_t weights_tz = {g, 1, 1, 3, 3};

    const dims_t dst_tz
            = {n, oc, utils::div_up(ih, stride), utils::div_up(iw, stride)};

    const dims_t bias_tz = {oc};
    const dims_t pad_tz = {1, 1};
    const dims_t stride_tz = {stride, stride};

    memory_desc_t src_md, weights_md, bias_md, dst_md;

    dnnl_memory_desc_init_by_tag(&src_md, ndims, src_dw_md.dims,
            src_dw_md.data_type, format_tag::any);

    dnnl_memory_desc_init_by_tag(
            &weights_md, ndims + 1, weights_tz, dw_po.wei_dt, format_tag::any);

    if (with_bias)
        dnnl_memory_desc_init_by_tag(
                &bias_md, 1, bias_tz, dw_po.bias_dt, format_tag::a);

    dnnl_memory_desc_init_by_tag(
            &dst_md, ndims, dst_tz, dw_po.dst_dt, format_tag::any);

    CHECK(conv_desc_init(&cd_dw, prop_kind::forward_inference,
            alg_kind::convolution_auto, &src_md, &weights_md,
            with_bias ? &bias_md : nullptr, &dst_md, stride_tz, nullptr, pad_tz,
            pad_tz));

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
