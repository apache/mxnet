/*******************************************************************************
* Copyright 2020 Arm Ltd. and affiliates
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

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "common/bfloat16.hpp"
#include "cpu/aarch64/acl_gemm_convolution_utils.hpp"

#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::alg_kind;
using namespace prop_kind;
using namespace data_type;

namespace acl_gemm_convolution_utils {

status_t init_conf(acl_conv_gemm_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr) {

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);

    // Compute Library currently supports forward propagation only
    const prop_kind_t prop_kind = cd.prop_kind;
    const bool is_bwd_d = prop_kind == backward_data;
    const bool is_bwd_w = prop_kind == backward_weights;
    const bool is_fwd = !(is_bwd_d || is_bwd_w);
    if (!is_fwd) return status::unimplemented;

    // Current implementation does not support int8 or bf16
    bool is_int8_conv = utils::one_of(src_d.data_type(), s8, u8)
            && weights_d.data_type() == s8;
    bool is_bf16_conv = utils::everyone_is(
            bf16, src_d.data_type(), weights_d.data_type());
    if (is_int8_conv || is_bf16_conv) return status::unimplemented;

    const int with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;
    bool is_nspc;

    // Compute Library unsupported shape scenarios
    if (one_of(true, is_3d, is_1d, with_groups)) {
        return status::unimplemented;
    }

    // batch size
    const int mb = src_d.dims()[0];

    // src/input  channels, height, width
    const int ic = src_d.dims()[1];
    const int ih = src_d.dims()[ndims - 2];
    const int iw = src_d.dims()[ndims - 1];

    // dst/output channels, height, width
    const int oc = dst_d.dims()[1];
    const int oh = dst_d.dims()[ndims - 2];
    const int ow = dst_d.dims()[ndims - 1];

    // weights height and width
    const int kh = weights_d.dims()[with_groups + ndims - 2];
    const int kw = weights_d.dims()[with_groups + ndims - 1];

    // left, right, top, bottom padding
    const int l_pad = cd.padding[0][1];
    const int r_pad = cd.padding[1][1];
    const int t_pad = cd.padding[0][0];
    const int b_pad = cd.padding[1][0];

    // height and width strides
    const int stride_h = cd.strides[ndims - 4];
    const int stride_w = cd.strides[ndims - 3];

    acp.padstride_info = arm_compute::PadStrideInfo(stride_w, stride_h, l_pad,
            r_pad, t_pad, b_pad, arm_compute::DimensionRoundingType::FLOOR);

    // height and width dilations
    int dilate_h = cd.dilates[ndims - 4];
    int dilate_w = cd.dilates[ndims - 3];
    // oneDNN dilations:          dk = 1 + (k_size - 1) * (dilate_size + 1)
    // Compute Library dilations: dk = dilate_size * (k_size - 1) + 1
    // thus acl_dilation = oneDNN_dilation + 1
    dilate_h += 1;
    dilate_w += 1;

    acp.dilation_info = arm_compute::Size2D(dilate_w, dilate_h);

    acp.with_bias = cd.bias_desc.format_kind != format_kind::undef
            || cd.diff_bias_desc.format_kind != format_kind::undef;

    auto set_or_check_tags = [&](format_tag_t desired_src_tag,
                                     format_tag_t desired_dst_tag) -> status_t {
        using namespace format_tag;
        auto src_tag = any, dst_tag = any;

        if (src_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, desired_src_tag));
            src_tag = desired_src_tag;
        } else {
            src_tag = memory_desc_matches_one_of_tag(
                    src_md, nwc, nhwc, ncw, nchw);
        }

        if (dst_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(dst_md, desired_dst_tag));
            dst_tag = desired_dst_tag;
        } else {
            dst_tag = memory_desc_matches_one_of_tag(
                    dst_md, nwc, nhwc, ncw, nchw);
        }

        if (acp.with_bias && bias_md.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));

        is_nspc = utils::one_of(src_tag, nwc, nhwc);

        memory_desc_t want_wei_md = weights_md;
        auto wei_tag = is_nspc ? utils::pick(ndims - 3, wio, hwio)
                               : utils::pick(ndims - 3, oiw, oihw);
        CHECK(memory_desc_init_by_tag(want_wei_md, wei_tag));

        // Compute Library does not support mismatching layouts
        if ((src_tag != wei_tag) || (src_tag != dst_tag))
            return status::unimplemented;

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
        }
        return (want_wei_md == weights_md) ? status::success
                                           : status::unimplemented;
    };

    // TODO: look into changing default tag to the Compute Library default NHWC
    auto default_dat_tag
            = utils::pick(ndims - 3, format_tag::ncw, format_tag::nchw);
    if (set_or_check_tags(default_dat_tag, default_dat_tag) != status::success)
        return status::unimplemented;

    const auto acl_layout = is_nspc ? arm_compute::DataLayout::NHWC
                                    : arm_compute::DataLayout::NCHW;

    // clang-format off
    acp.src_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(iw, ih, ic, mb),
            1,
            arm_compute::DataType::F32,
            acl_layout);

    acp.wei_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(kw, kh, ic, oc),
            1,
            arm_compute::DataType::F32,
            acl_layout);

    acp.dst_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(ow, oh, oc, mb),
            1,
            arm_compute::DataType::F32,
            acl_layout);

    acp.bia_info = arm_compute::TensorInfo(
            acp.with_bias ? arm_compute::TensorShape(oc)
                          : arm_compute::TensorShape(),
            1,
            arm_compute::DataType::F32,
            acl_layout);
    // clang-format on

    // Post-op activations
    acp.act_info = acl_gemm_convolution_utils::get_acl_act(attr);

    return status::success;
}

arm_compute::ActivationLayerInfo get_acl_act(const primitive_attr_t &attr) {
    const auto &post_ops = attr.post_ops_;
    const int entry_idx = post_ops.find(primitive_kind::eltwise);
    if (entry_idx == -1) { return arm_compute::ActivationLayerInfo(); }

    const auto eltwise_alg = post_ops.entry_[entry_idx].eltwise.alg;
    float alpha = post_ops.entry_[entry_idx].eltwise.alpha;
    float beta = post_ops.entry_[entry_idx].eltwise.beta;

    using acl_act_t = arm_compute::ActivationLayerInfo::ActivationFunction;
    acl_act_t acl_act_alg;
    switch (eltwise_alg) {
        case eltwise_relu:
            // oneDNN defines RELU: f(x) = (x > 0) ? x : a*x
            // Compute Library defines LEAKY_RELU: f(x) = (x > 0) ? x : a*x
            // whilst Compute Library RELU is defined as: f(x) = max(0,x)
            if (alpha == 0) {
                acl_act_alg = acl_act_t::RELU;
            } else {
                acl_act_alg = acl_act_t::LEAKY_RELU;
            }
            break;
        case eltwise_tanh:
            // oneDNN defines TANH activation as:          f(x) = tanh(x)
            // Compute Library defines TANH activation as: f(x) = a*tanh(b*x)
            // Setting a=b=1 makes the two equivalent
            alpha = 1.f;
            beta = 1.f;
            acl_act_alg = acl_act_t::TANH;
            break;
        case eltwise_elu: acl_act_alg = acl_act_t::ELU; break;
        case eltwise_square: acl_act_alg = acl_act_t::SQUARE; break;
        case eltwise_abs: acl_act_alg = acl_act_t::ABS; break;
        case eltwise_sqrt: acl_act_alg = acl_act_t::SQRT; break;
        case eltwise_linear: acl_act_alg = acl_act_t::LINEAR; break;
        case eltwise_bounded_relu: acl_act_alg = acl_act_t::BOUNDED_RELU; break;
        case eltwise_soft_relu: acl_act_alg = acl_act_t::SOFT_RELU; break;
        case eltwise_logistic: acl_act_alg = acl_act_t::LOGISTIC; break;
        default: return arm_compute::ActivationLayerInfo();
    }

    return arm_compute::ActivationLayerInfo(acl_act_alg, alpha, beta);
}

bool acl_act_ok(alg_kind_t eltwise_activation) {
    return utils::one_of(eltwise_activation, eltwise_relu, eltwise_tanh,
            eltwise_elu, eltwise_square, eltwise_abs, eltwise_sqrt,
            eltwise_linear, eltwise_bounded_relu, eltwise_soft_relu,
            eltwise_logistic);
}

} // namespace acl_gemm_convolution_utils

} // namespace cpu
} // namespace impl
} // namespace dnnl
