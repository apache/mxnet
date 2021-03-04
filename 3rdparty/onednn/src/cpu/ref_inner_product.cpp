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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/ref_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using math::get_bias;

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type,
        data_type_t acc_type>
void ref_inner_product_fwd_t<src_type, wei_type, dst_type,
        acc_type>::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto IC = pd()->IC();

    const bool src_has_spatial = utils::one_of(src_d.ndims(), 3, 4, 5);
    const auto ndims = src_d.ndims() - 2;

    auto maybe_oscale = [=](float &d, dim_t oc) {
        // scale_idx_mult = 1 for per_oc scales and 0, otherwise
        const int scale_idx_mult
                = pd()->attr()->output_scales_.mask_ == (1 << 1);
        const float *scales = pd()->attr()->output_scales_.scales_;
        d *= scales[oc * scale_idx_mult];
    };

    auto ker_has_spatial = [=](dim_t mb, dim_t oc) {
        acc_data_t d = 0;
        const dim_t KD = pd()->KD();
        const dim_t KH = pd()->KH();
        const dim_t KW = pd()->KW();
        for_(dim_t ic = 0; ic < IC; ++ic)
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            switch (ndims) {
                case 3:
                    d += (acc_data_t)src[src_d.off(mb, ic, kd, kh, kw)]
                            * weights[weights_d.off(oc, ic, kd, kh, kw)];
                    break;
                case 2:
                    d += (acc_data_t)src[src_d.off(mb, ic, kh, kw)]
                            * weights[weights_d.off(oc, ic, kh, kw)];
                    break;
                case 1:
                    d += (acc_data_t)src[src_d.off(mb, ic, kw)]
                            * weights[weights_d.off(oc, ic, kw)];
                    break;
                default: assert(!"unsupported ndims size");
            }
        }
        return d;
    };

    auto ker_no_spatial = [=](dim_t mb, dim_t oc) {
        acc_data_t d = 0;
        for (dim_t ic = 0; ic < IC; ++ic) {
            d += (acc_data_t)src[src_d.off(mb, ic)]
                    * weights[weights_d.off(oc, ic)];
        }
        return d;
    };

    parallel_nd(MB, OC, [&](dim_t mb, dim_t oc) {
        float a = bias ? get_bias(bias, bias_d.off(oc),
                          pd()->desc()->bias_desc.data_type)
                       : 0;
        if (src_has_spatial)
            a += ker_has_spatial(mb, oc);
        else
            a += ker_no_spatial(mb, oc);

        dim_t dst_off = dst_d.off(mb, oc);
        dim_t dst_l_off = (mb * OC + oc);

        maybe_oscale(a, oc);

        ref_post_ops_t::args_t args;
        args.dst_val = dst[dst_off];
        args.ctx = &ctx;
        args.l_offset = dst_l_off;
        args.dst_md = pd()->dst_md();
        ref_post_ops->execute(a, args);

        dst[dst_off] = cpu::saturate_and_round<dst_data_t>(a);
    });
}

using namespace data_type;
template struct ref_inner_product_fwd_t<f32>;
template struct ref_inner_product_fwd_t<bf16, bf16, bf16, f32>;
template struct ref_inner_product_fwd_t<bf16, bf16, f32, f32>;
template struct ref_inner_product_fwd_t<u8, s8, f32, s32>;
template struct ref_inner_product_fwd_t<u8, s8, s32, s32>;
template struct ref_inner_product_fwd_t<u8, s8, s8, s32>;
template struct ref_inner_product_fwd_t<u8, s8, u8, s32>;
template struct ref_inner_product_fwd_t<s8, s8, f32, s32>;
template struct ref_inner_product_fwd_t<s8, s8, s32, s32>;
template struct ref_inner_product_fwd_t<s8, s8, s8, s32>;
template struct ref_inner_product_fwd_t<s8, s8, u8, s32>;

template <data_type_t diff_src_type, data_type_t wei_type,
        data_type_t diff_dst_type, data_type_t acc_type>
void ref_inner_product_bwd_data_t<diff_src_type, wei_type, diff_dst_type,
        acc_type>::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto IC = pd()->IC();

    const bool diff_src_has_spatial
            = utils::one_of(diff_src_d.ndims(), 3, 4, 5);
    const auto ndims = diff_src_d.ndims() - 2;

    parallel_nd(MB, IC, [&](dim_t mb, dim_t ic) {
        if (diff_src_has_spatial) {
            const dim_t KD = pd()->KD();
            const dim_t KH = pd()->KH();
            const dim_t KW = pd()->KW();
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                acc_data_t ds = acc_data_t(0);
                for (dim_t oc = 0; oc < OC; ++oc) {
                    switch (ndims) {
                        case 3:
                            ds += (acc_data_t)(diff_dst[diff_dst_d.off(mb, oc)]
                                    * weights[weights_d.off(
                                            oc, ic, kd, kh, kw)]);
                            break;
                        case 2:
                            ds += (acc_data_t)(diff_dst[diff_dst_d.off(mb, oc)]
                                    * weights[weights_d.off(oc, ic, kh, kw)]);
                            break;
                        case 1:
                            ds += (acc_data_t)(diff_dst[diff_dst_d.off(mb, oc)]
                                    * weights[weights_d.off(oc, ic, kw)]);
                            break;
                        default: assert(!"unsupported ndims size");
                    }
                }
                switch (ndims) {
                    case 3:
                        diff_src[diff_src_d.off(mb, ic, kd, kh, kw)]
                                = (diff_src_data_t)ds;
                        break;
                    case 2:
                        diff_src[diff_src_d.off(mb, ic, kh, kw)]
                                = (diff_src_data_t)ds;
                        break;
                    case 1:
                        diff_src[diff_src_d.off(mb, ic, kw)]
                                = (diff_src_data_t)ds;
                        break;
                    default: assert(!"unsupported ndims size");
                }
            }
        } else {
            acc_data_t ds = acc_data_t(0);
            for (dim_t oc = 0; oc < OC; ++oc) {
                ds += (acc_data_t)(diff_dst[diff_dst_d.off(mb, oc)]
                        * weights[weights_d.off(oc, ic)]);
            }
            diff_src[diff_src_d.off(mb, ic)] = (diff_src_data_t)ds;
        }
    });
}

template struct ref_inner_product_bwd_data_t<f32, f32, f32, f32>;

template <data_type_t data_type>
void ref_inner_product_bwd_weights_t<data_type>::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_BIAS);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto IC = pd()->IC();

    const bool src_has_spatial = utils::one_of(src_d.ndims(), 3, 4, 5);
    const auto ndims = src_d.ndims() - 2;

    parallel_nd(OC, IC, [&](dim_t oc, dim_t ic) {
        if (src_has_spatial) {
            const dim_t KD = pd()->KD();
            const dim_t KH = pd()->KH();
            const dim_t KW = pd()->KW();
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                data_t *dw(nullptr);
                switch (ndims) {
                    case 3:
                        dw = &diff_weights[diff_weights_d.off(
                                oc, ic, kd, kh, kw)];
                        break;
                    case 2:
                        dw = &diff_weights[diff_weights_d.off(oc, ic, kh, kw)];
                        break;
                    case 1:
                        dw = &diff_weights[diff_weights_d.off(oc, ic, kw)];
                        break;
                    default: assert(!"unsupported ndims size");
                }
                *dw = data_t(0);
                for (dim_t mb = 0; mb < MB; ++mb) {
                    switch (ndims) {
                        case 3:
                            *dw += diff_dst[diff_dst_d.off(mb, oc)]
                                    * src[src_d.off(mb, ic, kd, kh, kw)];
                            break;
                        case 2:
                            *dw += diff_dst[diff_dst_d.off(mb, oc)]
                                    * src[src_d.off(mb, ic, kh, kw)];
                            break;
                        case 1:
                            *dw += diff_dst[diff_dst_d.off(mb, oc)]
                                    * src[src_d.off(mb, ic, kw)];
                            break;
                        default: assert(!"unsupported ndims size");
                    }
                }
            }
        } else {
            data_t *dw = &diff_weights[diff_weights_d.off(oc, ic)];
            *dw = data_t(0);
            for (dim_t mb = 0; mb < MB; ++mb) {
                *dw += diff_dst[diff_dst_d.off(mb, oc)]
                        * src[src_d.off(mb, ic)];
            }
        }
    });

    if (diff_bias) {
        diff_bias += diff_bias_d.offset0();

        parallel_nd(OC, [&](dim_t oc) {
            data_t *db = &diff_bias[oc];
            *db = data_t(0);
            for (dim_t mb = 0; mb < MB; ++mb)
                *db += diff_dst[diff_dst_d.off(mb, oc)];
        });
    }
}

template struct ref_inner_product_bwd_weights_t<data_type::f32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
