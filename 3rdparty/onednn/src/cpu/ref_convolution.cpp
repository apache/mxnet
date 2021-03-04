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

#include "cpu/cpu_primitive.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/ref_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using math::get_bias;

namespace {
inline dim_t get_data_off(const memory_desc_wrapper &mdw, int ndims, dim_t mb,
        dim_t c, dim_t id, dim_t ih, dim_t iw) {
    switch (ndims) {
        case 5: return mdw.off(mb, c, id, ih, iw);
        case 4: return mdw.off(mb, c, ih, iw);
        case 3: return mdw.off(mb, c, iw);
        default: assert(!"unsupported ndims"); return dim_t(0);
    }
}

inline dim_t get_weights_off(const memory_desc_wrapper &mdw, bool with_groups,
        int ndims, dim_t g, dim_t oc, dim_t ic, dim_t kd, dim_t kh, dim_t kw) {
    switch (ndims) {
        case 5:
            return with_groups ? mdw.off(g, oc, ic, kd, kh, kw)
                               : mdw.off(oc, ic, kd, kh, kw);
        case 4:
            return with_groups ? mdw.off(g, oc, ic, kh, kw)
                               : mdw.off(oc, ic, kh, kw);
        case 3:
            return with_groups ? mdw.off(g, oc, ic, kw) : mdw.off(oc, ic, kw);
        default: assert(!"unsupported ndims"); return dim_t(0);
    }
}
} // namespace

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type,
        data_type_t acc_type>
status_t
ref_convolution_fwd_t<src_type, wei_type, dst_type, acc_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    DEFINE_ZERO_POINTS_BUFFER(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(dst_zero_point, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const bool with_groups = pd()->with_groups();

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OD = pd()->OD();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto ID = pd()->ID();
    const auto IH = pd()->IH();
    const auto IW = pd()->IW();

    const auto OC = pd()->OC() / G;
    const auto IC = pd()->IC() / G;
    const auto KD = pd()->KD();
    const auto KH = pd()->KH();
    const auto KW = pd()->KW();

    const auto KSD = pd()->KSD();
    const auto KSH = pd()->KSH();
    const auto KSW = pd()->KSW();

    const auto KDD = pd()->KDD() + 1;
    const auto KDH = pd()->KDH() + 1;
    const auto KDW = pd()->KDW() + 1;

    const auto padFront = pd()->padFront();
    const auto padT = pd()->padT();
    const auto padL = pd()->padL();

    const auto ndims = pd()->desc()->src_desc.ndims;

    using namespace data_type;
    bool is_int_conv = utils::one_of(src_type, s32, s8, u8);

    auto maybe_oscale = [=](float &d, dim_t g, dim_t oc) {
        // scale_idx_mult = 1 for per_oc scales and 0, otherwise
        const int scale_idx_mult
                = pd()->attr()->output_scales_.mask_ == (1 << 1);
        const float *scales = pd()->attr()->output_scales_.scales_;
        d *= scales[(g * OC + oc) * scale_idx_mult];
    };

    // zp_idx_mult = 1 for per_dim1 zero points and 0, otherwise
    const int src_zp_idx_mult
            = !pd()->attr()->zero_points_.common(DNNL_ARG_SRC);
    const int dst_zp_idx_mult
            = !pd()->attr()->zero_points_.common(DNNL_ARG_DST);

    auto ker = [=](dim_t g, dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
        acc_data_t d = 0;
        for_(dim_t ic = 0; ic < IC; ++ic)
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            const dim_t id = od * KSD - padFront + kd * KDD;
            const dim_t ih = oh * KSH - padT + kh * KDH;
            const dim_t iw = ow * KSW - padL + kw * KDW;

            if (id < 0 || id >= ID) continue;
            if (ih < 0 || ih >= IH) continue;
            if (iw < 0 || iw >= IW) continue;

            const auto src_off
                    = get_data_off(src_d, ndims, mb, g * IC + ic, id, ih, iw);
            const auto wei_off = get_weights_off(
                    weights_d, with_groups, ndims, g, oc, ic, kd, kh, kw);

            acc_data_t s = static_cast<acc_data_t>(src[src_off]);
            if (src_zero_point)
                s -= static_cast<acc_data_t>(
                        src_zero_point[src_zp_idx_mult * (g * IC + ic)]);
            d += (acc_data_t)s * weights[wei_off];
        }
        return d;
    };

    // help compiler optimize the code
    // constants for plain layouts kernel
    const dnnl_dims_t &src_str = src_d.blocking_desc().strides;
    const dim_t src_ic_stride = src_str[1];
    const dim_t src_id_stride = (ndims == 5) ? src_str[2] : 0;
    const dim_t src_ih_stride = (ndims >= 4) ? src_str[ndims - 2] : 0;
    const dim_t src_iw_stride = (ndims >= 3) ? src_str[ndims - 1] : 0;
    const dnnl_dims_t &weights_str = weights_d.blocking_desc().strides;
    const int gr_shift = with_groups ? 1 : 0;
    const dim_t weights_ic_stride = weights_str[1 + gr_shift];
    const dim_t weights_kd_stride
            = (ndims == 5) ? weights_str[2 + gr_shift] : 0;
    const dim_t weights_kh_stride
            = (ndims >= 4) ? weights_str[ndims - 2 + gr_shift] : 0;
    const dim_t weights_kw_stride
            = (ndims >= 3) ? weights_str[ndims - 1 + gr_shift] : 0;

    auto ker_plain = [=](dim_t g, dim_t mb, dim_t oc, dim_t od, dim_t oh,
                             dim_t ow) {
        assert(3 <= ndims && ndims <= 5);
        acc_data_t d = 0;

        const dim_t src_loc_off
                = get_data_off(src_d, ndims, mb, g * IC, 0, 0, 0);
        const dim_t weights_loc_off = get_weights_off(
                weights_d, with_groups, ndims, g, oc, 0, 0, 0, 0);

        const src_data_t *__restrict src_loc = src + src_loc_off;
        const wei_data_t *__restrict weights_loc = weights + weights_loc_off;

        if (IC > KW) {
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                const dim_t id = od * KSD - padFront + kd * KDD;
                const dim_t ih = oh * KSH - padT + kh * KDH;
                const dim_t iw = ow * KSW - padL + kw * KDW;
                if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                        || iw >= IW)
                    continue;

                for (dim_t ic = 0; ic < IC; ++ic) {
                    const dim_t src_off = ic + id * src_id_stride
                            + ih * src_ih_stride + iw * src_iw_stride;
                    const dim_t weights_off = ic * weights_ic_stride
                            + kd * weights_kd_stride + kh * weights_kh_stride
                            + kw;
                    acc_data_t s = static_cast<acc_data_t>(src_loc[src_off]);
                    if (src_zero_point)
                        s -= static_cast<acc_data_t>(
                                src_zero_point[src_zp_idx_mult
                                        * (g * IC + ic)]);
                    d += (acc_data_t)s * weights_loc[weights_off];
                }
            }
        } else {
            for_(dim_t ic = 0; ic < IC; ++ic)
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                const dim_t id = od * KSD - padFront + kd * KDD;
                const dim_t ih = oh * KSH - padT + kh * KDH;
                const dim_t iw = ow * KSW - padL + kw * KDW;
                if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0
                        || iw >= IW)
                    continue;

                const dim_t src_off = ic + id * src_id_stride
                        + ih * src_ih_stride + iw * src_iw_stride;
                const dim_t weights_off = ic * weights_ic_stride
                        + kd * weights_kd_stride + kh * weights_kh_stride + kw;
                acc_data_t s = static_cast<acc_data_t>(src_loc[src_off]);
                if (src_zero_point)
                    s -= static_cast<acc_data_t>(
                            src_zero_point[src_zp_idx_mult * (g * IC + ic)]);
                d += (acc_data_t)s * weights_loc[weights_off];
            }
        }
        return d;
    };

    parallel_nd(G, MB, OC, OD, OH, OW,
            [&](dim_t g, dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
                float a = bias ? get_bias(bias, bias_d.off(g * OC + oc),
                                  pd()->desc()->bias_desc.data_type)
                               : 0;

                if (src_d.is_plain() && weights_d.is_plain()
                        && src_ic_stride == 1 && weights_kw_stride == 1)
                    a += ker_plain(g, mb, oc, od, oh, ow);
                else
                    a += ker(g, mb, oc, od, oh, ow);

                dim_t dst_off = get_data_off(
                        dst_d, ndims, mb, g * OC + oc, od, oh, ow);

                dim_t dst_l_off = (mb * OC * G + g * OC + oc) * OD * OH * OW
                        + od * OH * OW + oh * OW + ow;

                maybe_oscale(a, g, oc);

                ref_post_ops_t::args_t args;
                args.dst_val = dst[dst_off];
                args.ctx = &ctx;
                args.l_offset = dst_l_off;
                args.dst_md = pd()->dst_md();
                ref_post_ops->execute(a, args);

                if (dst_zero_point)
                    a += static_cast<acc_data_t>(
                            dst_zero_point[dst_zp_idx_mult * (g * OC + oc)]);

                if (is_int_conv)
                    dst[dst_off] = qz_a1b0<float, dst_data_t>()(a);
                else
                    dst[dst_off] = saturate<dst_data_t>(a);
            });
    return status::success;
}

template <data_type_t diff_src_type, data_type_t wei_type,
        data_type_t diff_dst_type, data_type_t acc_type>
status_t ref_convolution_bwd_data_t<diff_src_type, wei_type, diff_dst_type,
        acc_type>::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const bool with_groups = pd()->with_groups();

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OD = pd()->OD();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto ID = pd()->ID();
    const auto IH = pd()->IH();
    const auto IW = pd()->IW();

    const auto OC = pd()->OC() / G;
    const auto IC = pd()->IC() / G;
    const auto KD = pd()->KD();
    const auto KH = pd()->KH();
    const auto KW = pd()->KW();

    const auto KSD = pd()->KSD();
    const auto KSH = pd()->KSH();
    const auto KSW = pd()->KSW();

    const auto KDD = pd()->KDD() + 1;
    const auto KDH = pd()->KDH() + 1;
    const auto KDW = pd()->KDW() + 1;

    const auto padFront = pd()->padFront();
    const auto padT = pd()->padT();
    const auto padL = pd()->padL();

    const auto ndims = pd()->desc()->diff_src_desc.ndims;

    using namespace data_type;
    bool is_int_conv = utils::one_of(diff_dst_type, s32, s8, u8);

    auto maybe_oscale = [=](float &d, dim_t g, dim_t ic) {
        /* scale_idx_mult = 1 for per_oc scales and 0, otherwise */
        const int scale_idx_mult
                = pd()->attr()->output_scales_.mask_ == (1 << 1);
        const float *scales = pd()->attr()->output_scales_.scales_;
        d *= scales[(g * IC + ic) * scale_idx_mult];
    };

    auto ker = [=](dim_t g, dim_t mb, dim_t ic, dim_t id, dim_t ih, dim_t iw) {
        acc_data_t d = 0;
        for_(dim_t oc = 0; oc < OC; ++oc)
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            if (iw + padL < kw * KDW || ih + padT < kh * KDH
                    || id + padFront < kd * KDD)
                continue;
            dim_t ow = iw - kw * KDW + padL;
            dim_t oh = ih - kh * KDH + padT;
            dim_t od = id - kd * KDD + padFront;
            if (ow % KSW != 0 || oh % KSH != 0 || od % KSD != 0) continue;

            ow /= KSW;
            oh /= KSH;
            od /= KSD;

            if (od < OD && oh < OH && ow < OW) {
                const auto diff_dst_off = get_data_off(
                        diff_dst_d, ndims, mb, g * OC + oc, od, oh, ow);
                const auto weights_off = get_weights_off(
                        weights_d, with_groups, ndims, g, oc, ic, kd, kh, kw);

                d += (acc_data_t)diff_dst[diff_dst_off] * weights[weights_off];
            }
        }
        return d;
    };

    // help compiler optimize the code
    // constants for plain layouts kernel
    const dnnl_dims_t &diff_dst_str = diff_dst_d.blocking_desc().strides;
    const dim_t diff_dst_oc_stride = diff_dst_str[1];
    const dim_t diff_dst_ow_stride = diff_dst_str[ndims - 1];
    const dim_t diff_dst_oh_stride = (ndims >= 4) ? diff_dst_str[ndims - 2] : 0;
    const dim_t diff_dst_od_stride = (ndims >= 5) ? diff_dst_str[ndims - 3] : 0;

    const dnnl_dims_t &weights_str = weights_d.blocking_desc().strides;
    const int gr_shift = with_groups ? 1 : 0;
    const dim_t weights_oc_stride = weights_str[0 + gr_shift];
    const dim_t weights_kw_stride = weights_str[ndims - 1 + gr_shift];
    const dim_t weights_kh_stride
            = (ndims >= 4) ? weights_str[ndims - 2 + gr_shift] : 0;
    const dim_t weights_kd_stride
            = (ndims >= 4) ? weights_str[ndims - 3 + gr_shift] : 0;

    auto ker_plain = [=](dim_t g, dim_t mb, dim_t ic, dim_t id, dim_t ih,
                             dim_t iw) {
        assert(3 <= ndims && ndims <= 5);
        acc_data_t d = 0;
        const dim_t diff_dst_loc_off
                = get_data_off(diff_dst_d, ndims, mb, g * OC, 0, 0, 0);
        const dim_t weights_loc_off = get_weights_off(
                weights_d, with_groups, ndims, g, 0, ic, 0, 0, 0);

        const diff_dst_data_t *__restrict diff_dst_loc
                = diff_dst + diff_dst_loc_off;
        const wei_data_t *__restrict weights_loc = weights + weights_loc_off;

        if (OC > KW) {
            for_(dim_t kd = 0; kd < KD; ++kd)
            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                dim_t ow = iw - kw * KDW + padL;
                dim_t oh = ih - kh * KDH + padT;
                dim_t od = id - kd * KDD + padFront;
                if (ow < 0 || oh < 0 || od < 0 || ow % KSW != 0 || oh % KSH != 0
                        || od % KSD != 0)
                    continue;
                ow /= KSW;
                oh /= KSH;
                od /= KSD;
                if (od >= OD || oh >= OH || ow >= OW) continue;
                for (dim_t oc = 0; oc < OC; ++oc) {
                    const dim_t diff_dst_off = oc + od * diff_dst_od_stride
                            + oh * diff_dst_oh_stride + ow * diff_dst_ow_stride;
                    const dim_t weights_off = oc * weights_oc_stride
                            + kd * weights_kd_stride + kh * weights_kh_stride
                            + kw;
                    d += (acc_data_t)diff_dst_loc[diff_dst_off]
                            * weights_loc[weights_off];
                }
            }
        } else {
            for_(dim_t oc = 0; oc < OC; ++oc)
            for_(dim_t kd = 0; kd < KD; ++kd)
            for (dim_t kh = 0; kh < KH; ++kh) {
                // Note: placing these 2 params outside the `kw-loop` because
                // of a compiler-generated bug. Declaring 'od' as volatile
                // fixes a recurring seg-fault.
                const volatile dim_t od_ = id - kd * KDD + padFront;
                const dim_t weights_off_ = oc * weights_oc_stride
                        + kd * weights_kd_stride + kh * weights_kh_stride;
                for (dim_t kw = 0; kw < KW; ++kw) {
                    dim_t ow = iw - kw * KDW + padL;
                    dim_t oh = ih - kh * KDH + padT;
                    dim_t od = od_;
                    if (ow < 0 || oh < 0 || od < 0 || ow % KSW != 0
                            || oh % KSH != 0 || od % KSD != 0)
                        continue;
                    ow /= KSW;
                    oh /= KSH;
                    od /= KSD;
                    if (od >= OD || oh >= OH || ow >= OW) continue;
                    const dim_t diff_dst_off = oc + od * diff_dst_od_stride
                            + oh * diff_dst_oh_stride + ow * diff_dst_ow_stride;
                    const dim_t weights_off = weights_off_ + kw;
                    d += (acc_data_t)diff_dst_loc[diff_dst_off]
                            * weights_loc[weights_off];
                }
            }
        }
        return d;
    };

    parallel_nd(G, MB, IC, ID, IH, IW,
            [&](dim_t g, dim_t mb, dim_t ic, dim_t id, dim_t ih, dim_t iw) {
                auto ds_idx = get_data_off(
                        diff_src_d, ndims, mb, g * IC + ic, id, ih, iw);
                float a = bias ? get_bias(bias, bias_d.off(g * IC + ic),
                                  pd()->desc()->bias_desc.data_type)
                               : 0;

                if (diff_dst_d.is_plain() && weights_d.is_plain()
                        && diff_dst_oc_stride == 1 && weights_kw_stride == 1)
                    a += ker_plain(g, mb, ic, id, ih, iw);
                else
                    a += ker(g, mb, ic, id, ih, iw);
                maybe_oscale(a, g, ic);
                if (is_int_conv)
                    diff_src[ds_idx] = saturate_and_round<diff_src_data_t>(a);
                else
                    diff_src[ds_idx] = saturate<diff_src_data_t>(a);
            });
    return status::success;
}

template <data_type_t src_type, data_type_t diff_wei_type,
        data_type_t diff_dst_type, data_type_t acc_type>
status_t ref_convolution_bwd_weights_t<src_type, diff_wei_type, diff_dst_type,
        acc_type>::execute_backward_weights(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(diff_wei_data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(diff_wei_data_t *, DNNL_ARG_DIFF_BIAS);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    const bool with_groups = pd()->with_groups();

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OD = pd()->OD();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto ID = pd()->ID();
    const auto IH = pd()->IH();
    const auto IW = pd()->IW();

    const auto OC = pd()->OC() / G;
    const auto IC = pd()->IC() / G;
    const auto KD = pd()->KD();
    const auto KH = pd()->KH();
    const auto KW = pd()->KW();

    const auto KSD = pd()->KSD();
    const auto KSH = pd()->KSH();
    const auto KSW = pd()->KSW();

    const auto KDD = pd()->KDD() + 1;
    const auto KDH = pd()->KDH() + 1;
    const auto KDW = pd()->KDW() + 1;

    const auto padFront = pd()->padFront();
    const auto padT = pd()->padT();
    const auto padL = pd()->padL();

    const auto ndims = pd()->desc()->src_desc.ndims;

    using namespace data_type;
    bool is_int_conv = utils::one_of(src_type, s32, s8, u8);

    auto ker = [=](acc_data_t &d, dim_t g, dim_t oc, dim_t ic, dim_t kd,
                       dim_t kh, dim_t kw) {
        for_(dim_t mb = 0; mb < MB; ++mb)
        for_(dim_t od = 0; od < OD; ++od)
        for_(dim_t oh = 0; oh < OH; ++oh)
        for (dim_t ow = 0; ow < OW; ++ow) {
            if (ow * KSW + kw * KDW < padL || oh * KSH + kh * KDH < padT
                    || od * KSD + kd * KDD < padFront
                    || ow * KSW + kw * KDW >= IW + padL
                    || oh * KSH + kh * KDH >= IH + padT
                    || od * KSD + kd * KDD >= ID + padFront)
                continue;

            dim_t id = od * KSD - padFront + kd * KDD;
            dim_t ih = oh * KSH - padT + kh * KDH;
            dim_t iw = ow * KSW - padL + kw * KDW;

            const auto diff_dst_off = get_data_off(
                    diff_dst_d, ndims, mb, g * OC + oc, od, oh, ow);
            const auto src_off
                    = get_data_off(src_d, ndims, mb, g * IC + ic, id, ih, iw);

            d += (acc_data_t)diff_dst[diff_dst_off] * src[src_off];
        }
    };

    auto ker_plain = [=](acc_data_t &d, dim_t g, dim_t oc, dim_t ic, dim_t kd,
                             dim_t kh, dim_t kw) {
        assert(3 <= ndims && ndims <= 5);
        // help compiler optimize the code
        // constants for plain layouts kernel
        const dnnl_dims_t &diff_dst_str = diff_dst_d.blocking_desc().strides;
        const dim_t diff_dst_mb_stride = diff_dst_str[0];
        const dim_t diff_dst_ow_stride = diff_dst_str[ndims - 1];
        const dim_t diff_dst_oh_stride
                = (ndims >= 4) ? diff_dst_str[ndims - 2] : 0;
        const dim_t diff_dst_od_stride
                = (ndims >= 5) ? diff_dst_str[ndims - 3] : 0;
        const dnnl_dims_t &src_str = src_d.blocking_desc().strides;
        const dim_t src_mb_stride = src_str[0];
        const dim_t src_iw_stride = src_str[ndims - 1];
        const dim_t src_ih_stride = (ndims >= 4) ? src_str[ndims - 2] : 0;
        const dim_t src_id_stride = (ndims >= 5) ? src_str[ndims - 3] : 0;

        const dim_t diff_dst_loc_off
                = get_data_off(diff_dst_d, ndims, 0, g * OC + oc, 0, 0, 0);
        const dim_t src_loc_off
                = get_data_off(src_d, ndims, 0, g * IC + ic, 0, 0, 0);

        const diff_dst_data_t *__restrict diff_dst_loc
                = diff_dst + diff_dst_loc_off;
        const src_data_t *__restrict src_loc = src + src_loc_off;

        for_(dim_t mb = 0; mb < MB; ++mb)
        for_(dim_t od = 0; od < OD; ++od)
        for_(dim_t oh = 0; oh < OH; ++oh)
        for (dim_t ow = 0; ow < OW; ++ow) {
            const dim_t id = od * KSD - padFront + kd * KDD;
            const dim_t ih = oh * KSH - padT + kh * KDH;
            const dim_t iw = ow * KSW - padL + kw * KDW;
            if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                continue;
            const dim_t diff_dst_off = mb * diff_dst_mb_stride
                    + od * diff_dst_od_stride + oh * diff_dst_oh_stride
                    + ow * diff_dst_ow_stride;
            const dim_t src_off = mb * src_mb_stride + id * src_id_stride
                    + ih * src_ih_stride + iw * src_iw_stride;
            d += (acc_data_t)diff_dst_loc[diff_dst_off] * src_loc[src_off];
        }
    };

    auto ker_bias = [=](acc_data_t &d, dim_t g, dim_t oc) {
        for_(dim_t mb = 0; mb < MB; ++mb)
        for_(dim_t od = 0; od < OD; ++od)
        for_(dim_t oh = 0; oh < OH; ++oh)
        for (dim_t ow = 0; ow < OW; ++ow) {
            const auto diff_dst_off = get_data_off(
                    diff_dst_d, ndims, mb, g * OC + oc, od, oh, ow);
            d += (acc_data_t)diff_dst[diff_dst_off];
        }
    };

    parallel_nd(G, OC, [&](dim_t g, dim_t oc) {
        if (diff_bias) {
            // XXX: loss of precision when bias is a float...
            acc_data_t db = 0;
            ker_bias(db, g, oc);
            if (is_int_conv)
                diff_bias[diff_bias_d.off(g * OC + oc)]
                        = saturate_and_round<diff_wei_data_t>(db);
            else
                diff_bias[diff_bias_d.off(g * OC + oc)]
                        = saturate<diff_wei_data_t>(db);
        }

        for_(dim_t ic = 0; ic < IC; ++ic)
        for_(dim_t kd = 0; kd < KD; ++kd)
        for_(dim_t kh = 0; kh < KH; ++kh)
        for (dim_t kw = 0; kw < KW; ++kw) {
            acc_data_t dw = 0;
            if (diff_dst_d.is_plain() && src_d.is_plain())
                ker_plain(dw, g, oc, ic, kd, kh, kw);
            else
                ker(dw, g, oc, ic, kd, kh, kw);

            dim_t idx = get_weights_off(
                    diff_weights_d, with_groups, ndims, g, oc, ic, kd, kh, kw);
            if (is_int_conv)
                diff_weights[idx] = saturate_and_round<diff_wei_data_t>(dw);
            else
                diff_weights[idx] = saturate<diff_wei_data_t>(dw);
        }
    });
    return status::success;
}

using namespace data_type;

template struct ref_convolution_fwd_t<f32>;
template struct ref_convolution_fwd_t<bf16, bf16, bf16, f32>;
template struct ref_convolution_fwd_t<bf16, bf16, f32, f32>;

template struct ref_convolution_fwd_t<u8, s8, f32, s32>;
template struct ref_convolution_fwd_t<u8, s8, s32, s32>;
template struct ref_convolution_fwd_t<u8, s8, s8, s32>;
template struct ref_convolution_fwd_t<u8, s8, u8, s32>;
template struct ref_convolution_fwd_t<s8, s8, f32, s32>;
template struct ref_convolution_fwd_t<s8, s8, s32, s32>;
template struct ref_convolution_fwd_t<s8, s8, s8, s32>;
template struct ref_convolution_fwd_t<s8, s8, u8, s32>;

template struct ref_convolution_bwd_data_t<f32, f32, f32, f32>;
template struct ref_convolution_bwd_data_t<f32, bf16, bf16, f32>;
template struct ref_convolution_bwd_data_t<bf16, bf16, bf16, f32>;

template struct ref_convolution_bwd_data_t<f32, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<s32, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<s8, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<u8, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<f32, s8, s8, s32>;
template struct ref_convolution_bwd_data_t<s32, s8, s8, s32>;
template struct ref_convolution_bwd_data_t<s8, s8, s8, s32>;
template struct ref_convolution_bwd_data_t<u8, s8, s8, s32>;

template struct ref_convolution_bwd_weights_t<f32, f32, f32, f32>;
template struct ref_convolution_bwd_weights_t<bf16, bf16, bf16, f32>;
template struct ref_convolution_bwd_weights_t<bf16, f32, bf16, f32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
