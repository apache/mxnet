/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include <atomic>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm_x8s8s32x_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace dnnl::impl::memory_tracking::names;

template <data_type_t src_type, data_type_t dst_type>
status_t _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_base = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst_base = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();

    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    assert(IMPLICATION(jcp.ow_block != jcp.ow, jcp.oh_block == 1));

    std::atomic<status_t> st(status::success);

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        status_t st_thr = execute_forward_thr(
                ithr, nthr, src_base, wei_base, bia_base, dst_base, scratchpad);

        if (st_thr != status::success) st = st_thr;
    });

    return st;
}

template <data_type_t src_type, data_type_t dst_type>
status_t
_gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::execute_forward_thr(
        const int ithr, const int nthr, const src_data_t *src_base,
        const wei_data_t *wei_base, const char *bia_base, dst_data_t *dst_base,
        const memory_tracking::grantor_t &scratchpad) const {
    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    const auto src_md = memory_desc_wrapper(pd()->src_md());
    const size_t src_mb_stride = src_md.blk_off(1);
    const size_t src_g_stride = src_md.blk_off(0, 1) * jcp.ic;

    const auto wei_md = memory_desc_wrapper(pd()->weights_md(0));
    const size_t wei_g_stride = pd()->with_groups() ? wei_md.blk_off(1) : 0;

    const auto dst_md = memory_desc_wrapper(pd()->dst_md());
    const size_t dst_mb_stride = dst_md.blk_off(1);
    const size_t dst_g_stride = dst_md.blk_off(0, 1) * jcp.oc;

    const float *scales = pd()->attr()->output_scales_.scales_;

    const auto &post_ops = pd()->attr()->post_ops_;
    const bool do_sum = post_ops.contain(primitive_kind::sum, 0);
    const float sum_scale = do_sum ? post_ops.entry_[0].sum.scale : 0;

    float nslope = 0;
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        const auto &e = post_ops.entry_[idx];
        if (e.is_relu(true, false)) {
            nslope = e.eltwise.alpha;
            break;
        }
    }

    uint8_t *__restrict col = scratchpad.get<uint8_t>(key_conv_gemm_col)
            + (ptrdiff_t)ithr * jcp.im2col_sz;
    src_data_t *__restrict imtr = scratchpad.get<src_data_t>(key_conv_gemm_imtr)
            + (ptrdiff_t)ithr * jcp.is * jcp.ic;
    acc_data_t *__restrict acc
            = scratchpad.get<acc_data_t>(key_conv_int_dat_in_acc_dt)
            + (ptrdiff_t)ithr * jcp.oh_block * jcp.ow_block * jcp.oc;

    const ptrdiff_t offset = (ptrdiff_t)jcp.ngroups * jcp.ks * jcp.ic * jcp.oc;
    const int32_t *_wei_comp = (const int32_t *)(wei_base + offset);

    int g {0}, n {0}, ohb {0}, owb {0};
    size_t start = 0, end = 0;

    const bool is_problem_3d = pd()->ndims() == 5;
    assert(IMPLICATION(is_problem_3d,
            jcp.oh_block == jcp.oh && jcp.ow_block == jcp.ow
                    && jcp.ic_block == jcp.ic));

    const int nb_oh = div_up(jcp.oh, jcp.oh_block);
    const int nb_ow = div_up(jcp.ow, jcp.ow_block);
    const size_t work_amount = (size_t)jcp.ngroups * jcp.mb * nb_oh * nb_ow;
    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ohb, nb_oh, owb, nb_ow);
    uint8_t shift = jcp.signed_input ? 128 : 0;
    parallel_nd(jcp.im2col_sz, [&](ptrdiff_t i) { col[i] = shift; });

    status_t st = status::success;

    for (size_t iwork = start; iwork < end; ++iwork) {
        int oh = ohb * jcp.oh_block;
        int ow = owb * jcp.ow_block;
        const src_data_t *__restrict src
                = src_base + n * src_mb_stride + g * src_g_stride;
        const wei_data_t *__restrict wei = wei_base + g * wei_g_stride;
        const int32_t *__restrict wei_comp = _wei_comp + g * jcp.oc;
        const int h_step = nstl::min(jcp.oh_block, jcp.oh - oh);
        const int w_step = nstl::min(jcp.ow_block, jcp.ow - ow);
        if (jcp.im2col_sz && is_problem_3d)
            jit_gemm_convolution_utils::transpose_dt<src_data_t>(
                    jcp, src, imtr);

        for (int od = 0; od < jcp.od; od++) {
            dst_data_t *__restrict dst = dst_base + n * dst_mb_stride
                    + g * dst_g_stride
                    + ((od * jcp.oh + oh) * jcp.ow + ow)
                            * pp_ker_->dst_os_stride_;
            if (jcp.im2col_sz) {
                if (is_problem_3d)
                    jit_gemm_convolution_utils::im2col_dt_3d<src_data_t,
                            uint8_t>(jcp, imtr, col, od);
                else
                    jit_gemm_convolution_utils::im2col_dt<src_data_t, uint8_t>(
                            jcp, src, imtr, col, oh, h_step, ow, w_step);
            }

            const dim_t M = jcp.oc;
            const dim_t K = jcp.ks * jcp.ic;
            const dim_t N = h_step * w_step;
            const dim_t LDA = M * jcp.ngroups;
            const dim_t LDB = jcp.im2col_sz ? N : K * jcp.ngroups;
            const char *BT = jcp.im2col_sz ? "T" : "N";
            const int8_t off_a = 0;
            const uint8_t off_b = 0;
            const int32_t off_c = 0;
            const float onef = 1.f, zerof = 0.f;
            const src_data_t *__restrict src_od
                    = src + od * jcp.oh * jcp.ow * jcp.ngroups * jcp.ic;
            st = gemm_s8x8s32("N", BT, jcp.signed_input ? "C" : "F", &M, &N, &K,
                    &onef, wei, &LDA, &off_a,
                    jcp.im2col_sz ? col : (uint8_t *)src_od, &LDB, &off_b,
                    &zerof, acc, &M, jcp.signed_input ? wei_comp : &off_c);

            if (st != status::success) return st;

            auto wei_adj_scale
                    = (wei_md.extra().flags & memory_extra_flags::scale_adjust)
                    ? wei_md.extra().scale_adjust
                    : 1.f;

            parallel(0, [&](int ithr, int nthr) {
                size_t start, end;
                balance211((size_t)N * jcp.oc, nthr, ithr, start, end);
                (*pp_ker_)(dst, acc, bia_base, scales, nslope, sum_scale,
                        1.f / wei_adj_scale, g, start, end);
            });
        }
        nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ohb, nb_oh, owb, nb_ow);
    }

    return st;
}

template <data_type_t dst_type>
status_t _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto diff_dst_base = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto diff_src_base = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    auto scratchpad = ctx.get_scratchpad_grantor();

    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    std::atomic<status_t> st(status::success);

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        status_t st_thr = execute_backward_data_thr(ithr, nthr, diff_dst_base,
                wei_base, bia_base, diff_src_base, scratchpad);

        if (st_thr != status::success) st = st_thr;
    });

    return st;
}

template <data_type_t dst_type>
status_t
_gemm_u8s8s32x_convolution_bwd_data_t<dst_type>::execute_backward_data_thr(
        const int ithr, const int nthr, const diff_dst_data_t *diff_dst_base,
        const wei_data_t *wei_base, const char *bia_base,
        diff_src_data_t *diff_src_base,
        const memory_tracking::grantor_t &scratchpad) const {
    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    const auto diff_dst_md = memory_desc_wrapper(pd()->diff_dst_md());
    const size_t diff_dst_mb_stride = diff_dst_md.blk_off(1);
    const size_t diff_dst_g_stride = diff_dst_md.blk_off(0, 1) * jcp.oc;

    const auto wei_md = memory_desc_wrapper(pd()->weights_md(0));
    const size_t wei_g_stride = pd()->with_groups() ? wei_md.blk_off(1) : 0;

    const auto diff_src_md = memory_desc_wrapper(pd()->diff_src_md());
    const size_t diff_src_mb_stride = diff_src_md.blk_off(1);
    const size_t diff_src_g_stride = diff_src_md.blk_off(0, 1) * jcp.ic;
    const size_t diff_src_os_stride
            = diff_src_md.blocking_desc().strides[pd()->ndims() - 1];

    /* scale_idx_mult = 1 for per_oc scales and 0, otherwise */
    const int scale_idx_mult = pd()->attr()->output_scales_.mask_ == (1 << 1);
    const float *__restrict scales = pd()->attr()->output_scales_.scales_;
    const size_t work_amount = jcp.ngroups * jcp.mb;

    acc_data_t *__restrict col = scratchpad.get<acc_data_t>(key_conv_gemm_col)
            + (ptrdiff_t)ithr * jcp.im2col_sz;
    acc_data_t *__restrict acc
            = scratchpad.get<acc_data_t>(key_conv_int_dat_in_acc_dt)
            + (ptrdiff_t)ithr * jcp.is * jcp.id * jcp.ic;

    int n {0}, g {0};
    size_t start = 0, end = 0;

    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups);

    for (size_t iwork = start; iwork < end; ++iwork) {
        const diff_dst_data_t *__restrict diff_dst = diff_dst_base
                + n * diff_dst_mb_stride + g * diff_dst_g_stride;
        const wei_data_t *__restrict wei = wei_base + g * wei_g_stride;
        diff_src_data_t *__restrict diff_src = diff_src_base
                + n * diff_src_mb_stride + g * diff_src_g_stride;

        const dim_t M = jcp.ks * jcp.ic;
        const dim_t N = jcp.os * jcp.od;
        const dim_t K = jcp.oc;
        const int8_t off_a = 0;
        const diff_dst_data_t off_b = 0;
        const int32_t off_c = 0;
        const float onef = 1.0, zerof = 0.0;
        const dim_t LD = K * jcp.ngroups;

        status_t st = gemm_s8x8s32("T", "N", "F", &M, &N, &K, &onef, wei, &LD,
                &off_a, diff_dst, &LD, &off_b, &zerof,
                jcp.im2col_sz ? col : acc, &M, &off_c);

        if (st != status::success) return st;

        if (jcp.im2col_sz)
            jit_gemm_convolution_utils::col2im_dt<int32_t>(jcp, col, acc);

        // TODO: the code below is not tested and broken anyway.
        parallel_nd(jcp.is * jcp.id, [&](int is) {
            diff_src_data_t *__restrict diff_src_loc
                    = diff_src + is * diff_src_os_stride;
            const acc_data_t *__restrict acc_loc = acc + is * jcp.ic;
            const float *__restrict scales_loc
                    = scales + g * jcp.ic * scale_idx_mult;
            for (int ic = 0; ic < jcp.ic; ic++) {
                acc_data_t d = acc_loc[ic];
                if (jcp.with_bias)
                    d += get_bias(bia_base, g * jcp.ic + ic,
                            pd()->desc()->bias_desc.data_type);
                d *= scales_loc[ic * scale_idx_mult];
                diff_src_loc[ic] = qz_a1b0<acc_data_t, diff_src_data_t>()(d);
            }
        });
        nd_iterator_step(n, jcp.mb, g, jcp.ngroups);
    }

    return status::success;
}

using namespace data_type;

template struct _gemm_x8s8s32x_convolution_fwd_t<u8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, u8>;

template struct _gemm_x8s8s32x_convolution_fwd_t<s8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, u8>;

template struct _gemm_u8s8s32x_convolution_bwd_data_t<f32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s8>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<u8>;
} // namespace cpu
} // namespace impl
} // namespace dnnl
