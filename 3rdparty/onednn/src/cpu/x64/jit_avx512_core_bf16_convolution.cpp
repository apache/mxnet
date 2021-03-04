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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "common/bfloat16.hpp"
#include "cpu/x64/jit_avx512_core_bf16_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

void jit_avx512_core_bf16_convolution_fwd_t::prepare_padded_bias(
        const char *&bias, const memory_tracking::grantor_t &scratchpad) const {
    if (!pd()->wants_padded_bias()) return;

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;
    auto padded_bias = scratchpad.template get<char>(
            memory_tracking::names::key_conv_padded_bias);
    utils::array_copy(
            padded_bias, bias, bia_dt_size * pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + bia_dt_size * pd()->jcp_.oc_without_padding,
            0.f, bia_dt_size * (pd()->jcp_.oc - pd()->jcp_.oc_without_padding));
    bias = padded_bias;
}

void jit_avx512_core_bf16_convolution_fwd_t::execute_forward_1d(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    // TODO: experiment with g_blocking for perf fine tuning
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    dim_t work_amount = jcp.mb * nb_groups * oc_chunks * jcp.nb_ow;

    int nthr = jcp.aligned_threads ? jcp.aligned_threads : jcp.nthr;
    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();

        int n {0}, gg {0}, occ {0}, owb {0};

        if (jcp.loop_order == loop_cwgn) {
            int dummy {0};
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                    nb_groups, n, jcp.mb, dummy, 1);
        } else if (jcp.loop_order == loop_gncw) {
            int dummy {0};
            nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, dummy, 1);
        } else if (jcp.loop_order == loop_nhwcg) {
            nd_iterator_init(start, n, jcp.mb, owb, jcp.nb_ow, occ, oc_chunks,
                    gg, nb_groups);
        } else
            assert(!"unsupported loop order");

        while (start < end) {
            int ocb = occ * jcp.nb_oc_blocking;
            int g = gg * g_blocking;
            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::nwc;
            const int oc_idx = is_dst_layout_nxc
                    ? g * jcp.oc + ocb * jcp.oc_block
                    : g * jcp.nb_oc + ocb;
            auto dst_w
                    = dst + jcp.typesize_out * dst_d.blk_off(n, oc_idx, ow_s);
            auto bias_w = bias ? bias
                            + bia_dt_size * oc_idx
                                    * (is_dst_layout_nxc ? 1 : jcp.oc_block)
                               : nullptr;
            const bool is_src_layout_nxc = jcp.src_tag == format_tag::nwc;
            const int ic_idx = is_src_layout_nxc ? g * jcp.ic : g * jcp.nb_ic;
            auto src_w = src + src_d.blk_off(n, ic_idx, iw_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb);

            par_conv.load_work = this_block_size(ocb * jcp.oc_block, jcp.oc,
                    jcp.nb_oc_blocking * jcp.oc_block);
            par_conv.src = src_w;
            par_conv.dst = dst_w;
            par_conv.filt = wht_w;
            par_conv.bias = bias_w;
            par_conv.owb = owb;
            (*kernel_)(&par_conv);

            if (jcp.loop_order == loop_cwgn) {
                int dummy {0};
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == loop_gncw) {
                int dummy {0};
                nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, dummy, 1);
            } else if (jcp.loop_order == loop_nhwcg) {
                ++start;
                nd_iterator_step(n, jcp.mb, owb, jcp.nb_ow, occ, oc_chunks, gg,
                        nb_groups);
            } else
                assert(!"unsupported loop order");
        }
    });
}

void jit_avx512_core_bf16_convolution_fwd_t::execute_forward_2d(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    // TODO: experiment with g_blocking for perf fine tuning
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    dim_t work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh * jcp.nb_ow;

    int nthr = jcp.aligned_threads ? jcp.aligned_threads : jcp.nthr;
    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();

        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        int n {0}, gg {0}, occ {0}, oh_s {0}, owb {0};

        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                    nb_groups, n, jcp.mb, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_nhwcg)
            nd_iterator_init(start, n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow,
                    occ, oc_chunks, gg, nb_groups);
        else
            assert(!"unsupported loop order");

        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g = gg * g_blocking;
            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
            if (jcp.loop_order == loop_nhwcg) oh_e = oh_s + 1; //step instead

            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::nhwc;
            const int oc_idx = is_dst_layout_nxc
                    ? g * jcp.oc + ocb * jcp.oc_block
                    : g * jcp.nb_oc + ocb;
            auto dst_w = dst
                    + jcp.typesize_out * dst_d.blk_off(n, oc_idx, oh_s, ow_s);
            auto bias_w = bias ? bias
                            + bia_dt_size * oc_idx
                                    * (is_dst_layout_nxc ? 1 : jcp.oc_block)
                               : nullptr;
            const bool is_src_layout_nxc = jcp.src_tag == format_tag::nhwc;
            const int ic_idx = is_src_layout_nxc ? g * jcp.ic : g * jcp.nb_ic;
            auto src_w = src + src_d.blk_off(n, ic_idx, ih_s, iw_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0);

            for (int oj = oh_s, ij = ih_s; oj < oh_e;
                    ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = div_up(max(0, -ij), dilate_h);
                int i_b_overflow = div_up(
                        max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                        dilate_h);
                int kh_padding
                        = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow);
                auto aux_src = src_w + i_t_overflow * dilate_h * src_h_stride;
                auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                par_conv.load_work = utils::this_block_size(ocb * jcp.oc_block,
                        jcp.oc, jcp.nb_oc_blocking * jcp.oc_block);
                par_conv.src = aux_src;
                par_conv.dst = dst_w;
                par_conv.filt = aux_wht;
                par_conv.bias = bias_w;
                par_conv.kh_padding = kh_padding;
                par_conv.owb = owb;
                (*kernel_)(&par_conv);

                src_w += src_h_stride * jcp.stride_h;
                dst_w += jcp.typesize_out * dst_h_stride;
            }
            if (jcp.loop_order == loop_cwgn)
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_nhwcg) {
                ++start;
                nd_iterator_step(n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow, occ,
                        oc_chunks, gg, nb_groups);
            } else
                assert(!"unsupported loop order");
        }
    });
}

void jit_avx512_core_bf16_convolution_fwd_t::execute_forward_3d(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    // TODO: experiment with g_blocking for perf fine tuning
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    dim_t work_amount
            = jcp.mb * nb_groups * oc_chunks * jcp.od * jcp.oh * jcp.nb_ow;

    int nthr = jcp.aligned_threads ? jcp.aligned_threads : jcp.nthr;
    parallel(nthr, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        auto par_conv = jit_conv_call_s();

        size_t src_d_stride = src_d.blk_off(0, 0, 1);
        size_t src_h_stride = src_d.blk_off(0, 0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 0, 1);
        size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);

        int n {0}, gg {0}, occ {0}, od_s {0}, oh_s {0}, owb {0};

        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                    nb_groups, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow, od_s, jcp.od, oh_s, jcp.oh);
        else if (jcp.loop_order == loop_nhwcg)
            nd_iterator_init(start, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh, owb,
                    jcp.nb_ow, occ, oc_chunks, gg, nb_groups);
        else
            assert(!"unsupported loop order");

        while (start < end) {

            int ocb = occ * jcp.nb_oc_blocking;
            int g = gg * g_blocking;
            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
            if (jcp.loop_order == loop_nhwcg) oh_e = oh_s + 1; //step instead
            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            int id_s = -jcp.f_pad + od_s * jcp.stride_d;
            int dilate_d = jcp.dilate_d + 1;
            int d_t_overflow = div_up(max(0, -id_s), dilate_d);
            int d_b_overflow = div_up(
                    max(0, id_s - jcp.id + (jcp.kd - 1) * dilate_d + 1),
                    dilate_d);
            int kd_padding = nstl::max(0, jcp.kd - d_t_overflow - d_b_overflow);

            const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
            const int oc_idx = is_dst_layout_nxc
                    ? g * jcp.oc + ocb * jcp.oc_block
                    : g * jcp.nb_oc + ocb;
            auto dst_w = dst
                    + jcp.typesize_out
                            * dst_d.blk_off(n, oc_idx, od_s, oh_s, ow_s);
            auto bias_w = bias ? bias
                            + bia_dt_size * oc_idx
                                    * (is_dst_layout_nxc ? 1 : jcp.oc_block)
                               : nullptr;
            const bool is_src_layout_nxc = jcp.src_tag == format_tag::ndhwc;
            const int ic_idx = is_src_layout_nxc ? g * jcp.ic : g * jcp.nb_ic;
            auto src_w = src + src_d.blk_off(n, ic_idx, id_s, ih_s, iw_s)
                    + d_t_overflow * dilate_d * src_d_stride;
            auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0)
                    + d_t_overflow * wht_d_stride;

            for (int oj = oh_s, ij = ih_s; oj < oh_e;
                    ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = div_up(max(0, -ij), dilate_h);
                int i_b_overflow = div_up(
                        max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                        dilate_h);
                int kh_padding
                        = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow);
                auto aux_src = src_w + i_t_overflow * dilate_h * src_h_stride;
                auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                par_conv.load_work = utils::this_block_size(ocb * jcp.oc_block,
                        jcp.oc, jcp.nb_oc_blocking * jcp.oc_block);
                par_conv.src = aux_src;
                par_conv.dst = dst_w;
                par_conv.filt = aux_wht;
                par_conv.bias = bias_w;
                par_conv.kh_padding = kh_padding;
                par_conv.kd_padding = kd_padding;
                par_conv.owb = owb;
                (*kernel_)(&par_conv);

                src_w += src_h_stride * jcp.stride_h;
                dst_w += jcp.typesize_out * dst_h_stride;
            }
            if (jcp.loop_order == loop_cwgn)
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_nhwcg) {
                ++start;
                nd_iterator_step(n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh, owb,
                        jcp.nb_ow, occ, oc_chunks, gg, nb_groups);
            } else
                assert(!"unsupported loop order");
        }
    });
}

void jit_avx512_core_bf16_convolution_bwd_data_t ::execute_backward_data_3d(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        // TODO: experiment with g_blocking for perf fine tuning
        int g_blocking = 1;
        int nb_groups = jcp.ngroups / g_blocking;
        int work_amount = nb_groups * jcp.mb * ic_chunks * jcp.id * jcp.ih;
        balance211(work_amount, nthr, ithr, start, end);

        auto par_conv = jit_conv_call_s();

        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);

        bool is_fast_path_d = jcp.dilate_d == 0 && jcp.stride_d == 1;
        bool is_fast_path_h = jcp.dilate_h == 0 && jcp.stride_h == 1;

        int n {0}, gg {0}, icc {0}, id_s {0}, ih_s {0};
        if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, icc, ic_chunks, gg, nb_groups, n, jcp.mb,
                    id_s, jcp.id, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_gnc)
            nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc, ic_chunks,
                    id_s, jcp.id, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_nhwcg)
            nd_iterator_init(start, n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih, icc,
                    ic_chunks, gg, nb_groups);
        else
            assert(!"unsupported loop order");

        while (start < end) {
            int icb = icc * jcp.nb_ic_blocking;
            int g = gg * g_blocking;
            int work_rem = end - start;
            int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
            if (jcp.loop_order == loop_nhwcg) ih_e = ih_s + 1; //step instead

            int od_s = 0, kd_len = 0, kd_lo = 0;
            if (is_fast_path_d) {
                int d_t_overflow = max(0, jcp.kd - 1 - id_s - jcp.f_pad);
                int d_b_overflow
                        = max(0, jcp.kd - jcp.id + id_s - jcp.back_pad);
                kd_len = jcp.kd - d_t_overflow - d_b_overflow;
                kd_lo = d_b_overflow;
                od_s = id_s + jcp.f_pad - d_b_overflow;
            } else if (jcp.dilate_d != 0) { // stride == 1
                int dilate_d = jcp.dilate_d + 1;
                // Note: use div_up to account for "holes" in filter
                int d_t_overflow = div_up(
                        max(0, (jcp.kd - 1) * dilate_d - id_s - jcp.f_pad),
                        dilate_d);
                int d_b_overflow
                        = div_up(max(0,
                                         (jcp.kd - 1) * dilate_d + 1 - jcp.id
                                                 + id_s - jcp.back_pad),
                                dilate_d);
                kd_len = jcp.kd - d_t_overflow - d_b_overflow;
                kd_lo = d_b_overflow;
                od_s = id_s + jcp.f_pad - d_b_overflow * dilate_d;
            } else { // dilate == 0
                int d_t_overflow = max(
                        0, (jcp.kd - 1 - id_s - jcp.f_pad) / jcp.stride_d);
                int d_b_overflow = max(0,
                        (jcp.kd - jcp.id + id_s - jcp.back_pad) / jcp.stride_d);
                int overflow_kd_hi = jcp.kd - 1
                        - modulo(
                                jcp.id - 1 + jcp.back_pad - id_s, jcp.stride_d);
                int overflow_kd_lo = (id_s + jcp.f_pad) % jcp.stride_d;

                kd_len = (overflow_kd_hi - overflow_kd_lo) / jcp.stride_d + 1
                        - d_t_overflow - d_b_overflow;
                kd_lo = overflow_kd_lo + d_b_overflow * jcp.stride_d;
                od_s = (id_s + jcp.f_pad - kd_lo) / jcp.stride_d;
            }
            assert(kd_len >= 0);

            const bool is_dsrc_layout_nxc = jcp.src_tag == format_tag::ndhwc;
            const int ic_idx = is_dsrc_layout_nxc
                    ? g * jcp.ic + icb * jcp.ic_block
                    : g * jcp.nb_ic + icb;
            auto diff_src_w = diff_src
                    + jcp.typesize_out * diff_src_d.blk_off(n, ic_idx, id_s);
            const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
            const int oc_idx = is_ddst_layout_nxc ? g * jcp.oc : g * jcp.nb_oc;
            auto diff_dst_w = diff_dst + diff_dst_d.blk_off(n, oc_idx, od_s);
            auto wht_w = weights + wht_blk_off(weights_d, g, 0, icb, kd_lo);

            for (int ij = ih_s; ij < ih_e; ++ij) {
                int oj, kh_len, kh_lo;
                if (is_fast_path_h) { // dilate == 0 && stride == 1
                    int i_t_overflow = max(0, jcp.kh - 1 - ij - jcp.t_pad);
                    int i_b_overflow = max(0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                    kh_len = jcp.kh - i_t_overflow - i_b_overflow;
                    kh_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow;
                } else if (jcp.dilate_h != 0) { // stride == 1
                    int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    int i_t_overflow = div_up(
                            max(0, (jcp.kh - 1) * dilate_h - ij - jcp.t_pad),
                            dilate_h);
                    int i_b_overflow
                            = div_up(max(0,
                                             (jcp.kh - 1) * dilate_h + 1
                                                     - jcp.ih + ij - jcp.b_pad),
                                    dilate_h);
                    kh_len = jcp.kh - i_t_overflow - i_b_overflow;
                    kh_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                } else { // dilate == 0
                    int i_t_overflow = max(
                            0, (jcp.kh - 1 - ij - jcp.t_pad) / jcp.stride_h);
                    int i_b_overflow = max(0,
                            (jcp.kh - jcp.ih + ij - jcp.b_pad) / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1
                            - modulo(jcp.ih - 1 + jcp.b_pad - ij, jcp.stride_h);
                    int overflow_kh_lo = (ij + jcp.t_pad) % jcp.stride_h;

                    kh_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h
                            + 1 - i_t_overflow - i_b_overflow;
                    kh_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                    oj = (ij + jcp.t_pad - kh_lo) / jcp.stride_h;
                }
                assert(kh_len >= 0);

                par_conv.load_work = utils::this_block_size(icb * jcp.ic_block,
                        jcp.ic, jcp.nb_ic_blocking * jcp.ic_block);
                par_conv.src = diff_src_w
                        + jcp.typesize_out * ij * diff_src_h_stride;
                par_conv.dst = diff_dst_w + oj * diff_dst_h_stride;
                par_conv.filt = wht_w + kh_lo * wht_h_stride;
                par_conv.kh_padding = kh_len;
                par_conv.kd_padding = kd_len;

                (*kernel_)(&par_conv);
            }

            if (jcp.loop_order == loop_cgn)
                nd_iterator_jump(start, end, icc, ic_chunks, gg, nb_groups, n,
                        jcp.mb, id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gnc)
                nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_nhwcg) {
                ++start;
                nd_iterator_step(n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih, icc,
                        ic_chunks, gg, nb_groups);
            } else
                assert(!"unsupported loop order");
        }
    });
}

void jit_avx512_core_bf16_convolution_bwd_data_t ::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        // TODO: experiment with g_blocking for perf fine tuning
        int g_blocking = 1;
        int nb_groups = jcp.ngroups / g_blocking;
        int work_amount = nb_groups * jcp.mb * ic_chunks * jcp.ih * jcp.nb_iw;
        balance211(work_amount, nthr, ithr, start, end);

        auto par_conv = jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        bool is_fast_path = jcp.dilate_h == 0 && jcp.stride_h == 1;

        int n {0}, gg {0}, icc {0}, ih_s {0}, iwb {0};
        if (jcp.loop_order == loop_cwgn)
            nd_iterator_init(start, icc, ic_chunks, iwb, jcp.nb_iw, gg,
                    nb_groups, n, jcp.mb, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_gncw)
            nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc, ic_chunks,
                    iwb, jcp.nb_iw, ih_s, jcp.ih);
        else if (jcp.loop_order == loop_nhwcg)
            nd_iterator_init(start, n, jcp.mb, ih_s, jcp.ih, iwb, jcp.nb_iw,
                    icc, ic_chunks, gg, nb_groups);
        else
            assert(!"unsupported loop order");

        while (start < end) {
            int icb = icc * jcp.nb_ic_blocking;
            int g = gg * g_blocking;
            int work_rem = end - start;
            int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
            if (jcp.loop_order == loop_nhwcg) ih_e = ih_s + 1; //step instead
            int iw_s = iwb * jcp.iw_block;
            int ow_s = iw_s / jcp.stride_w;

            auto diff_src_w = diff_src;
            auto diff_dst_w = diff_dst;
            const bool is_ddst_layout_nxc = utils::one_of(
                    jcp.dst_tag, format_tag::nwc, format_tag::nhwc);
            const int oc_idx = is_ddst_layout_nxc ? g * jcp.oc : g * jcp.nb_oc;
            const bool is_dsrc_layout_nxc = utils::one_of(
                    jcp.src_tag, format_tag::nwc, format_tag::nhwc);
            const int ic_idx = is_dsrc_layout_nxc
                    ? g * jcp.ic + icb * jcp.ic_block
                    : g * jcp.nb_ic + icb;
            if (jcp.ndims == 3) {
                diff_src_w += jcp.typesize_out
                        * diff_src_d.blk_off(n, ic_idx, iw_s);
                diff_dst_w += diff_dst_d.blk_off(n, oc_idx, ow_s);
            } else {
                diff_src_w += jcp.typesize_out
                        * diff_src_d.blk_off(n, ic_idx, 0, iw_s);
                diff_dst_w += diff_dst_d.blk_off(n, oc_idx, 0, ow_s);
            }
            auto wht_w = weights + wht_blk_off(weights_d, g, 0, icb);

            for (int ij = ih_s; ij < ih_e; ++ij) {
                int oj, k_len, k_lo;
                if (is_fast_path) { // dilate == 0 && stride == 1
                    int i_t_overflow = max(0, jcp.kh - 1 - ij - jcp.t_pad);
                    int i_b_overflow = max(0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                    k_len = jcp.kh - i_t_overflow - i_b_overflow;
                    k_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow;
                } else if (jcp.dilate_h != 0) { // stride == 1
                    int dilate_h = jcp.dilate_h + 1;
                    // Note: use div_up to account for "holes" in filter
                    int i_t_overflow = div_up(
                            max(0, (jcp.kh - 1) * dilate_h - ij - jcp.t_pad),
                            dilate_h);
                    int i_b_overflow
                            = div_up(max(0,
                                             (jcp.kh - 1) * dilate_h + 1
                                                     - jcp.ih + ij - jcp.b_pad),
                                    dilate_h);
                    k_len = jcp.kh - i_t_overflow - i_b_overflow;
                    k_lo = i_b_overflow;
                    oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                } else { // dilate == 0
                    int i_t_overflow = max(
                            0, (jcp.kh - 1 - ij - jcp.t_pad) / jcp.stride_h);
                    int i_b_overflow = max(0,
                            (jcp.kh - jcp.ih + ij - jcp.b_pad) / jcp.stride_h);
                    int overflow_kh_hi = jcp.kh - 1
                            - modulo(jcp.ih - 1 + jcp.b_pad - ij, jcp.stride_h);
                    int overflow_kh_lo = (ij + jcp.t_pad) % jcp.stride_h;

                    k_len = (overflow_kh_hi - overflow_kh_lo) / jcp.stride_h + 1
                            - i_t_overflow - i_b_overflow;
                    k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                    oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                }
                assert(k_len >= 0);

                par_conv.load_work = utils::this_block_size(icb * jcp.ic_block,
                        jcp.ic, jcp.nb_ic_blocking * jcp.ic_block);
                par_conv.src = diff_src_w
                        + jcp.typesize_out * ij * diff_src_h_stride;
                par_conv.dst = diff_dst_w + oj * diff_dst_h_stride;
                par_conv.filt = wht_w + k_lo * wht_h_stride;
                par_conv.kh_padding = k_len;
                par_conv.iwb = iwb;

                (*kernel_)(&par_conv);
            }

            if (jcp.loop_order == loop_cwgn)
                nd_iterator_jump(start, end, icc, ic_chunks, iwb, jcp.nb_iw, gg,
                        nb_groups, n, jcp.mb, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, iwb, jcp.nb_iw, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_nhwcg) {
                ++start;
                nd_iterator_step(n, jcp.mb, ih_s, jcp.ih, iwb, jcp.nb_iw, icc,
                        ic_chunks, gg, nb_groups);
            } else
                assert(!"unsupported loop order");
        }
    });
}

status_t jit_avx512_core_bf16_convolution_bwd_weights_t ::init(
        engine_t *engine) {
    const auto &j = pd()->jcp_;

    nthr_ = j.nthr;
    nthr_mb_ = j.nthr_mb;
    nthr_g_ = j.nthr_g;
    nthr_oc_b_ = j.nthr_oc_b;
    nthr_ic_b_ = j.nthr_ic_b;

    CHECK(safe_ptr_assign(
            kernel_, new jit_avx512_core_bf16_conv_bwd_weights_kernel_f32(j)));
    CHECK(kernel_->create_kernel());

    if (j.transpose_src) {
        CHECK(safe_ptr_assign(trans_kernel_, create_trans_src(&j)));
        CHECK(trans_kernel_->create_kernel());
    }
    if (j.transpose_dst) {
        CHECK(safe_ptr_assign(trans_dst_kernel_, create_trans_dst(&j)));
        CHECK(trans_dst_kernel_->create_kernel());
    }
    if (nthr_mb_ > 1) {
        CHECK(safe_ptr_assign(
                acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
        CHECK(acc_ker_->create_kernel());
    }

    return status::success;
}

struct jit_avx512_core_bf16_convolution_bwd_weights_t ::thread_info_t {
    const src_data_t *src = nullptr;
    const diff_dst_data_t *diff_dst = nullptr;
    const void *diff_weights = nullptr;
    const void *diff_bias = nullptr;

    const memory_tracking::grantor_t scratchpad;

    src_data_t *tr_src = nullptr;
    diff_dst_data_t *tr_diff_dst = nullptr;
    simple_barrier::ctx_t *tr_src_bctx = nullptr;
    simple_barrier::ctx_t *tr_diff_dst_bctx = nullptr;

    float *wei_bia_reduction = nullptr;
    float *bia_reduction = nullptr;
    simple_barrier::ctx_t *wei_bia_reduction_bctx = nullptr;

    int ithr = 0;
    int ithr_ic_b = 0, ithr_oc_b = 0, ithr_g = 0, ithr_mb = 0;
    int ithr_but_oc = 0;
    int ithr_but_ic = 0;

    int img_start = 0, img_end = 0, img_work = 0;
    int g_start = 0, g_end = 0, g_work = 0;
    int oc_b_start = 0, oc_b_end = 0, oc_b_work = 0;
    int ic_b_start = 0, ic_b_end = 0, ic_b_work = 0;

    thread_info_t(const jit_avx512_core_bf16_convolution_bwd_weights_t *self,
            const exec_ctx_t &ctx, int ithr)
        : scratchpad(ctx.get_scratchpad_grantor()), ithr(ithr) {
        diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
        src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
        diff_weights = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_WEIGHTS);

        const auto &jcp = self->kernel_->jcp;
        diff_bias = self->pd()->with_bias()
                        && (jcp.oc_without_padding % jcp.oc_block != 0)
                        && self->pd()->jcp_.bia_dt == data_type::f32
                ? (void *)scratchpad.template get<float>(key_conv_padded_bias)
                : CTX_OUT_MEM(void *, DNNL_ARG_DIFF_BIAS);

        if (jcp.transpose_src) {
            tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);

            if (jcp.global_transpose)
                tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                        key_conv_tr_src_bctx);
        }
        if (jcp.transpose_dst) {
            tr_diff_dst = scratchpad.template get<diff_dst_data_t>(
                    key_conv_tr_diff_dst);

            if (jcp.global_transpose)
                tr_diff_dst_bctx
                        = scratchpad.template get<simple_barrier::ctx_t>(
                                key_conv_tr_diff_dst_bctx);
        }
        wei_bia_reduction
                = scratchpad.template get<float>(key_conv_wei_bia_reduction);
        bia_reduction = nullptr;
        if (jcp.with_bias) {
            const size_t wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block
                    * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
            const int num_wei_buffers = jcp.wei_dt == data_type::bf16
                    ? jcp.nthr_mb
                    : jcp.nthr_mb - 1;
            bia_reduction = wei_bia_reduction + wei_size * num_wei_buffers;
        }

        if (jcp.global_transpose)
            wei_bia_reduction_bctx
                    = scratchpad.template get<simple_barrier::ctx_t>(
                            key_conv_wei_bia_reduction_bctx);

        ithr_ic_b = ithr % self->nthr_ic_b_;
        ithr_oc_b = ithr / self->nthr_ic_b_ % self->nthr_oc_b_;
        ithr_g = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ % self->nthr_g_;
        ithr_mb = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ / self->nthr_g_;

        ithr_but_oc = (ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_ic_b_
                + ithr_ic_b;

        ithr_but_ic = (ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_oc_b_
                + ithr_oc_b;

        int work_amount = jcp.nthr_mb_work;
        /* reduction dimension */
        balance211(work_amount, self->nthr_mb_, ithr_mb, img_start, img_end);
        img_work = img_end - img_start;

        /* independent dimensions */
        balance211(jcp.ngroups, self->nthr_g_, ithr_g, g_start, g_end);
        g_work = g_end - g_start;

        balance211(
                jcp.nb_oc, self->nthr_oc_b_, ithr_oc_b, oc_b_start, oc_b_end);
        oc_b_work = oc_b_end - oc_b_start;

        balance211(
                jcp.nb_ic, self->nthr_ic_b_, ithr_ic_b, ic_b_start, ic_b_end);
        ic_b_work = ic_b_end - ic_b_start;
    }
};

size_t jit_avx512_core_bf16_convolution_bwd_weights_t::tr_src_buf_number(
        const thread_info_t *ti, int g, int ic) const {
    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    return jcp.global_transpose
            ? ti->ithr_mb * jcp.nb_ic * jcp.ngroups + g * jcp.nb_ic + ic
            : ti->ithr;
}
size_t jit_avx512_core_bf16_convolution_bwd_weights_t::tr_diff_dst_buf_number(
        const thread_info_t *ti, int g, int oc) const {
    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    return jcp.global_transpose
            ? ti->ithr_mb * jcp.nb_oc * jcp.ngroups + g * jcp.nb_oc + oc
            : ti->ithr;
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::trans_src(
        src_data_t *tr_src, const src_data_t *src, int row_count) const {
    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    const int pf_depth = 2;
    struct {
        const src_data_t *src;
        src_data_t *tr_src;
    } pf_circ_buf_src[pf_depth];

    assert(jcp.ic_block == 16 || jcp.is_1stconv);
    const int src_stride = jcp.iw * jcp.ic_block;
    const int tr_src_stride = jcp.tr_iw * jcp.ic_block;

    for (int iwork = 0; iwork < row_count + pf_depth - 1; iwork++) {
        pf_circ_buf_src[iwork % pf_depth] = {src, tr_src};

        if (iwork >= pf_depth - 1) {
            int old_idx = (iwork - pf_depth + 1) % pf_depth;
            auto ctx = jit_trans_src_t::ctx_t();
            ctx.src = pf_circ_buf_src[old_idx].src;
            ctx.tr_src = pf_circ_buf_src[old_idx].tr_src;
            ctx.src_prf = src;
            ctx.tr_src_prf = tr_src;
            (*trans_kernel_)(&ctx);
        }
        src += src_stride;
        tr_src += tr_src_stride;
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t::trans_src_nxc(
        src_data_t *tr_src, const src_data_t *src_base, int spatial_start,
        dim_t spatial_start_offset, int icb_start, dim_t chb_stride,
        int row_count) const {
    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    const int src_stride = jcp.iw * jcp.ngroups * jcp.ic;
    const int tr_src_stride = jcp.tr_iw * jcp.ic_block;

    int work_rest = row_count;
    int max_spatial_work = jcp.id * jcp.ih;
    int sp_work = nstl::min(work_rest, max_spatial_work - spatial_start);
    const src_data_t *src = src_base + spatial_start_offset;
    int icb = 0;
    const int ic_tail_work = jcp.ic_tail ? jcp.ic_tail : jcp.ic_block;
    while (work_rest > 0) {
        for (int iwork = 0; iwork < sp_work; iwork++) {
            auto ctx = jit_trans_src_t::ctx_t();
            ctx.src = src;
            ctx.tr_src = tr_src;
            assert(icb_start + icb < jcp.nb_ic);
            ctx.ch_work = (icb_start + icb + 1) == jcp.nb_ic ? ic_tail_work
                                                             : jcp.ic_block;
            ctx.src_prf = nullptr;
            ctx.tr_src_prf = nullptr;
            (*trans_kernel_)(&ctx);
            src += src_stride;
            tr_src += tr_src_stride;
        }
        work_rest -= sp_work;
        sp_work = nstl::min(work_rest, max_spatial_work);
        icb++;
        src = src_base + icb * chb_stride;
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::trans_dst(
        diff_dst_data_t *tr_diff_dst, const diff_dst_data_t *diff_dst,
        int row_count) const {

    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    const int pf_depth = 2;
    struct {
        const diff_dst_data_t *diff_dst;
        diff_dst_data_t *tr_diff_dst;
    } pf_circ_buf_dst[pf_depth];

    assert(jcp.ic_block == 16 || jcp.is_1stconv);
    const int diff_dst_stride = jcp.ow * jcp.oc_block;
    const int tr_diff_dst_stride = jcp.tr_ow * jcp.oc_block;

    for (int iwork = 0; iwork < row_count + pf_depth - 1; iwork++) {
        pf_circ_buf_dst[iwork % pf_depth]
                = {(diff_dst_data_t *)diff_dst, tr_diff_dst};

        if (iwork >= pf_depth - 1) {
            int old_idx = (iwork - pf_depth + 1) % pf_depth;
            auto ctx = jit_trans_dst_t::ctx_t();
            ctx.src = pf_circ_buf_dst[old_idx].diff_dst;
            ctx.tr_src = pf_circ_buf_dst[old_idx].tr_diff_dst;
            ctx.src_prf = diff_dst;
            ctx.tr_src_prf = tr_diff_dst;
            (*trans_dst_kernel_)(&ctx);
        }
        diff_dst += diff_dst_stride;
        tr_diff_dst += tr_diff_dst_stride;
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t::trans_dst_nxc(
        diff_dst_data_t *tr_diff_dst, const diff_dst_data_t *diff_dst_base,
        int spatial_start, dim_t spatial_start_offset, int ocb_start,
        dim_t chb_stride, int row_count) const {
    const jit_conv_conf_t &jcp = this->kernel_->jcp;
    const int diff_dst_stride = jcp.ow * jcp.ngroups * jcp.oc;
    const int tr_diff_dst_stride = jcp.tr_ow * jcp.oc_block;
    int work_rest = row_count;
    int max_spatial_work = jcp.od * jcp.oh;
    int sp_work = nstl::min(work_rest, max_spatial_work - spatial_start);
    const src_data_t *diff_dst = diff_dst_base + spatial_start_offset;
    int ocb = 0;
    const int oc_tail_work = jcp.oc_tail ? jcp.oc_tail : jcp.oc_block;
    while (work_rest > 0) {
        for (int iwork = 0; iwork < sp_work; iwork++) {
            auto ctx = jit_trans_dst_t::ctx_t();
            ctx.src = diff_dst;
            ctx.tr_src = tr_diff_dst;
            assert(ocb_start + ocb < jcp.nb_oc);
            ctx.ch_work = (ocb_start + ocb + 1) == jcp.nb_oc ? oc_tail_work
                                                             : jcp.oc_block;
            ctx.src_prf = nullptr;
            ctx.tr_src_prf = nullptr;
            (*trans_dst_kernel_)(&ctx);
            diff_dst += diff_dst_stride;
            tr_diff_dst += tr_diff_dst_stride;
        }
        work_rest -= sp_work;
        sp_work = nstl::min(work_rest, max_spatial_work);
        ocb++;
        diff_dst = diff_dst_base + ocb * chb_stride;
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::compute_diff_weights_2d(
        const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto &jcp = kernel_->jcp;
    const bool is_src_layout_nxc = jcp.src_tag == format_tag::nhwc;
    const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::nhwc;

    const int wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
    const int bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
    const int optimal_hblock = jcp.spatial_blk_size;

    float *diff_wei;
    if (diff_weights_d.data_type() == data_type::bf16)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    float *diff_bias = nullptr;
    if (jcp.with_bias) {
        if (jcp.bia_dt == data_type::bf16)
            diff_bias = ti->bia_reduction + (ti->ithr_mb) * bias_buf_size;
        else
            diff_bias = ti->ithr_mb == 0
                    ? (float *)ti->diff_bias
                    : ti->bia_reduction + (ti->ithr_mb - 1) * bias_buf_size;
    }

    auto tr_diff_dst_off = [&](int g, int oc, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        return tr_diff_dst_buf_number(ti, g, oc) * jcp.tr_diff_dst_buf_size
                + oj * tr_row_size;
    };

    int img {0}, oh_s {0};
    int start = ti->img_start;
    int end = ti->img_end;

    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);

    nd_iterator_init(start, img, jcp.mb, oh_s, jcp.oh);

    while (start < end) {
        auto p = jit_conv_call_s();
        int work_rem = end - start;
        const int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
        int ih_s = nstl::max(0, -jcp.t_pad + oh_s * jcp.stride_h);
        const int ih_e = nstl::min(
                jcp.ih, -jcp.t_pad + (oh_e - 1) * jcp.stride_h + ext_kh);

        auto tr_src_off = [&](int g, int ic, int ih_end, int ij) {
            const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
            // Aligned to buffer end to use guard elements
            return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size
                    + (jcp.ih - ih_end + ij) * tr_row_size;
        };

        if (jcp.global_transpose) {
            using simple_barrier::barrier;
            // TODO: try to call local transpositions just before jit kernel
            /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
            if (jcp.transpose_src) {
                int j {0};
                const int work_amount
                        = ti->g_work * ti->ic_b_work * (ih_e - ih_s);
                int tr_start {0}, tr_end {0};
                balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, tr_start,
                        tr_end);

                int g {0}, ic_b {0};
                nd_iterator_init(tr_start, g, ti->g_work, ic_b, ti->ic_b_work,
                        j, ih_e - ih_s);

                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
                while (tr_start < tr_end) {
                    int g_ = g + ti->g_start;
                    int ic_b_ = ic_b + ti->ic_b_start;
                    int j_s = j + ih_s;
                    int j_e = j_s + nstl::min(tr_end - tr_start, ih_e - j_s);
                    const int ic_off_idx = is_src_layout_nxc
                            ? g_ * jcp.ic + ic_b_ * jcp.ic_block
                            : g_ * jcp.nb_ic + ic_b_;
                    const src_data_t *src
                            = &ti->src[src_d.blk_off(img, ic_off_idx, j_s)];
                    src_data_t *tr_src
                            = &ti->tr_src[tr_src_off(g_, ic_b_, ih_e, j_s)];

                    if (is_src_layout_nxc)
                        trans_src_nxc(tr_src, src, 0, 0, ic_b_, 0, j_e - j_s);
                    else
                        trans_src(tr_src, src, j_e - j_s);

                    nd_iterator_jump(tr_start, tr_end, g, ti->g_work, ic_b,
                            ti->ic_b_work, j, ih_e - ih_s);
                }
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            }
            if (jcp.transpose_dst) {
                int j {0};
                const int work_amount
                        = ti->g_work * ti->oc_b_work * (oh_e - oh_s);
                int tr_start {0}, tr_end {0};
                balance211(work_amount, nthr_ic_b_, ti->ithr_ic_b, tr_start,
                        tr_end);

                int g {0}, oc_b {0};
                nd_iterator_init(tr_start, g, ti->g_work, oc_b, ti->oc_b_work,
                        j, oh_e - oh_s);

                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
                while (tr_start < tr_end) {
                    int g_ = g + ti->g_start;
                    int oc_b_ = oc_b + ti->oc_b_start;
                    int j_s = j + oh_s;
                    int j_e = j_s + nstl::min(tr_end - tr_start, oh_e - j_s);
                    const int oc_off_idx = is_ddst_layout_nxc
                            ? g_ * jcp.oc + oc_b_ * jcp.oc_block
                            : g_ * jcp.nb_oc + oc_b_;
                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(
                                    img, oc_off_idx, j_s)];
                    diff_dst_data_t *tr_diff_dst
                            = &ti->tr_diff_dst[tr_diff_dst_off(g_, oc_b_, j_s)];

                    if (is_ddst_layout_nxc)
                        trans_dst_nxc(tr_diff_dst, diff_dst, 0, 0, oc_b_, 0,
                                j_e - j_s);
                    else
                        trans_dst(tr_diff_dst, diff_dst, j_e - j_s);

                    nd_iterator_jump(tr_start, tr_end, g, ti->g_work, oc_b,
                            ti->oc_b_work, j, oh_e - oh_s);
                }
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            }
        }
        int height_block = jcp.global_transpose ? oh_e - oh_s : optimal_hblock;
        int ic_b_step
                = jcp.uses_permw_transposition ? jcp.nb_ic_blocking_max : 1;
        int icb_work = ti->ic_b_end - ti->ic_b_start;
        if (ic_b_step > 1 && icb_work > ic_b_step && icb_work < 2 * ic_b_step)
            ic_b_step = utils::div_up(icb_work, 2);

        for_(int ohb_s = oh_s; ohb_s < oh_e; ohb_s += height_block)
        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += ic_b_step) {
            const int ohb_e = nstl::min(ohb_s + height_block, oh_e);
            const int ihb_s = nstl::max(0, -jcp.t_pad + ohb_s * jcp.stride_h);
            const int ihb_e = nstl::min(
                    jcp.ih, -jcp.t_pad + (ohb_e - 1) * jcp.stride_h + ext_kh);
            assert(IMPLICATION(jcp.global_transpose,
                    oh_s == ohb_s && oh_e == ohb_e && ih_s == ihb_s
                            && ih_e == ihb_e));
            const int ic_off_idx = is_src_layout_nxc
                    ? g * jcp.ic + ic_b * jcp.ic_block
                    : g * jcp.nb_ic + ic_b;
            const int oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + oc_b * jcp.oc_block
                    : g * jcp.nb_oc + oc_b;
            const int ic_to_compute = this_block_size(
                    ic_b * jcp.ic_block, jcp.ic, ic_b_step * jcp.ic_block);
            const int oc_to_compute = this_block_size(
                    oc_b * jcp.oc_block, jcp.oc, jcp.oc_block);

            if (jcp.transpose_src) {
                if (!jcp.global_transpose) {
                    const src_data_t *src
                            = (src_data_t *)&ti->src[src_d.blk_off(
                                    img, ic_off_idx, ihb_s)];
                    src_data_t *tr_src
                            = &ti->tr_src[tr_src_off(g, ic_b, ihb_e, ihb_s)];
                    if (is_src_layout_nxc)
                        trans_src_nxc(
                                tr_src, src, 0, 0, ic_b, 0, ihb_e - ihb_s);
                    else
                        trans_src(tr_src, src, ihb_e - ihb_s);
                    p.src = tr_src;
                } else {
                    p.src = &ti->tr_src[tr_src_off(g, ic_b, ihb_e, ihb_s)];
                }
            } else {
                p.src = &ti->src[src_d.blk_off(img, ic_off_idx, ihb_s)];
            }

            if (jcp.transpose_dst) {
                if (!jcp.global_transpose) {
                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(
                                    img, oc_off_idx, ohb_s)];
                    diff_dst_data_t *tr_diff_dst
                            = &ti->tr_diff_dst[tr_diff_dst_off(0, 0, 0)];
                    if (is_ddst_layout_nxc)
                        trans_dst_nxc(tr_diff_dst, diff_dst, 0, 0, oc_b, 0,
                                ohb_e - ohb_s);
                    else
                        trans_dst(tr_diff_dst, diff_dst, ohb_e - ohb_s);
                    p.dst = tr_diff_dst;
                } else {
                    p.dst = &ti->tr_diff_dst[tr_diff_dst_off(g, oc_b, ohb_s)];
                }
            } else {
                p.dst = &ti->diff_dst[diff_dst_d.blk_off(
                        img, oc_off_idx, ohb_s)];
            }

            p.filt = diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = diff_bias + g * rnd_up(jcp.oc, jcp.oc_block)
                    + oc_b * jcp.oc_block;
            p.channel = (start == ti->img_start) && (ohb_s == oh_s);
            p.reduce_work = ic_to_compute;
            p.load_work = oc_to_compute;
            p.os_index_begin = ohb_s;
            p.os_index_end = ohb_e;
            p.flags = 0 | (ic_b == 0 ? FLAG_IC_FIRST : 0);
            assert(ohb_e <= jcp.oh);
            (*kernel_)(&p);
        }

        nd_iterator_jump(start, end, img, jcp.mb, oh_s, jcp.oh);
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::compute_diff_weights_3d(
        const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto &jcp = kernel_->jcp;
    const bool is_src_layout_nxc = jcp.src_tag == format_tag::ndhwc;
    const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
    const int wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
    const int bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
    const int optimal_dblock = jcp.spatial_blk_size;

    float *diff_wei;
    if (diff_weights_d.data_type() == data_type::bf16)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    float *diff_bias = nullptr;
    if (jcp.with_bias) {
        if (jcp.bia_dt == data_type::bf16)
            diff_bias = ti->bia_reduction + (ti->ithr_mb) * bias_buf_size;
        else
            diff_bias = ti->ithr_mb == 0
                    ? (float *)ti->diff_bias
                    : ti->bia_reduction + (ti->ithr_mb - 1) * bias_buf_size;
    }

    auto tr_diff_dst_off_3d = [&](int g, int oc, int od) {
        assert(IMPLICATION(is_ddst_layout_nxc, jcp.transpose_dst));
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        const size_t tr_3d_size = tr_row_size * jcp.oh;
        return tr_diff_dst_buf_number(ti, g, oc) * jcp.tr_diff_dst_buf_size
                + od * tr_3d_size;
    };
    int img {0}, od_s {0};
    int start = ti->img_start;
    int end = ti->img_end;

    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);

    nd_iterator_init(start, img, jcp.mb, od_s, jcp.od);
    while (start < end) {
        auto p = jit_conv_call_s();
        int work_rem = end - start;
        const int od_e = od_s + work_rem > jcp.od ? jcp.od : od_s + work_rem;
        int id_s = nstl::max(0, -jcp.f_pad + od_s * jcp.stride_d);
        const int id_e = nstl::min(
                jcp.id, -jcp.f_pad + (od_e - 1) * jcp.stride_d + ext_kd);

        auto tr_src_off_3d = [&](int g, int ic, int id_end, int id) {
            const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
            const size_t tr_3d_size = tr_row_size * jcp.ih;
            // Aligned to buffer end to use guard elements
            return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size
                    + (jcp.id - id_end + id) * tr_3d_size;
        };

        if (jcp.global_transpose) {
            using simple_barrier::barrier;
            // TODO: try to call local transpositions just before jit kernel
            /* tr_src[nb_ic][id][16][~iw~] <- src[nb_ic][id][iw][16] */
            if (jcp.transpose_src) {
                int d {0};

                const int work_amount
                        = ti->g_work * ti->ic_b_work * (id_e - id_s);

                int tr_start {0}, tr_end {0};
                balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, tr_start,
                        tr_end);

                int g {0}, ic_b {0};

                nd_iterator_init(tr_start, g, ti->g_work, ic_b, ti->ic_b_work,
                        d, id_e - id_s);

                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
                while (tr_start < tr_end) {
                    int g_ = g + ti->g_start;
                    int ic_b_ = ic_b + ti->ic_b_start;
                    int d_s = d + id_s;
                    int d_e = d_s + nstl::min(tr_end - tr_start, id_e - d_s);

                    const int ic_off_idx = is_src_layout_nxc
                            ? g_ * jcp.ic + ic_b_ * jcp.ic_block
                            : g_ * jcp.nb_ic + ic_b_;
                    const src_data_t *src
                            = &ti->src[src_d.blk_off(img, ic_off_idx, d_s)];
                    src_data_t *tr_src
                            = &ti->tr_src[tr_src_off_3d(g_, ic_b_, id_e, d_s)];

                    if (is_src_layout_nxc)
                        trans_src_nxc(tr_src, src, 0, 0, ic_b_, 0,
                                (d_e - d_s) * jcp.ih);
                    else
                        trans_src(tr_src, src, (d_e - d_s) * jcp.ih);

                    nd_iterator_jump(tr_start, tr_end, g, ti->g_work, ic_b,
                            ti->ic_b_work, d, id_e - id_s);
                }
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            }
            if (jcp.transpose_dst) {
                int d {0};

                const int work_amount
                        = ti->g_work * ti->oc_b_work * (od_e - od_s);

                int tr_start {0}, tr_end {0};
                balance211(work_amount, nthr_ic_b_, ti->ithr_ic_b, tr_start,
                        tr_end);

                int g {0}, oc_b {0};

                nd_iterator_init(tr_start, g, ti->g_work, oc_b, ti->oc_b_work,
                        d, od_e - od_s);

                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
                while (tr_start < tr_end) {
                    int g_ = g + ti->g_start;
                    int oc_b_ = oc_b + ti->oc_b_start;
                    int d_s = d + od_s;
                    int d_e = d_s + nstl::min(tr_end - tr_start, od_e - d_s);
                    const int oc_off_idx = is_ddst_layout_nxc
                            ? g_ * jcp.oc + oc_b_ * jcp.oc_block
                            : g_ * jcp.nb_oc + oc_b_;

                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(
                                    img, oc_off_idx, d_s)];
                    diff_dst_data_t *tr_diff_dst
                            = &ti->tr_diff_dst[tr_diff_dst_off_3d(
                                    g_, oc_b_, d_s)];

                    if (is_ddst_layout_nxc)
                        trans_dst_nxc(tr_diff_dst, diff_dst, 0, 0, oc_b_, 0,
                                (d_e - d_s) * jcp.oh);
                    else
                        trans_dst(tr_diff_dst, diff_dst, (d_e - d_s) * jcp.oh);

                    nd_iterator_jump(tr_start, tr_end, g, ti->g_work, oc_b,
                            ti->oc_b_work, d, od_e - od_s);
                }
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            }
        }

        int depth_block = jcp.global_transpose ? od_e - od_s : optimal_dblock;

        for_(int odb_s = od_s; odb_s < od_e; odb_s += depth_block)
        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
            const int odb_e = nstl::min(odb_s + depth_block, od_e);
            const int idb_s = nstl::max(0, -jcp.f_pad + odb_s * jcp.stride_d);
            const int idb_e = nstl::min(
                    jcp.id, -jcp.f_pad + (odb_e - 1) * jcp.stride_d + ext_kd);
            const int kdb_front_pad
                    = nstl::max(0, jcp.f_pad - odb_s * jcp.stride_d);
            // Assumes kd_back_pad = 0 when kernel is dilated
            const int kdb_back_pad = nstl::max(
                    0, odb_s * jcp.stride_d + jcp.kd - jcp.f_pad - jcp.id);
            const int kdb_pad_off = nstl::min(jcp.kd - 1, kdb_front_pad)
                    * jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block
                    * jcp.typesize_out;

            assert(IMPLICATION(jcp.global_transpose,
                    od_s == odb_s && od_e == odb_e && id_s == idb_s
                            && id_e == idb_e));
            const int ic_off_idx = is_src_layout_nxc
                    ? g * jcp.ic + ic_b * jcp.ic_block
                    : g * jcp.nb_ic + ic_b;
            const int oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + oc_b * jcp.oc_block
                    : g * jcp.nb_oc + oc_b;
            const int ic_to_compute = this_block_size(
                    ic_b * jcp.ic_block, jcp.ic, jcp.ic_block);
            const int oc_to_compute = this_block_size(
                    oc_b * jcp.oc_block, jcp.oc, jcp.oc_block);
            if (jcp.transpose_src) {
                if (!jcp.global_transpose) {
                    const src_data_t *src
                            = (src_data_t *)&ti->src[src_d.blk_off(
                                    img, ic_off_idx, idb_s)];
                    src_data_t *tr_src
                            = &ti->tr_src[tr_src_off_3d(g, ic_b, idb_e, idb_s)];
                    if (is_src_layout_nxc)
                        trans_src_nxc(tr_src, src, 0, 0, ic_b, 0,
                                (idb_e - idb_s) * jcp.ih);
                    else
                        trans_src(tr_src, src, (idb_e - idb_s) * jcp.ih);
                    p.src = tr_src;
                } else {
                    p.src = &ti->tr_src[tr_src_off_3d(g, ic_b, idb_e, idb_s)];
                }
            } else {
                p.src = &ti->src[src_d.blk_off(img, ic_off_idx, idb_s)];
            }

            if (jcp.transpose_dst) {
                if (!jcp.global_transpose) {
                    const diff_dst_data_t *diff_dst
                            = &ti->diff_dst[diff_dst_d.blk_off(
                                    img, oc_off_idx, odb_s)];
                    diff_dst_data_t *tr_diff_dst
                            = &ti->tr_diff_dst[tr_diff_dst_off_3d(0, 0, 0)];
                    if (is_ddst_layout_nxc)
                        trans_dst_nxc(tr_diff_dst, diff_dst, 0, 0, oc_b, 0,
                                (odb_e - odb_s) * jcp.oh);
                    else
                        trans_dst(tr_diff_dst, diff_dst,
                                (odb_e - odb_s) * jcp.oh);
                    p.dst = tr_diff_dst;
                } else {
                    p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(
                            g, oc_b, odb_s)];
                }
            } else {
                p.dst = &ti->diff_dst[diff_dst_d.blk_off(
                        img, oc_off_idx, odb_s)];
            }

            p.filt = diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = diff_bias + g * rnd_up(jcp.oc, jcp.oc_block)
                    + oc_b * jcp.oc_block;
            p.channel = (start == ti->img_start) && (odb_s == od_s);
            p.reduce_work = ic_to_compute;
            p.load_work = oc_to_compute;
            p.os_index_begin = odb_s;
            p.os_index_end = odb_e;
            p.kd_padding = jcp.kd - kdb_front_pad - kdb_back_pad;
            p.kd_offset = kdb_pad_off;
            p.flags = 0 | (ic_b == 0 ? FLAG_IC_FIRST : 0);
            assert(odb_e <= jcp.od);
            (*kernel_)(&p);
        }

        nd_iterator_jump(start, end, img, jcp.mb, od_s, jcp.od);
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::compute_diff_weights(
        const thread_info_t *ti) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto &jcp = kernel_->jcp;
    const bool is_src_layout_nxc = utils::one_of(
            jcp.src_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    const bool is_ddst_layout_nxc = utils::one_of(
            jcp.dst_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    const int wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
    const int bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;

    float *diff_wei;
    if (diff_weights_d.data_type() == data_type::bf16)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    float *diff_bias = nullptr;
    if (jcp.with_bias) {
        if (jcp.bia_dt == data_type::bf16)
            diff_bias = ti->bia_reduction + (ti->ithr_mb) * bias_buf_size;
        else
            diff_bias = ti->ithr_mb == 0
                    ? (float *)ti->diff_bias
                    : ti->bia_reduction + (ti->ithr_mb - 1) * bias_buf_size;
    }

    auto tr_src_off = [&](int g, int ic, int ij) {
        assert(IMPLICATION(is_src_layout_nxc, jcp.transpose_src));
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size
                + ij * tr_row_size;
    };

    auto tr_src_off_3d = [&](int g, int ic, int id, int ij) {
        assert(IMPLICATION(is_ddst_layout_nxc, jcp.transpose_dst));
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_3d_size = tr_row_size * jcp.ih;
        return tr_src_buf_number(ti, g, ic) * jcp.tr_src_buf_size
                + id * tr_3d_size + ij * tr_row_size;
    };

    auto tr_diff_dst_off = [&](int g, int oc, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        return tr_diff_dst_buf_number(ti, g, oc) * jcp.tr_diff_dst_buf_size
                + oj * tr_row_size;
    };

    auto tr_diff_dst_off_3d = [&](int g, int oc, int od, int oj) {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        const size_t tr_3d_size = tr_row_size * jcp.oh;
        return tr_diff_dst_buf_number(ti, g, oc) * jcp.tr_diff_dst_buf_size
                + od * tr_3d_size + oj * tr_row_size;
    };

    auto uker_trans = [&](int img, int g = 0, int ic_b = 0) {
        int j {0}, d {0};
        int my_work = jcp.ih * jcp.id;
        int ic;
        int icb_start = ic_b;
        if (jcp.global_transpose) {
            const int work_amount = is_src_layout_nxc
                    ? ti->ic_b_work * jcp.ih * jcp.id
                    : ti->g_work * ti->ic_b_work * jcp.ih * jcp.id;

            int start {0}, end {0};
            balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, start, end);
            my_work = end - start;

            if (is_src_layout_nxc) {
                if (jcp.ndims == 5)
                    nd_iterator_init(
                            start, ic_b, ti->ic_b_work, d, jcp.id, j, jcp.ih);
                else
                    nd_iterator_init(start, ic_b, ti->ic_b_work, j, jcp.ih);
            } else {
                if (jcp.ndims == 5)
                    nd_iterator_init(start, g, ti->g_work, ic_b, ti->ic_b_work,
                            d, jcp.id, j, jcp.ih);
                else
                    nd_iterator_init(start, g, ti->g_work, ic_b, ti->ic_b_work,
                            j, jcp.ih);
            }
            g += ti->g_start;
            ic_b += ti->ic_b_start;
            icb_start = ic_b;
            ic = is_src_layout_nxc ? g * jcp.ic + ic_b * jcp.ic_block
                                   : g * jcp.nb_ic + ic_b;
        } else {
            ic = is_src_layout_nxc ? g * jcp.ic + ic_b * jcp.ic_block
                                   : g * jcp.nb_ic + ic_b;
            g = 0;
            ic_b = 0;
        }
        const bool need_local_gwork = is_src_layout_nxc && jcp.global_transpose;
        const auto local_gwork = need_local_gwork ? ti->g_work : 1;

        for (int gg = g; gg < g + local_gwork; ++gg) {
            if (need_local_gwork) ic = gg * jcp.ic + ic_b * jcp.ic_block;
            src_data_t *tr_src = (jcp.ndims == 5)
                    ? &ti->tr_src[tr_src_off_3d(gg, ic_b, d, j)]
                    : &ti->tr_src[tr_src_off(gg, ic_b, j)];
            auto src_offset = is_src_layout_nxc
                    ? src_d.blk_off(img, ic)
                    : (jcp.ndims == 5 ? src_d.blk_off(img, ic, d, j)
                                      : src_d.blk_off(img, ic, j));
            src_data_t *src = (src_data_t *)&ti->src[src_offset];

            if (is_src_layout_nxc) {
                dim_t sp_start_offset = (jcp.ndims == 5)
                        ? src_d.blk_off(0, 0, d, j)
                        : src_d.blk_off(0, 0, j);
                dim_t ch_shift = src_d.blk_off(0, jcp.ic_block);
                int sp_start_idx = d * jcp.ih + j;
                trans_src_nxc(tr_src, src, sp_start_idx, sp_start_offset,
                        icb_start, ch_shift, my_work);
            } else
                trans_src(tr_src, src, my_work);
        }
    };

    auto diff_dst_trans = [&](int img, int g = 0, int oc_b = 0) {
        int j {0}, d {0};
        int my_work = jcp.oh * jcp.od;
        int oc;
        int ocb_start = oc_b;

        if (jcp.global_transpose) {
            const size_t work_amount = is_ddst_layout_nxc
                    ? ti->oc_b_work * jcp.oh * jcp.od
                    : ti->g_work * ti->oc_b_work * jcp.oh * jcp.od;

            size_t start {0}, end {0};
            balance211(work_amount, nthr_ic_b_, ti->ithr_ic_b, start, end);
            my_work = end - start;

            if (is_ddst_layout_nxc) {
                if (jcp.ndims == 5)
                    nd_iterator_init(
                            start, oc_b, ti->oc_b_work, d, jcp.od, j, jcp.oh);
                else
                    nd_iterator_init(start, oc_b, ti->oc_b_work, j, jcp.oh);
            } else {
                if (jcp.ndims == 5)
                    nd_iterator_init(start, g, ti->g_work, oc_b, ti->oc_b_work,
                            d, jcp.od, j, jcp.oh);
                else
                    nd_iterator_init(start, g, ti->g_work, oc_b, ti->oc_b_work,
                            j, jcp.oh);
            }
            g += ti->g_start;
            oc_b += ti->oc_b_start;
            ocb_start = oc_b;
            oc = is_ddst_layout_nxc ? g * jcp.oc + oc_b * jcp.oc_block
                                    : g * jcp.nb_oc + oc_b;
        } else {
            oc = is_ddst_layout_nxc ? g * jcp.oc + oc_b * jcp.oc_block
                                    : g * jcp.nb_oc + oc_b;
            g = 0;
            oc_b = 0;
        }
        const bool need_local_gwork
                = is_ddst_layout_nxc && jcp.global_transpose;
        const auto local_gwork = need_local_gwork ? ti->g_work : 1;

        for (int gg = g; gg < g + local_gwork; ++gg) {
            if (need_local_gwork) oc = gg * jcp.oc + oc_b * jcp.oc_block;
            diff_dst_data_t *tr_diff_dst = (jcp.ndims == 5)
                    ? &ti->tr_diff_dst[tr_diff_dst_off_3d(gg, oc_b, d, j)]
                    : &ti->tr_diff_dst[tr_diff_dst_off(gg, oc_b, j)];
            auto ddst_offset = is_ddst_layout_nxc
                    ? diff_dst_d.blk_off(img, oc)
                    : (jcp.ndims == 5 ? diff_dst_d.blk_off(img, oc, d, j)
                                      : diff_dst_d.blk_off(img, oc, j));
            const diff_dst_data_t *diff_dst = &ti->diff_dst[ddst_offset];

            if (is_ddst_layout_nxc) {
                dim_t sp_start_offset = (jcp.ndims == 5)
                        ? diff_dst_d.blk_off(0, 0, d, j)
                        : diff_dst_d.blk_off(0, 0, j);
                dim_t ch_shift = diff_dst_d.blk_off(0, jcp.oc_block);
                int sp_start_idx = d * jcp.oh + j;
                trans_dst_nxc(tr_diff_dst, diff_dst, sp_start_idx,
                        sp_start_offset, ocb_start, ch_shift, my_work);
            } else
                trans_dst(tr_diff_dst, diff_dst, my_work);
        }
    };

    for (int img = ti->img_start; img < ti->img_end; ++img) {
        auto p = jit_conv_call_s();
        if (jcp.global_transpose) {
            using simple_barrier::barrier;
            // TODO: try to call local transpositions just before jit kernel
            /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
            if (jcp.transpose_src) {
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
                uker_trans(img);
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            }
            if (jcp.transpose_dst) {
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
                diff_dst_trans(img);
                if (nthr_ic_b_ > 1)
                    barrier(&ti->tr_diff_dst_bctx[ti->ithr_but_ic], nthr_ic_b_);
            }
        }
        int ic_b_step
                = jcp.uses_permw_transposition ? jcp.nb_ic_blocking_max : 1;
        int icb_work = ti->ic_b_end - ti->ic_b_start;
        if (ic_b_step > 1 && icb_work > ic_b_step && icb_work < 2 * ic_b_step)
            ic_b_step = utils::div_up(icb_work, 2);
        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += ic_b_step) {
            const int ic_off_idx = is_src_layout_nxc
                    ? g * jcp.ic + ic_b * jcp.ic_block
                    : g * jcp.nb_ic + ic_b;
            const int oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + oc_b * jcp.oc_block
                    : g * jcp.nb_oc + oc_b;
            const int ic_to_compute = this_block_size(
                    ic_b * jcp.ic_block, jcp.ic, ic_b_step * jcp.ic_block);
            const int oc_to_compute = this_block_size(
                    oc_b * jcp.oc_block, jcp.oc, jcp.oc_block);
            if (jcp.transpose_src) {
                if (!jcp.global_transpose) {
                    uker_trans(img, g, ic_b);
                    if (jcp.ndims == 5) {
                        p.src = &ti->tr_src[tr_src_off_3d(g, ic_b, 0, 0)];
                    } else {
                        p.src = &ti->tr_src[tr_src_off(g, ic_b, 0)];
                    }
                } else {
                    if (jcp.ndims == 5) {
                        p.src = &ti->tr_src[tr_src_off_3d(g, ic_b, 0, 0)];
                    } else {
                        p.src = &ti->tr_src[tr_src_off(g, ic_b, 0)];
                    }
                }
            } else {
                p.src = &ti->src[src_d.blk_off(img, ic_off_idx)];
            }

            if (jcp.transpose_dst) {
                if (!jcp.global_transpose) {
                    diff_dst_trans(img, g, oc_b);
                    if (jcp.ndims == 5) {
                        p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(
                                0, 0, 0, 0)];
                    } else {
                        p.dst = &ti->tr_diff_dst[tr_diff_dst_off(0, 0, 0)];
                    }
                } else {
                    if (jcp.ndims == 5) {
                        p.dst = &ti->tr_diff_dst[tr_diff_dst_off_3d(
                                g, oc_b, 0, 0)];
                    } else {
                        p.dst = &ti->tr_diff_dst[tr_diff_dst_off(g, oc_b, 0)];
                    }
                }
            } else {
                p.dst = &ti->diff_dst[diff_dst_d.blk_off(img, oc_off_idx)];
            }

            p.filt = diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b);
            p.bias = diff_bias + g * rnd_up(jcp.oc, jcp.oc_block)
                    + oc_b * jcp.oc_block;
            p.channel = (img == ti->img_start);
            p.flags = 0 | (ic_b == 0 ? FLAG_IC_FIRST : 0);
            p.reduce_work = ic_to_compute;
            p.load_work = oc_to_compute;
            (*kernel_)(&p);
        }
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::
        reduce_and_convert_diff_weights_and_bias(
                const thread_info_t *ti) const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kh * jcp.kw * ((jcp.ndims == 5) ? jcp.kd : 1);

    const bool is_bf16_out = diff_weights_d.data_type() == data_type::bf16;
    const bool is_bf16_bias = jcp.with_bias && jcp.bia_dt == data_type::bf16;
    if (nthr_mb_ == 1) {
        if (is_bf16_out) {
            // reduction is not required, only conversion
            for_(int g = ti->g_start; g < ti->g_end; g++)
            for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; oc_b++) {
                const size_t acc_size = (size_t)ti->ic_b_work * jcp.kh * jcp.kw
                        * ((jcp.ndims == 5) ? jcp.kd : 1) * jcp.ic_block
                        * jcp.oc_block;
                const size_t off
                        = wht_blk_off(diff_weights_d, g, oc_b, ti->ic_b_start);
                cvt_float_to_bfloat16((bfloat16_t *)(ti->diff_weights) + off,
                        (ti->wei_bia_reduction + off), acc_size);
            }
        }

        if (is_bf16_bias && ti->ithr_ic_b == 0 && ti->ic_b_work > 0) {
            for (int g = ti->g_start; g < ti->g_end; g++) {
                int result_start_idx = g * jcp.oc_without_padding
                        + ti->oc_b_start * jcp.oc_block;
                int buffer_start_idx = g * rnd_up(jcp.oc, jcp.oc_block)
                        + ti->oc_b_start * jcp.oc_block;
                const size_t acc_size = nstl::min(jcp.oc_without_padding,
                                                ti->oc_b_end * jcp.oc_block)
                        - ti->oc_b_start * jcp.oc_block;
                bfloat16_t *diff_bias
                        = (bfloat16_t *)ti->diff_bias + result_start_idx;
                float *buffer = ti->bia_reduction + buffer_start_idx;
                cvt_float_to_bfloat16(diff_bias, buffer, acc_size);
            }
        }
        return;
    }

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    if (jcp.global_transpose)
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const int ic_b_kh_work
            = ti->ic_b_work * ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
    const int work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int start {0}, end {0};
    balance211(work, nthr_mb_, ti->ithr_mb, start, end);
    if (start == end) return;

    const int _start_nthr_mb = 1;
    for (int thr_mb = _start_nthr_mb; thr_mb < nthr_mb_; ++thr_mb) {
        int w = start;
        int sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_kh_start {0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const int g = ti->g_start + sub_g_start;
            const int oc_b = ti->oc_b_start + sub_oc_b_start;
            const int ic_b = ti->ic_b_start
                    + sub_ic_b_kh_start / ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
            const int kX
                    = sub_ic_b_kh_start % ((jcp.ndims == 5) ? jcp.kd : jcp.kh);

            const size_t acc_size = (size_t)jcp.kw * jcp.ic_block * jcp.oc_block
                    * ((jcp.ndims == 5) ? jcp.kh : 1)
                    * nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start);

            const size_t off = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kX);

            float *wei_reduced = is_bf16_out
                    ? ti->wei_bia_reduction + off
                    : (float *)(ti->diff_weights) + off;

            int thr_mb_buffer_idx = is_bf16_out ? thr_mb : thr_mb - 1;
            float *wei_to_reduce = ti->wei_bia_reduction
                    + thr_mb_buffer_idx * wei_size + off;

            if (is_bf16_out && thr_mb == nthr_mb_ - 1)
                // the last iteration for bfloat16 requires conversion and
                // store to diff_weights array
                add_floats_and_cvt_to_bfloat16(
                        (bfloat16_t *)(ti->diff_weights) + off, wei_reduced,
                        wei_to_reduce, acc_size);
            else
                acc_ker_->accumulate(wei_reduced, wei_to_reduce, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }
        if (jcp.with_bias && ti->ithr_ic_b == 0 && ti->ic_b_work > 0
                && ti->ithr_mb == 0 && ti->img_work > 0) {
            for (int g = ti->g_start; g < ti->g_end; g++) {
                float *bias_reduced = is_bf16_bias ? ti->bia_reduction
                                                   : (float *)(ti->diff_bias);
                int thr_mb_buffer_idx = is_bf16_bias ? thr_mb : thr_mb - 1;
                int bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
                float *bias_to_reduce
                        = ti->bia_reduction + thr_mb_buffer_idx * bias_buf_size;
                const size_t acc_size = nstl::min(jcp.oc_without_padding,
                                                ti->oc_b_end * jcp.oc_block)
                        - ti->oc_b_start * jcp.oc_block;
                int idx = g * rnd_up(jcp.oc, jcp.oc_block)
                        + ti->oc_b_start * jcp.oc_block;
                if (is_bf16_bias && thr_mb == nthr_mb_ - 1) {
                    // the last iteration for bfloat16 requires conversion and
                    // store to diff_weights array
                    int diff_bias_idx = g * jcp.oc_without_padding
                            + ti->oc_b_start * jcp.oc_block;
                    add_floats_and_cvt_to_bfloat16(
                            (bfloat16_t *)(ti->diff_bias) + diff_bias_idx,
                            &bias_reduced[idx], &bias_to_reduce[idx], acc_size);
                } else {
                    acc_ker_->accumulate(
                            &bias_reduced[idx], &bias_to_reduce[idx], acc_size);
                }
            }
        }
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t::prepare_scratchpad_data(
        const exec_ctx_t &ctx) const {
    auto scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = pd()->jcp_;

    if (jcp.transpose_src) {
        // XXX: See the comment about tr_iw and guarding elements in
        // jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf()
        auto tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
        // Zero out guard elements that cross a buffer boundary to prevent a
        // race condition due to buffer overflows from memory optimization where
        // buffers sharing padding
        for (size_t ithr = 1; ithr <= jcp.tr_src_buf_count; ++ithr) {
            src_data_t *ts = &tr_src[ithr * jcp.tr_src_buf_size];
            for (int i = 0; i < jcp.tr_src_num_guard_elems; ++i)
                ts[i] = 0;
        }

        if (jcp.global_transpose && jcp.nthr_oc_b > 1) {
            const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
            auto tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_src_bctx);
            for (int i = 0; i < tr_src_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_src_bctx[i]);
        }
    }
    if (jcp.global_transpose && jcp.transpose_dst) {
        if (jcp.nthr_ic_b > 1) {
            const int tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
            auto tr_diff_dst_bctx
                    = scratchpad.template get<simple_barrier::ctx_t>(
                            key_conv_tr_diff_dst_bctx);
            for (int i = 0; i < tr_diff_dst_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_diff_dst_bctx[i]);
        }
    }

    if (jcp.global_transpose
            && (nthr_mb_ > 1
                    || pd()->diff_weights_md(0)->data_type
                            == data_type::bf16)) {
        // TODO: don't use barrier for case
        // diff_weights_type == data_type::bf16 && nthr_mb_ == 1
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx));
    }
}

void jit_avx512_core_bf16_convolution_bwd_weights_t ::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    prepare_scratchpad_data(ctx);

    const auto &jcp = pd()->jcp_;
    parallel(nthr_, [&](const int ithr, const int nthr) {
        assert(nthr_ == nthr);
        assert(utils::one_of(pd()->ndims(), 3, 4, 5));

        thread_info_t thread_info(this, ctx, ithr);
        switch (jcp.harness) {
            case harness_2d_reduction:
                compute_diff_weights_2d(&thread_info);
                if (jcp.global_transpose)
                    reduce_and_convert_diff_weights_and_bias(&thread_info);
                break;
            case harness_3d_reduction:
                compute_diff_weights_3d(&thread_info);
                if (jcp.global_transpose)
                    reduce_and_convert_diff_weights_and_bias(&thread_info);
                break;
            case harness_compute_full_spatial:
            case harness_mb_reduction:
                compute_diff_weights(&thread_info);
                if (jcp.global_transpose)
                    reduce_and_convert_diff_weights_and_bias(&thread_info);
                break;
            default: assert(!"Invalid harness type");
        }
    });

    if (!jcp.global_transpose) {
        parallel(nthr_, [&](const int ithr, const int nthr) {
            assert(nthr_ == nthr);
            thread_info_t thread_info(this, ctx, ithr);
            reduce_and_convert_diff_weights_and_bias(&thread_info);
        });
    }

    if (pd()->with_bias() && (jcp.oc_without_padding % jcp.oc_block != 0)
            && jcp.bia_dt != data_type::bf16) {
        auto diff_bias = ctx.get_scratchpad_grantor().template get<const float>(
                key_conv_padded_bias);
        auto diff_bias_in = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);
        const int padded_stride = rnd_up(jcp.oc, jcp.oc_block);
        const int stride = jcp.oc_without_padding;
        for (int g = 0; g < jcp.ngroups; ++g) {
            utils::array_copy(diff_bias_in + g * stride,
                    diff_bias + g * padded_stride, stride);
        }
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
