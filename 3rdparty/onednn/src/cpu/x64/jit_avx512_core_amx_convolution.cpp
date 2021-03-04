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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_amx_convolution.hpp"

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

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_core_amx_convolution_fwd_t<src_type, wei_type,
        dst_type>::prepare_padded_bias(const char *&bias,
        const memory_tracking::grantor_t &scratchpad) const {
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

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_core_amx_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_reduced_lowering(const exec_ctx_t &ctx)
        const {

    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const auto &jcp = pd()->jcp_;
    assert(jcp.is_relo);
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    const float *oscales = pd()->attr()->output_scales_.scales_;

    auto inp_p_buffer = ctx.get_scratchpad_grantor().template get<src_data_t>(
            key_conv_amx_inp_buffer); // fix the template
    auto wei_buffer = ctx.get_scratchpad_grantor().template get<wei_data_t>(
            key_conv_amx_wei_buffer); // fix the template
    auto wsp = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_amx_wsp_buffer);
    auto tcfg = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_tilecfg);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int oh_chunks = utils::div_up(jcp.oh, jcp.oh_blk_size);
    int work_amount = jcp.mb * jcp.ngroups * oh_chunks * jcp.nb_ow * oc_chunks;

    // reorder weights from (g)Owhi16o to (g)OR16r16o4r, where r := whi
    auto p = jit_conv_call_s();
    p.src = weights;
    p.dst = wei_buffer;
    kernel_->copy_to_wbuffer()(&p);
    const wei_data_t *wei = wei_buffer;

    size_t oc_subblock_step
            = jcp.kh * jcp.kw * jcp.ic_block_int_np * jcp.oc_block;
    size_t wei_oc_shift = (size_t)jcp.nb_oc_blocking * jcp.nb_ic_int
            * rnd_up(oc_subblock_step, jcp.ic_block_int * jcp.oc_block);

    kernel_->tile_configure(tcfg);
    const bool is_1d = pd()->ndims() == 3;

    // TODO: implement 2D parallelization driver (g * spatial x oc) to increase
    // input data reuse and parallelize input data reorders
    parallel(0, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_s();
        amx_tile_configure(tcfg);

        int mb {0}, g {0}, ohc {0}, owb {0}, occ {0};
        // need "inner" oh blocks w.r.t. ow blocks to allow pbuffer reuse
        nd_iterator_init(start, mb, jcp.mb, g, jcp.ngroups, owb, jcp.nb_ow, ohc,
                oh_chunks, occ, oc_chunks);
        int last_copied_mb = -1;
        int last_copied_ohc = -1;
        int last_copied_owb = -1;
        int last_copied_g = -1;
        while (start < end) {
            src_data_t *inp_buffer = inp_p_buffer + ithr * jcp.inp_buffer_size;

            assert(IMPLICATION(
                    jcp.ngroups > 1, jcp.oc == jcp.oc_without_padding));
            int oc = g * jcp.oc + occ * jcp.nb_oc_blocking * jcp.oc_block;
            int ocb = jcp.is_nspc ? oc : oc / jcp.oc_block;
            auto bias_w = bias ? bias + (bias_d.blk_off(oc) * bia_dt_size)
                               : nullptr;

            int oh_s = ohc * jcp.oh_blk_size;
            int oh_e = nstl::min(jcp.oh, oh_s + jcp.oh_blk_size);
            bool is_inp_buffer_relevant = true && last_copied_mb == mb
                    && last_copied_ohc == ohc && last_copied_owb == owb
                    && last_copied_g == g;
            bool has_inp_buffer_overlap = true && last_copied_mb == mb
                    && last_copied_owb == owb && last_copied_g == g
                    && jcp.oh_blk_size == jcp.nb_oh_blocking;

            int oh_step = jcp.nb_oh_blocking * jcp.oh_per_tile;
            for (int oh = oh_s; oh < oh_e; oh += oh_step) {
                const int inp_buffer_h_step
                        = jcp.stride_h * jcp.ic_without_padding;
                assert(jcp.is_nspc);
                assert(jcp.stride_h <= jcp.kh);

                int ow = owb * jcp.ow_block;

                src_data_t *inp_buffer_oh
                        = inp_buffer + (size_t)oh * inp_buffer_h_step;

                if (!is_inp_buffer_relevant) {
                    // prepare padded input buffer
                    const int icb = g * jcp.ic;
                    size_t inp_offset = is_1d ? src_d.blk_off(mb, icb, 0)
                                              : src_d.blk_off(mb, icb, 0, 0);
                    const int iw_step = jcp.ngroups * jcp.ic_without_padding;
                    const src_data_t *psrc = src + inp_offset;
                    // calculate overlap...
                    const int ih_overlap = has_inp_buffer_overlap
                            * nstl::max(0, jcp.kh - oh_step * jcp.stride_h);
                    const int kh_eff = jcp.kh - ih_overlap;
                    // prepare padded input buffer
                    src_data_t *pdst = inp_buffer_oh
                            + ih_overlap * jcp.ic_without_padding;
                    for (int doh = 0; doh < oh_step; doh++) {
                        const int ih_s = (doh + oh) * jcp.stride_h - jcp.t_pad
                                + ih_overlap;
                        const int ih_e = ih_s + kh_eff;
                        const int ih = nstl::max(0, ih_s);
                        p.t_overflow = nstl::max(0, -ih_s);
                        p.b_overflow = nstl::min<int>(
                                kh_eff, nstl::max(0, ih_e - jcp.ih));
                        p.kh_padding = nstl::max<int>(
                                0, (kh_eff - p.t_overflow - p.b_overflow));
                        p.kh_offset = kh_eff;

                        const int iw_s = ow * jcp.stride_w - jcp.l_pad;
                        const int iw_e = iw_s + jcp.iwp;
                        const int iw = nstl::max(0, iw_s);
                        p.f_overflow = nstl::max(0, -iw_s);
                        p.back_overflow = nstl::max(0, iw_e - jcp.iw);
                        p.kw_padding = nstl::max<int>(
                                0, jcp.iwp - p.f_overflow - p.back_overflow);

                        p.src = psrc + (ih * jcp.iw + iw) * iw_step;
                        p.dst = pdst
                                + doh * jcp.iwp * jcp.kh
                                        * jcp.ic_without_padding;

                        kernel_->copy_to_pbuffer()(&p);
                    }
                }

                p.src = inp_buffer_oh;
                size_t dst_offset = is_1d ? dst_d.blk_off(mb, ocb, ow)
                                          : dst_d.blk_off(mb, ocb, oh, ow);
                p.dst = dst + dst_offset;
                p.filt = wei + (g * oc_chunks + occ) * wei_oc_shift;
                p.bias = bias_w;
                p.scales = &oscales[jcp.is_oc_scale * oc];

                p.acc_s32 = wsp + ithr * jcp.wsp_buffer_size;
                p.last_h = (oh + oh_step <= oh_e);
                p.owb = owb;
                p.oc_blocks = occ * jcp.nb_oc_blocking;

                (*kernel_)(&p);
            }
            last_copied_mb = mb;
            last_copied_ohc = ohc;
            last_copied_owb = owb;
            last_copied_g = g;
            ++start;
            // need "inner" oh blocks w.r.t. ow blocks to allow pbuffer reuse
            nd_iterator_step(mb, jcp.mb, g, jcp.ngroups, owb, jcp.nb_ow, ohc,
                    oh_chunks, occ, oc_chunks);
        }
    });
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_core_amx_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    //const memory_desc_wrapper weights_d(pd()->weights_md(0)); // unused
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;
    const size_t dst_dt_size
            = types::data_type_size(pd()->desc()->dst_desc.data_type);
    const size_t src_dt_size
            = types::data_type_size(pd()->desc()->src_desc.data_type);
    const size_t wei_dt_size
            = types::data_type_size(pd()->desc()->weights_desc.data_type);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    const float *oscales = pd()->attr()->output_scales_.scales_;

    // TODO: use block offset instead of hand-calculated one
    //size_t wei_oc_shift = wht_blk_off(weights_d, 0, 1);
    size_t wei_oc_shift = (size_t)jcp.nb_oc_blocking * jcp.nb_ic_int * jcp.kh
            * jcp.kw * jcp.ic_block_int_np * jcp.oc_block;

    auto inp_p_buffer = ctx.get_scratchpad_grantor().template get<src_data_t>(
            key_conv_amx_inp_buffer); // fix the template
    auto wsp = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_amx_wsp_buffer);
    auto tcfg = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_tilecfg);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int oh_chunks = utils::div_up(jcp.oh, jcp.oh_blk_size);
    int work_amount = jcp.mb * jcp.ngroups * oh_chunks * jcp.nb_ow * oc_chunks;

    kernel_->tile_configure(tcfg);
    const bool is_1d = pd()->ndims() == 3;

    // TODO: implement 2D parallelization driver (g * spatial x oc) to increase
    // input data reuse and parallelize input data reorders
    parallel(0, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_s();
        amx_tile_configure(tcfg);

        int mb {0}, g {0}, ohc {0}, owb {0}, occ {0};
        nd_iterator_init(start, mb, jcp.mb, g, jcp.ngroups, ohc, oh_chunks, owb,
                jcp.nb_ow, occ, oc_chunks);
        int last_copied_mb = -1;
        int last_copied_ohc = -1;
        int last_copied_owb = -1;
        int last_copied_g = -1;
        while (start < end) {
            src_data_t *inp_buffer = inp_p_buffer + ithr * jcp.inp_buffer_size;

            assert(IMPLICATION(
                    jcp.ngroups > 1, jcp.oc == jcp.oc_without_padding));
            int oc = g * jcp.oc + occ * jcp.nb_oc_blocking * jcp.oc_block;
            int ocb = jcp.is_nspc ? oc : oc / jcp.oc_block;
            auto bias_w = bias ? bias + (bias_d.blk_off(oc) * bia_dt_size)
                               : nullptr;

            int oh_s = ohc * jcp.oh_blk_size;
            int oh_e = nstl::min(jcp.oh, oh_s + jcp.oh_blk_size);
            bool is_inp_buffer_relevant = true && last_copied_mb == mb
                    && last_copied_ohc == ohc && last_copied_owb == owb
                    && last_copied_g == g;

            int oh_step = jcp.nb_oh_blocking * jcp.oh_per_tile;
            for (int oh = oh_s; oh < oh_e; oh += oh_step) {
                const int gen_stride_h = nstl::min(jcp.stride_h, jcp.kh);
                if (!is_inp_buffer_relevant) {
                    const int iw = nstl::max(
                            0, owb * jcp.ow_block * jcp.stride_w - jcp.l_pad);

                    assert(IMPLICATION(
                            jcp.ngroups > 1, jcp.ic == jcp.ic_without_padding));
                    const int icb = g * (jcp.is_nspc ? jcp.ic : jcp.nb_ic);

                    // generalized kh including dilation
                    // the current implementation of copy routine is not
                    // optimal for small jcp.oh_blk_size as it copies
                    // dilation rows to buffer
                    const int gen_kh = ((jcp.kh - 1) * (jcp.dilate_h + 1) + 1);
                    const bool continuous_copy = gen_kh >= jcp.stride_h;
                    int current_oh_block = nstl::min(oh_e - oh, oh_step);
                    int num_copy_calls = continuous_copy ? 1 : current_oh_block;
                    for (int ohi = 0; ohi < num_copy_calls; ohi++) {
                        int ih_copy_start
                                = (oh + ohi) * jcp.stride_h - jcp.t_pad;
                        int ih_copy_end = ih_copy_start + gen_kh;
                        if (continuous_copy) {
                            ih_copy_end
                                    += jcp.stride_h * (current_oh_block - 1);
                            if (oh > oh_s)
                                // it's non-first block, shift start to the end
                                // of previous block
                                // ih_copy_end_prev =
                                //     (oh - 1) * str_h - t_pad + kh
                                ih_copy_start += gen_kh - jcp.stride_h;
                        }
                        int ih_zero_top = nstl::max(0, -ih_copy_start);
                        int ih_zero_bottom = nstl::max(0, ih_copy_end - jcp.ih);
                        // how many real data rows to copy (including padding)
                        int rows_to_copy = ih_copy_end - ih_copy_start;
                        p.kh_padding = max(0, rows_to_copy);
                        p.t_overflow = ih_zero_top;
                        p.b_overflow = ih_zero_bottom;
                        p.owb = owb;
                        int ih = nstl::max(ih_copy_start, 0);
                        size_t inp_offset = is_1d
                                ? src_d.blk_off(mb, icb, iw)
                                : src_d.blk_off(mb, icb, ih, iw);
                        p.src = src + src_dt_size * inp_offset;
                        // inp_buffer has physical padding
                        int ih_buf = continuous_copy
                                ? ih_copy_start + jcp.t_pad
                                        - oh_s * jcp.stride_h
                                : gen_stride_h * (oh + ohi - oh_s);
                        p.dst = inp_buffer
                                + (size_t)ih_buf * jcp.iwp
                                        * jcp.ic_block_int_np;

                        kernel_->copy_to_pbuffer()(&p);
                    }
                }

                int ih_buf = gen_stride_h * (oh - oh_s);
                int ow = owb * jcp.ow_block;
                p.src = inp_buffer
                        + (size_t)ih_buf * jcp.iwp * jcp.ic_block_int_np;

                size_t dst_offset = is_1d ? dst_d.blk_off(mb, ocb, ow)
                                          : dst_d.blk_off(mb, ocb, oh, ow);
                p.dst = dst + dst_dt_size * dst_offset;
                p.filt = weights
                        + wei_dt_size * (g * oc_chunks + occ) * wei_oc_shift;
                p.bias = bias_w;
                p.scales = &oscales[jcp.is_oc_scale * oc];

                p.acc_s32 = wsp + ithr * jcp.wsp_buffer_size;
                p.last_h = (oh + oh_step <= oh_e);
                p.owb = owb;
                p.oc_blocks = occ * jcp.nb_oc_blocking;

                (*kernel_)(&p);
            }
            last_copied_mb = mb;
            last_copied_ohc = ohc;
            last_copied_owb = owb;
            last_copied_g = g;
            ++start;
            nd_iterator_step(mb, jcp.mb, g, jcp.ngroups, ohc, oh_chunks, owb,
                    jcp.nb_ow, occ, oc_chunks);
        }
    });
}

template struct jit_avx512_core_amx_convolution_fwd_t<data_type::s8,
        data_type::s8, data_type::u8>;
template struct jit_avx512_core_amx_convolution_fwd_t<data_type::u8,
        data_type::s8, data_type::u8>;
template struct jit_avx512_core_amx_convolution_fwd_t<data_type::s8,
        data_type::s8, data_type::s8>;
template struct jit_avx512_core_amx_convolution_fwd_t<data_type::u8,
        data_type::s8, data_type::s8>;
template struct jit_avx512_core_amx_convolution_fwd_t<data_type::s8,
        data_type::s8, data_type::s32>;
template struct jit_avx512_core_amx_convolution_fwd_t<data_type::u8,
        data_type::s8, data_type::s32>;
template struct jit_avx512_core_amx_convolution_fwd_t<data_type::s8,
        data_type::s8, data_type::f32>;
template struct jit_avx512_core_amx_convolution_fwd_t<data_type::u8,
        data_type::s8, data_type::f32>;
template struct jit_avx512_core_amx_convolution_fwd_t<data_type::bf16,
        data_type::bf16, data_type::bf16>;
template struct jit_avx512_core_amx_convolution_fwd_t<data_type::bf16,
        data_type::bf16, data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
