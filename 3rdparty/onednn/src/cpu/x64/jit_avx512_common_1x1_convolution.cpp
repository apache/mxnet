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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx512_common_1x1_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

#define data_blk_off(f, n, c, d, h, w) \
    ((ndims == 3) ? (f).blk_off(n, c, w) \
                  : ((ndims == 4) ? (f).blk_off(n, c, h, w) \
                                  : (f).blk_off(n, c, d, h, w)))
/* convolution forward */

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_1x1_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    auto weights_dw = CTX_IN_MEM(
            const wei_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
    auto bias_dw = CTX_IN_MEM(
            const dst_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);

    auto scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = kernel_->jcp;
    if (pd()->wants_padded_bias()) {
        auto padded_bias
                = scratchpad.template get<dst_data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;
    }

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, weights_dw, bias_dw,
                dst, scratchpad);
    });

    if (pd()->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad(ctx);
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_1x1_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_thr(const int ithr, const int nthr,
        const src_data_t *src, const wei_data_t *weights,
        const dst_data_t *bias, const wei_data_t *weights_dw,
        const dst_data_t *bias_dw, dst_data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper dw_weights_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS));
    const memory_desc_wrapper dw_bias_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS));

    const auto &jcp = kernel_->jcp;
    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<src_data_t>(key_conv_rtus_space)
            : nullptr;

    const int ndims = src_d.ndims();
    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();

    const int nb_oc = jcp.nb_load;
    const int nb_ic = jcp.nb_reduce;
    const int nb_ic_blocking = jcp.nb_reduce_blocking;

    // override some constants for fused dw_conv
    const int os_block = jcp.with_dw_conv ? jcp.ow : jcp.bcast_block;
    const int nb_bcast = jcp.with_dw_conv ? jcp.oh : jcp.nb_bcast;
    const int nb_bcast_blocking = jcp.with_dw_conv ? 1 : jcp.nb_bcast_blocking;
    const int nb_bcast_blocking_max
            = jcp.with_dw_conv ? 1 : jcp.nb_bcast_blocking_max;
    const int nb_load_blocking = jcp.nb_load_blocking;
    const int nb_load_blocking_max = jcp.with_dw_conv
            ? jcp.nb_load_blocking
            : jcp.nb_load_blocking_max;

    // Begin: declare Variables needed for dw conv.
    memory_tracking::grantor_t dw_scratchpad(
            scratchpad, memory_tracking::names::prefix_fusion);
    dst_data_t *pbuf;
    size_t row_offset;
    const int jcp_dw_kh = 3;
    const int nb_buffer = jcp.nb_load_blocking;
    std::vector<dst_data_t *> addrs;
    // End

    auto init_bcast = [&](int iwork, int bcast_end, int &n, int &g,
                              int &bcast_step, int &od, int &oh, int &ow,
                              int &id, int &ih, int &iw) {
        int osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, nb_bcast);
        bcast_step = step(
                nb_bcast_blocking, nb_bcast - osb, nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        od = os / (jcp.oh * jcp.ow);
        int os_2d = os % (jcp.oh * jcp.ow);
        oh = os_2d / jcp.ow;
        ow = os_2d % jcp.ow;

        id = od * stride_d;
        ih = oh * stride_h;
        iw = ow * stride_w;
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](int ocb, int ocb_end, int &load_step) {
        load_step = step(nb_load_blocking, ocb_end - ocb, nb_load_blocking_max);
        const auto max_oc = nstl::min(ocb_end * jcp.oc_block, jcp.oc);
        p.load_dim = this_block_size(
                ocb * jcp.oc_block, max_oc, load_step * jcp.oc_block);
    };

    auto init_reduce = [&](int icb) {
        const int nb_ic_blocking_step
                = nstl::min(icb + nb_ic_blocking, nb_ic) - icb;
        p.first_last_flag = 0 | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                | (icb + nb_ic_blocking_step >= nb_ic ? FLAG_REDUCE_LAST : 0);

        p.reduce_dim = this_block_size(
                icb * jcp.ic_block, jcp.ic, nb_ic_blocking_step * jcp.ic_block);
        rp.icb = p.reduce_dim;
    };

    auto ker_1x1 = [&](int ocb, int ocb_start, int icb, int n, int g, int od,
                           int oh, int ow, int id, int ih, int iw) {
        const bool is_dst_layout_nxc = utils::one_of(jcp.dst_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        const int oc_off_idx = is_dst_layout_nxc
                ? g * jcp.oc + ocb * jcp.oc_block
                : g * nb_oc + ocb;
        const size_t dst_off = data_blk_off(dst_d, n, oc_off_idx, od, oh, ow);

        p.output_data = jcp.with_dw_conv ? pbuf + (oh % jcp_dw_kh) * row_offset
                                         : &dst[dst_off];
        p.bias_data
                = &bias[oc_off_idx * (is_dst_layout_nxc ? 1 : jcp.oc_block)];

        p.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];
        const bool is_src_layout_nxc = utils::one_of(jcp.src_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        const int ic_off_idx = is_src_layout_nxc
                ? g * jcp.ic + icb * jcp.ic_block
                : g * nb_ic + icb;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_
                    + (is_src_layout_nxc ? ic_off_idx
                                         : jcp.is * ic_off_idx * jcp.ic_block);
            if (ocb == ocb_start) {
                rp.src = src + data_blk_off(src_d, n, ic_off_idx, id, ih, iw);
                (*rtus_driver_)(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src + data_blk_off(src_d, n, ic_off_idx, id, ih, iw);

        (*kernel_)(&p);
    };
    auto conv_1x1 = [&](int bcast_start, int bcast_end, int ocb_start,
                            int ocb_end) {
        if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;

        if (jcp.loop_order == loop_rlb) {
            for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, ocb_end, load_step);
                    int iwork = bcast_start;
                    while (iwork < bcast_end) {
                        int n {0}, g {0}, bcast_step {0}, od {0}, oh {0},
                                ow {0}, id {0}, ih {0}, iw {0};
                        init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh,
                                ow, id, ih, iw);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                        iwork += bcast_step;
                    }
                    ocb += load_step;
                }
            }
        } else if (jcp.loop_order == loop_lbr) {
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, ocb_end, load_step);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n {0}, g {0}, bcast_step {0}, od {0}, oh {0}, ow {0},
                            id {0}, ih {0}, iw {0};
                    init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow,
                            id, ih, iw);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                    }
                    iwork += bcast_step;
                }
                ocb += load_step;
            }
        } else if (jcp.loop_order == loop_rbl) {
            for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n {0}, g {0}, bcast_step {0}, od {0}, oh {0}, ow {0},
                            id {0}, ih {0}, iw {0};
                    init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow,
                            id, ih, iw);
                    int ocb = ocb_start;
                    while (ocb < ocb_end) {
                        int load_step;
                        init_load(ocb, ocb_end, load_step);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                        ocb += load_step;
                    }
                    iwork += bcast_step;
                }
            }
        } else if (jcp.loop_order == loop_blr) {
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n {0}, g {0}, bcast_step {0}, od {0}, oh {0}, ow {0},
                        id {0}, ih {0}, iw {0};
                init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow, id,
                        ih, iw);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, ocb_end, load_step);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                    }
                    ocb += load_step;
                }
                iwork += bcast_step;
            }
        } else {
            assert(!"unsupported loop order");
        }
    };

    auto ker_dw = [&](int n, int ocb_start, int load_step, int &dw_oh) {
        auto &jcp_dw = pd()->dw_conv_pd_->jcp_;
        int oh_1x1 = nstl::max(dw_oh * jcp_dw.stride_h - jcp_dw.t_pad, 0);

        for (int i = 0; i < jcp_dw.kh; ++i)
            addrs[i] = pbuf + ((oh_1x1++) % jcp_dw.kh) * row_offset;

        const auto ocb_end = ocb_start + load_step;
        const auto wch_stride
                = jcp_dw.iw * jcp_dw.nb_ch_blocking * jcp_dw.ch_block;
        const int dil_h = jcp_dw.dilate_h + 1;
        const int str_h = jcp_dw.stride_h;
        const int ch_num = jcp_dw.nb_ch_blocking;
        const int ow = 0;
        const int kw = 0;

        for (int ch = ocb_start; ch < ocb_end; ch += jcp_dw.nb_ch_blocking) {

            const int i_t_overflow
                    = nstl::max(0, (int)(jcp_dw.t_pad - dw_oh * str_h));
            const int i_b_overflow
                    = nstl::max(jcp_dw.ih,
                              (int)(dw_oh * str_h + (jcp_dw.kh - 1) * dil_h
                                      - jcp_dw.t_pad + 1))
                    - jcp_dw.ih;

            const int kh = div_up(i_t_overflow, dil_h);
            const int kh_padding = jcp_dw.kh - div_up(i_t_overflow, dil_h)
                    - div_up(i_b_overflow, dil_h);

            jit_conv_call_s par_conv_dw;

            par_conv_dw.src = addrs.data();
            par_conv_dw.dst = &dst[dst_d.blk_off(n, ch, dw_oh, ow)];

            par_conv_dw.filt
                    = &weights_dw[dw_weights_d.blk_off(ch, 0, 0, kh, kw)];
            if (bias)
                par_conv_dw.bias
                        = &bias_dw[dw_bias_d.blk_off(ch * jcp_dw.ch_block)];

            par_conv_dw.kh_padding = (size_t)nstl::max(0, kh_padding);

            par_conv_dw.ch_blocks = nstl::min(ch + ch_num, jcp_dw.nb_ch) - ch;

            (*kernel_dw_)(&par_conv_dw);

            for (int i = 0; i < jcp_dw.kh; ++i)
                addrs[i] += wch_stride;
        }
    };

    auto conv_dw = [&]() {
        // Set variables
        auto dw_conv_buffer
                = dw_scratchpad.get<dst_data_t>(key_fusion_inout_buffer);
        auto &jcp_dw = pd()->dw_conv_pd_->jcp_;

        const auto dw_conv_buffer_size_
                = (size_t)jcp_dw.kh * jcp.ow * nb_buffer * jcp.oc_block;
        pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;
        row_offset = dw_conv_buffer_size_ / jcp_dw.kh;
        addrs.resize(jcp_dw.kh);

        int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
        balance2D(nthr, ithr, jcp.mb * jcp.ngroups * jcp_dw.oh, bcast_start,
                bcast_end, nb_oc, ocb_start, ocb_end, jcp.load_grp_count);

        while (ocb_start < ocb_end) {
            int load_step;
            init_load(ocb_start, ocb_end, load_step);

            int oh_1x1 = 0;
            auto bcast_iter = bcast_start;
            while (bcast_iter < bcast_end) {
                int n {0}, g {0}, oh_dw {0};
                nd_iterator_init(bcast_iter, n, jcp.mb, g, jcp.ngroups, oh_dw,
                        jcp_dw.oh);
                if (oh_dw == 0) oh_1x1 = 0; // Reset over mb boundary
                const int oh_1x1_range = oh_dw * jcp_dw.stride_h - jcp_dw.t_pad;
                const int oh_1x1_begin = nstl::max(oh_1x1_range, 0);
                const int oh_1x1_end
                        = nstl::min(oh_1x1_range + jcp_dw.kh, jcp.oh);
                oh_1x1 = nstl::max(
                        oh_1x1_begin, oh_1x1); // Skip rows computed previously

                // dw_spatial to 1x1 spatial conversion. if jcp.oh != jcp_dw.oh
                const int bcast_start_1x1
                        = n * jcp.ngroups * jcp.oh + g * jcp.oh + oh_1x1;
                const int bcast_end_1x1 = bcast_start_1x1 - oh_1x1 + oh_1x1_end;

                conv_1x1(bcast_start_1x1, bcast_end_1x1, ocb_start,
                        ocb_start + load_step);
                oh_1x1 = oh_1x1_end;
                ker_dw(n, g * nb_oc + ocb_start, load_step, oh_dw);

                bcast_iter += nb_bcast_blocking;
            }
            ocb_start += load_step;
        }
    };

    if (jcp.with_dw_conv) {
        conv_dw();
    } else {

        const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
        int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
                ocb_start, ocb_end, jcp.load_grp_count);

        conv_1x1(bcast_start, bcast_end, ocb_start, ocb_end);
    }
}

template struct jit_avx512_common_1x1_convolution_fwd_t<data_type::f32>;
/* convolution backward wtr data */

template <data_type_t diff_dst_type, data_type_t wei_type,
        data_type_t diff_src_type>
void jit_avx512_common_1x1_convolution_bwd_data_t<diff_dst_type, wei_type,
        diff_src_type>::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const auto &jcp = kernel_->jcp;
    auto rtus_space = pd()->rtus_.reduce_src_
            ? ctx.get_scratchpad_grantor().template get<diff_src_data_t>(
                    key_conv_rtus_space)
            : nullptr;

    const int ndims = diff_src_d.ndims();

    assert(jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1);

    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    const int nb_ic = jcp.nb_load;
    const int nb_oc = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;
    const int nb_oc_blocking = jcp.nb_reduce_blocking;

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        auto p = jit_1x1_conv_call_s();
        auto rp = rtus_driver_t<avx512_common>::call_params_t();

        int bcast_start {0}, bcast_end {0}, icb_start {0}, icb_end {0};
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
                icb_start, icb_end, jcp.load_grp_count);

        bool reduce_outer
                = (jcp.loop_order == loop_rbl || jcp.loop_order == loop_rlb);
        int nboc_outer = reduce_outer ? nb_oc : 1;
        int ocb_outer_step = reduce_outer ? nb_oc_blocking : 1;

        int nboc_inner = reduce_outer ? 1 : nb_oc;
        int ocb_inner_step = reduce_outer ? 1 : nb_oc_blocking;
        const int max_ic = nstl::min(icb_end * jcp.ic_block, jcp.ic);

        for (int ocb_outer = 0; ocb_outer < nboc_outer;
                ocb_outer += ocb_outer_step) {
            size_t cur_ocb_outer
                    = nstl::min(ocb_outer + ocb_outer_step, nboc_outer)
                    - ocb_outer;

            int load_step = 0;
            for (int icb = icb_start; icb < icb_end; icb += load_step) {
                load_step = step(jcp.nb_load_blocking, jcp.nb_load - icb,
                        jcp.nb_load_blocking_max);

                p.load_dim = this_block_size(
                        icb * jcp.ic_block, max_ic, load_step * jcp.ic_block);
                rp.icb = p.load_dim;

                int bcast_step;
                for (int iwork = bcast_start; iwork < bcast_end;
                        iwork += bcast_step) {
                    int n {0}, g {0}, osb {0};
                    nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
                            jcp.nb_bcast);

                    bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                            jcp.nb_bcast_blocking_max);
                    bcast_step = nstl::min(bcast_step, bcast_end - iwork);

                    const int os = osb * os_block;
                    p.bcast_dim = this_block_size(
                            os, jcp.os, bcast_step * os_block);
                    rp.os = p.bcast_dim;

                    const int od = os / (jcp.oh * jcp.ow);
                    const int os_2d = os % (jcp.oh * jcp.ow);
                    const int oh = os_2d / jcp.ow;
                    const int ow = os_2d % jcp.ow;
                    const int id = od * stride_d;
                    const int ih = oh * stride_h;
                    const int iw = ow * stride_w;
                    rp.iw_start = iw;
                    const bool is_dsrc_layout_nxc
                            = utils::one_of(jcp.src_tag, format_tag::nwc,
                                    format_tag::nhwc, format_tag::ndhwc);
                    const int ic_off_idx = is_dsrc_layout_nxc
                            ? g * jcp.ic + icb * jcp.ic_block
                            : g * nb_ic + icb;
                    rp.src = diff_src
                            + data_blk_off(
                                    diff_src_d, n, ic_off_idx, id, ih, iw);
                    if (pd()->rtus_.reduce_src_) {
                        rp.ws = rtus_space
                                + ithr * pd()->rtus_.space_per_thread_;
                        p.output_data = rp.ws;
                    } else
                        p.output_data = rp.src;

                    for (int ocb_inner = 0; ocb_inner < nboc_inner;
                            ocb_inner += ocb_inner_step) {
                        int cur_ocb_inner
                                = nstl::min(ocb_inner + ocb_inner_step,
                                          nboc_inner)
                                - ocb_inner;

                        int ocb = reduce_outer ? ocb_outer : ocb_inner;
                        int nb_oc_blocking_step
                                = reduce_outer ? cur_ocb_outer : cur_ocb_inner;
                        const bool is_ddst_layout_nxc
                                = utils::one_of(jcp.dst_tag, format_tag::nwc,
                                        format_tag::nhwc, format_tag::ndhwc);
                        const int oc_off_idx = is_ddst_layout_nxc
                                ? g * jcp.oc + ocb * jcp.oc_block
                                : g * nb_oc + ocb;
                        size_t diff_dst_off = data_blk_off(
                                diff_dst_d, n, oc_off_idx, od, oh, ow);
                        p.bcast_data = &diff_dst[diff_dst_off];

                        p.load_data = &weights[pd()->with_groups()
                                        ? weights_d.blk_off(g, ocb, icb)
                                        : weights_d.blk_off(ocb, icb)];

                        p.first_last_flag = ocb == 0 ? FLAG_REDUCE_FIRST : 0;

                        p.reduce_dim = this_block_size(ocb * jcp.oc_block,
                                jcp.oc, nb_oc_blocking_step * jcp.oc_block);

                        (*kernel_)(&p);
                    }
                    if (pd()->rtus_.reduce_src_) (*rtus_driver_)(&rp);
                }
            }
        }
    });
}

template struct jit_avx512_common_1x1_convolution_bwd_data_t<data_type::f32>;

/* convolution backward wtr weights */

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

status_t jit_avx512_common_1x1_convolution_bwd_weights_t ::init(
        engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_,
            new jit_avx512_common_1x1_conv_kernel(pd()->jcp_, *pd()->attr())));
    CHECK(safe_ptr_assign(
            acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
    CHECK(safe_ptr_assign(reducer_bias_,
            new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_)));
    CHECK(kernel_->create_kernel());
    CHECK(acc_ker_->create_kernel());
    CHECK(reducer_bias_->create_kernel());

    const auto &jcp = kernel_->jcp;

    if (jcp.transpose_src) {
        auto tp = jit_transpose4x16_src_t();
        tp.src_pf0_distance = 4;
        tp.tr_src_pf0_distance = 0;
        tp.src_pf1 = true;
        tp.tr_src_pf1 = false;
        CHECK(safe_ptr_assign(
                trans_kernel_, new jit_transpose4x16_src(&jcp, &tp)));
        CHECK(trans_kernel_->create_kernel());
    }

    CHECK(init_rtus_driver<avx512_common>(this));
    return status::success;
}

void jit_avx512_common_1x1_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias_in = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_BIAS);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;

    const auto scratchpad = ctx.get_scratchpad_grantor();

    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<data_t>(key_conv_rtus_space)
            : nullptr;
    const bool is_bias_padded
            = pd()->with_bias() && jcp.oc_without_padding % jcp.oc_block != 0;

    data_t *diff_bias = is_bias_padded
            ? scratchpad.get<data_t>(key_conv_padded_bias)
            : diff_bias_in;
    auto wei_reduction = scratchpad.get<data_t>(key_conv_wei_reduction);

    /* prepare src transposition barriers */
    auto tr_src = scratchpad.get<data_t>(key_conv_tr_src);
    auto tr_src_bctx
            = scratchpad.get<simple_barrier::ctx_t>(key_conv_tr_src_bctx);
    if (jcp.transpose_src) {
        for (int i = 0; i < jcp.nthr; ++i)
            simple_barrier::ctx_init(&tr_src_bctx[i]);
    }

    const int ndims = src_d.ndims();
    const int wei_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block)
            * rnd_up(jcp.ic, jcp.ic_block);

    simple_barrier::ctx_t reduction_barrier;
    simple_barrier::ctx_init(&reduction_barrier);

    const auto reducer_bia_scratchpad
            = memory_tracking::grantor_t(scratchpad, prefix_reducer_bia);
    auto rb = this->reducer_bias_.get();
    rb->init(reducer_bia_scratchpad);

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int nb_ic = jcp.nb_bcast;
    const int nb_ic_blocking = jcp.nb_bcast_blocking;

    const int nb_oc = jcp.nb_load;
    const int nb_oc_blocking = jcp.nb_load_blocking;

    const int sp_nb = jcp.nb_reduce;
    const int mb_sp_work = jcp.mb * sp_nb;

    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    // TODO: use memory descriptor with the same fmt as src
    // (or use a macro :))
    auto tr_src_off = [&](int img, int icb, int is) {
        const size_t tr_chn_size = jcp.tr_is * jcp.ic_block;
        const size_t tr_img_size = tr_chn_size * nb_ic * jcp.ngroups;
        return img * tr_img_size + icb * tr_chn_size + is * jcp.ic_block;
    };

    const bool is_src_layout_nxc = utils::one_of(
            jcp.src_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    auto uker_trans = [&](int ithr_mb, int img, int sp_b_start, int sp_size,
                              int g_start, int g_work, int ic_b_start,
                              int ic_b_work, int ithr, int nthr,
                              int first_ic_b) {
        assert(!is_src_layout_nxc);
        const int work_amount = g_work * ic_b_work;

        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int g {0}, ic_b {0};
        nd_iterator_init(start, g, g_work, ic_b, ic_b_work);
        g += g_start;
        const int ic_b_tr = g * nb_ic + first_ic_b + ic_b;
        ic_b += ic_b_start;

        const int _ic = g * nb_ic + ic_b;

        const int is = sp_b_start * jcp.reduce_block;
        const int id = is / (jcp.ih * jcp.iw);
        const int is_2d = is % (jcp.ih * jcp.iw);
        const int ih = is_2d / jcp.iw;
        const int iw = is_2d % jcp.iw;

        const int src1_off = data_blk_off(src_d, img, _ic, id, ih, iw);
        data_t *src1 = (data_t *)&src[src1_off];
        data_t *tr_src1 = &tr_src[tr_src_off(ithr_mb, ic_b_tr, is)];

        assert(jcp.ic_block == 16);
        const int src_stride = jcp.is * jcp.ic_block;
        const int tr_src_stride = jcp.tr_is * jcp.ic_block;

        const int my_work = end - start;
        for (int iwork = 0; iwork < my_work; iwork++) {
            auto par_trans = jit_src_transpose_s();
            assert(sp_size % 4 == 0 || sp_size % 4 == jcp.is % 4);
            par_trans.size = sp_size;
            par_trans.src = src1;
            par_trans.tr_src = tr_src1;
            par_trans.src_prf = src1 + 64 * 16;
            par_trans.tr_src_prf = tr_src1 + 80 * 16;
            (*trans_kernel_)(&par_trans);

            src1 += src_stride;
            tr_src1 += tr_src_stride;
        }
    };

    const bool is_ddst_layout_nxc = utils::one_of(
            jcp.dst_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    auto ker = [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);
        const bool ready_for_async
                = utils::one_of(jcp.ver, ver_fma, ver_avx512_core);
        MAYBE_UNUSED(ready_for_async);
        assert(IMPLICATION(
                !ready_for_async && !dnnl_thr_syncable(), jcp.nthr_mb == 1));

        const int ithr_ic_b = ithr % jcp.nthr_ic_b;
        const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
        const int ithr_g = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
        const int ithr_mb = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b / jcp.nthr_g;

        const int ithr_but_oc
                = (ithr_mb * jcp.nthr_g + ithr_g) * jcp.nthr_ic_b + ithr_ic_b;

        /* reduction dimension */
        int mb_sp_b_start {0}, mb_sp_b_end {0};
        if (jcp.transpose_src && jcp.nthr_mb < jcp.mb / 2) {
            // it's preferable to parallelize by mb if possible
            int img_start {0}, img_end {0};
            balance211(jcp.mb, jcp.nthr_mb, ithr_mb, img_start, img_end);
            mb_sp_b_start = img_start * sp_nb;
            mb_sp_b_end = img_end * sp_nb;
        } else {
            balance211(mb_sp_work, jcp.nthr_mb, ithr_mb, mb_sp_b_start,
                    mb_sp_b_end);
        }

        /* independent dimensions */
        int g_start {0}, oc_b_start {0}, ic_b_start {0};
        int g_end {0}, oc_b_end {0}, ic_b_end {0};

        balance211(jcp.ngroups, jcp.nthr_g, ithr_g, g_start, g_end);
        balance211(jcp.nb_load, jcp.nthr_oc_b, ithr_oc_b, oc_b_start, oc_b_end);
        balance211(
                jcp.nb_bcast, jcp.nthr_ic_b, ithr_ic_b, ic_b_start, ic_b_end);

        const int g_work = g_end - g_start;
        const int oc_b_work = oc_b_end - oc_b_start;
        const int ic_b_work = ic_b_end - ic_b_start;
        const bool cache_aliasing
                = (jcp.ic * jcp.ngroups * sizeof(float)) % 1024 == 0;
        int reduce_step = jcp.nb_reduce_blocking;
        int reduce_step_max = jcp.nb_reduce_blocking_max;
        if (is_src_layout_nxc && cache_aliasing) {
            // Experiments show 4 is a magic number with the tested shapes.
            // TODO: maybe tune for shapes with sp_dim%4 != 0
            reduce_step = nstl::min(4, reduce_step);
            reduce_step_max = reduce_step;
        }

        data_t *diff_wei = ithr_mb == 0
                ? diff_weights
                : wei_reduction + (ithr_mb - 1) * wei_size;

        int sp_b_step = 0;
        for (int mb_sp_b = mb_sp_b_start; mb_sp_b < mb_sp_b_end;
                mb_sp_b += sp_b_step) {
            int img {0}, sp_b {0};
            nd_iterator_init(mb_sp_b, img, jcp.mb, sp_b, sp_nb);
            sp_b_step = step(reduce_step,
                    nstl::min(sp_nb - sp_b, mb_sp_b_end - mb_sp_b),
                    reduce_step_max);

            for (int g = g_start; g < g_end; ++g) {
                int load_step = 0;
                int bcast_step = 0;
                for (int ic_b = ic_b_start; ic_b < ic_b_end;
                        ic_b += bcast_step) {
                    if (is_src_layout_nxc && cache_aliasing) {
                        bcast_step = ic_b_work;
                    } else {
                        bcast_step = step(nb_ic_blocking, ic_b_end - ic_b,
                                jcp.nb_bcast_blocking_max);
                    }
                    if (jcp.transpose_src) {
                        if (jcp.nthr_oc_b > 1)
                            simple_barrier::barrier(
                                    &tr_src_bctx[ithr_but_oc], jcp.nthr_oc_b);
                        const int sp_size
                                = nstl::min(sp_b_step * jcp.reduce_block,
                                        jcp.is - sp_b * jcp.reduce_block);
                        uker_trans(ithr_mb, img, sp_b, sp_size, g, 1, ic_b,
                                bcast_step, ithr_oc_b, jcp.nthr_oc_b,
                                ic_b_start);
                        if (jcp.nthr_oc_b > 1)
                            simple_barrier::barrier(
                                    &tr_src_bctx[ithr_but_oc], jcp.nthr_oc_b);
                    }

                    for (int oc_b = oc_b_start; oc_b < oc_b_end;
                            oc_b += load_step) {
                        load_step = step(nb_oc_blocking, oc_b_end - oc_b,
                                jcp.nb_load_blocking_max);
                        const int _ic_b = g * nb_ic + ic_b;
                        const int _ic_b_tr = g * nb_ic + ic_b_start;
                        const int oc_off_idx = is_ddst_layout_nxc
                                ? g * jcp.oc + oc_b * jcp.oc_block
                                : g * nb_oc + oc_b;

                        data_t *store_to;

                        const size_t off
                                = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                        store_to = diff_wei + off;

                        const int ic_off_idx
                                = (is_src_layout_nxc ? jcp.ic_block : 1)
                                * _ic_b;
                        const data_t *diff_src = jcp.transpose_src
                                ? &tr_src[tr_src_off(ithr_mb, _ic_b_tr, 0)]
                                : &src[src_d.blk_off(img, ic_off_idx)];

                        int sp_b_end = sp_b + sp_b_step;
                        const data_t *pdiff_dst = &diff_dst[diff_dst_d.blk_off(
                                img, oc_off_idx)];
                        const data_t *local_src = diff_src;

                        auto p = jit_1x1_conv_call_s();
                        auto rp = rtus_driver_t<avx512_common>::call_params_t();

                        p.output_stride = utils::rnd_up(jcp.ic, jcp.ic_block)
                                * jcp.oc_block * jcp.typesize_out;

                        p.load_dim = this_block_size(oc_b * jcp.oc_block,
                                jcp.oc, load_step * jcp.oc_block);

                        p.bcast_dim = this_block_size(ic_b * jcp.ic_block,
                                jcp.ic, bcast_step * jcp.ic_block);
                        rp.icb = p.bcast_dim;
                        p.output_data = store_to;

                        p.reduce_dim = sp_b_step * jcp.reduce_block;
                        rp.os = p.reduce_dim;

                        p.first_last_flag = 0
                                | (mb_sp_b == mb_sp_b_start ? FLAG_REDUCE_FIRST
                                                            : 0)
                                | (sp_b_end == sp_nb ? FLAG_SP_LAST : 0);

                        int sp = sp_b * jcp.reduce_block;
                        int oc_mult
                                = is_ddst_layout_nxc ? jcp.oc : jcp.oc_block;
                        p.load_data = pdiff_dst + sp * oc_mult;

                        if (pd()->rtus_.reduce_src_) {
                            const int oh = sp / jcp.ow;
                            const int ow = sp % jcp.ow;

                            const int ih = oh * stride_h;
                            const int iw = ow * stride_w;
                            rp.iw_start = iw;

                            rp.ws = rtus_space
                                    + ithr * pd()->rtus_.space_per_thread_
                                    + sp * jcp.ic_block;

                            if (ndims == 3)
                                rp.src = local_src
                                        + iw * src_d.blocking_desc().strides[2];
                            else
                                rp.src = local_src
                                        + ih * src_d.blocking_desc().strides[2]
                                        + iw * src_d.blocking_desc().strides[3];
                            (*rtus_driver_)(&rp);

                            p.bcast_data = rp.ws;
                        } else {
                            int ic_mult
                                    = is_src_layout_nxc ? jcp.ic : jcp.ic_block;
                            p.bcast_data = local_src + sp * ic_mult;
                        }

                        (*kernel_)(&p);
                    }
                }
            }
        }

        /* diff_weights[:] += sum(wei_reduction[thr_mb][:]) */
        if (dnnl_thr_syncable() && jcp.nthr_mb > 1) {
            simple_barrier::barrier(&reduction_barrier, jcp.nthr);
            const int work = g_work * oc_b_work * ic_b_work;
            int start {0}, end {0};
            balance211(work, jcp.nthr_mb, ithr_mb, start, end);
            if (start == end) return;

            for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
                int w = start;
                int sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_start {0};
                nd_iterator_init(w, sub_g_start, g_work, sub_oc_b_start,
                        oc_b_work, sub_ic_b_start, ic_b_work);
                while (w < end) {
                    const int g = g_start + sub_g_start;
                    const int oc_b = oc_b_start + sub_oc_b_start;
                    const int ic_b = ic_b_start + sub_ic_b_start;
                    const int ic_to_accumulate
                            = nstl::min(end - w, ic_b_work - sub_ic_b_start)
                            * jcp.ic_block;
                    const int acc_size
                            = this_block_size(ic_b * jcp.ic_block,
                                      jcp.ic_without_padding, ic_to_accumulate)
                            * jcp.oc_block;

                    const size_t off
                            = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                    data_t *d = diff_weights + off;
                    data_t *s = wei_reduction + (thr_mb - 1) * wei_size + off;

                    acc_ker_->accumulate(d, s, acc_size);

                    nd_iterator_jump(w, end, sub_g_start, g_work,
                            sub_oc_b_start, oc_b_work, sub_ic_b_start,
                            ic_b_work);
                }
            }
        }
    };

    auto ker_bias = [&](int ithr, int nthr) {
        assert(nthr == rb->balancer().nthr_);

        const int b_job_start = rb->balancer().ithr_job_off(ithr);
        const int b_njobs = rb->balancer().ithr_njobs(ithr);

        if (b_njobs == 0) return;

        /* reduction dimension */
        int img_start {0}, img_end {0};

        balance211(jcp.mb, rb->balancer().nthr_per_group_,
                rb->balancer().id_in_group(ithr), img_start, img_end);

        /* jobs */
        int g_start {0}, ocb_start {0};
        nd_iterator_init(
                b_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_load);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const int oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb * jcp.oc_block
                        : g * jcp.nb_load + ocb;
                const data_t *d_dst
                        = &diff_dst[diff_dst_d.blk_off(img, oc_off_idx)];

                data_t *d_bias = rb->get_local_ptr(ithr, diff_bias,
                                         reducer_bia_scratchpad)
                        + b_job_loc * rb->balancer().job_size_;
                const int sp_shift = is_ddst_layout_nxc ? jcp.ngroups * jcp.oc
                                                        : jcp.oc_block;
                const auto max_oc = this_block_size(
                        ocb * jcp.oc_block, jcp.oc, jcp.oc_block);
                if (img == img_start)
                    for (int o = 0; o < 16; ++o)
                        d_bias[o] = 0.;

                for (int os = 0; os < jcp.os; ++os) {
                    PRAGMA_OMP_SIMD()
                    for (int o = 0; o < max_oc; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += sp_shift;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_load);
            }
        }

        if (dnnl_thr_syncable())
            rb->reduce(ithr, diff_bias, reducer_bia_scratchpad);
    };

    if (dnnl_thr_syncable()) {
        parallel(jcp.nthr, [&](const int ithr, const int nthr) {
            ker(ithr, jcp.nthr);
            if (pd()->with_bias()) ker_bias(ithr, jcp.nthr);
        });
    } else {
        parallel(jcp.nthr, [&](int ithr, int nthr) { ker(ithr, nthr); });
        if (jcp.nthr_mb > 1)
            parallel(jcp.nthr, [&](int ithr, int nthr) {
                assert(nthr == jcp.nthr);

                const int ithr_ic_b = ithr % jcp.nthr_ic_b;
                const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
                const int ithr_g
                        = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
                const int ithr_mb
                        = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b / jcp.nthr_g;

                /* independent dimensions */
                int g_start {0}, oc_b_start {0}, ic_b_start {0};
                int g_end {0}, oc_b_end {0}, ic_b_end {0};

                balance211(jcp.ngroups, jcp.nthr_g, ithr_g, g_start, g_end);
                balance211(jcp.nb_load, jcp.nthr_oc_b, ithr_oc_b, oc_b_start,
                        oc_b_end);
                balance211(jcp.nb_bcast, jcp.nthr_ic_b, ithr_ic_b, ic_b_start,
                        ic_b_end);

                const int g_work = g_end - g_start;
                const int oc_b_work = oc_b_end - oc_b_start;
                const int ic_b_work = ic_b_end - ic_b_start;

                const int work = g_work * oc_b_work * ic_b_work;
                int start {0}, end {0};
                balance211(work, jcp.nthr_mb, ithr_mb, start, end);
                if (start == end) return;

                for (int thr_mb = 1; thr_mb < jcp.nthr_mb; ++thr_mb) {
                    int w = start;
                    int sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_start {0};
                    nd_iterator_init(w, sub_g_start, g_work, sub_oc_b_start,
                            oc_b_work, sub_ic_b_start, ic_b_work);
                    while (w < end) {
                        const int g = g_start + sub_g_start;
                        const int oc_b = oc_b_start + sub_oc_b_start;
                        const int ic_b = ic_b_start + sub_ic_b_start;
                        const int ic_to_accumulate
                                = nstl::min(end - w, ic_b_work - sub_ic_b_start)
                                * jcp.ic_block;
                        const int acc_size
                                = this_block_size(ic_b * jcp.ic_block,
                                          jcp.ic_without_padding,
                                          ic_to_accumulate)
                                * jcp.oc_block;

                        const size_t off
                                = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                        data_t *d = diff_weights + off;
                        data_t *s
                                = wei_reduction + (thr_mb - 1) * wei_size + off;

                        acc_ker_->accumulate(d, s, acc_size);

                        nd_iterator_jump(w, end, sub_g_start, g_work,
                                sub_oc_b_start, oc_b_work, sub_ic_b_start,
                                ic_b_work);
                    }
                }
            });
        if (pd()->with_bias()) {
            parallel(jcp.nthr,
                    [&](int ithr, int nthr) { ker_bias(ithr, nthr); });
            parallel(jcp.nthr, [&](int ithr, int nthr) {
                assert(nthr == rb->balancer().nthr_);
                MAYBE_UNUSED(nthr);
                if (rb->balancer().ithr_njobs(ithr) == 0) return;
                rb->reduce_nolock(ithr, diff_bias, reducer_bia_scratchpad);
            });
        }
    }

    /* TODO: put this in ker_bias */
    if (is_bias_padded) {
        assert(IMPLICATION(!is_ddst_layout_nxc, jcp.ngroups == 1));
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
