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

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_sse41_1x1_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

#define data_blk_off(f, n, c, h, w) \
    ((ndims == 3) ? (f).blk_off(n, c, w) : (f).blk_off(n, c, h, w))

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

void jit_sse41_1x1_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto weights_dw = CTX_IN_MEM(
            const data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
    auto bias_dw = CTX_IN_MEM(
            const data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);

    auto scratchpad = ctx.get_scratchpad_grantor();
    parallel(kernel_->jcp.nthr, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, weights_dw, bias_dw,
                dst, scratchpad);
    });

    if (pd()->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad(ctx);
}

void jit_sse41_1x1_convolution_fwd_t::execute_forward_thr(const int ithr,
        const int nthr, const data_t *src, const data_t *weights,
        const data_t *bias, const data_t *weights_dw, const data_t *bias_dw,
        data_t *dst, const memory_tracking::grantor_t &scratchpad) const {

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper dw_weights_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS));
    const memory_desc_wrapper dw_bias_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS));

    const auto &jcp = kernel_->jcp;
    const int ndims = src_d.ndims();

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    auto par_conv = jit_1x1_conv_call_s();

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
    data_t *pbuf {nullptr};
    size_t row_offset {};
    const int nb_buffer = jcp.nb_load_blocking;
    const int jcp_dw_kh = 3;
    std::vector<data_t *> addrs;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto init_bcast = [&](int iwork, int &n, int &g, int &bcast_step,
                              int bcast_end, int &oh, int &ow, int &ih,
                              int &iw) {
        int osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, nb_bcast);

        bcast_step = step(
                nb_bcast_blocking, nb_bcast - osb, nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        ow = os % jcp.ow;
        oh = os / jcp.ow;

        ih = oh * jcp.stride_h;
        iw = ow * jcp.stride_w;

        par_conv.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
    };

    auto init_load = [&](int ocb, int ocb_end, int &load_step) {
        load_step = step(nb_load_blocking, ocb_end - ocb, nb_load_blocking_max);
        par_conv.load_dim = this_block_size(
                ocb * jcp.oc_block, jcp.oc, load_step * jcp.oc_block);
    };

    auto inner_ker = [&](int ocb, int icb, int n, int g, int oh, int ow, int ih,
                             int iw) {
        const size_t _ocb = g * nb_oc + ocb;
        const size_t _icb = g * nb_ic + icb;

        const bool is_dst_layout_nxc = utils::one_of(jcp.dst_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        const int oc_off_idx = (is_dst_layout_nxc ? jcp.oc_block : 1) * _ocb;

        par_conv.output_data = jcp.with_dw_conv
                ? pbuf + (oh % jcp_dw_kh) * row_offset
                : &dst[data_blk_off(dst_d, n, oc_off_idx, oh, ow)];

        par_conv.bias_data = &bias[_ocb * jcp.oc_block];

        par_conv.first_last_flag = 0 | (icb == 0) * FLAG_REDUCE_FIRST
                | (icb + nb_ic_blocking >= nb_ic) * FLAG_REDUCE_LAST;

        par_conv.reduce_dim = this_block_size(
                icb * jcp.ic_block, jcp.ic, nb_ic_blocking * jcp.ic_block);

        const bool is_src_layout_nxc = utils::one_of(jcp.src_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        const int ic_off_idx = (is_src_layout_nxc ? jcp.ic_block : 1) * _icb;

        const size_t src_off = data_blk_off(src_d, n, ic_off_idx, ih, iw);
        par_conv.bcast_data = &src[src_off];

        par_conv.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];

        (*kernel_)(&par_conv);
    };

    auto conv_1x1 = [&](int bcast_start, int bcast_end, int ocb_start,
                            int ocb_end) {
        if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;

        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n {0}, g {0}, bcast_step, oh, ow, ih, iw;
            init_bcast(iwork, n, g, bcast_step, bcast_end, oh, ow, ih, iw);
            int ocb = 0;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, ocb_end, load_step);
                for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                    inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
                }
                ocb += load_step;
            }
            iwork += bcast_step;
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

            const int ow = 0;
            const int kw = 0;
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
        auto &jcp_dw = pd()->dw_conv_pd_->jcp_;
        memory_tracking::grantor_t dw_scratchpad(
                scratchpad, memory_tracking::names::prefix_fusion);
        const auto dw_conv_buffer = dw_scratchpad.get<data_t>(
                memory_tracking::names::key_fusion_inout_buffer);

        const auto dw_conv_buffer_size_
                = (size_t)jcp_dw.kh * jcp.ow * nb_buffer * jcp.oc_block;
        pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;
        row_offset = dw_conv_buffer_size_ / jcp_dw.kh;
        addrs.resize(jcp_dw.kh);

        int bcast_start {0}, bcast_end {0}, ocb_start, ocb_end;
        balance2D(nthr, ithr, jcp.mb * jcp.ngroups * jcp_dw.oh, bcast_start,
                bcast_end, nb_oc, ocb_start, ocb_end, 1);

        while (ocb_start < ocb_end) {
            int load_step;
            init_load(ocb_start, ocb_end, load_step);

            int oh_1x1 = 0;
            auto bcast_iter = bcast_start;
            while (bcast_iter < bcast_end) {
                int n, g, oh_dw;
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
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        conv_1x1(start, end, 0, jcp.nb_load);
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
