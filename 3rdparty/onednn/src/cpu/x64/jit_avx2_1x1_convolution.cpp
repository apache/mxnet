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
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx2_1x1_convolution.hpp"

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

void jit_avx2_1x1_convolution_fwd_t::execute_forward(
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

    const auto &jcp = kernel_->jcp;
    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad.get<data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;
    }

    parallel(0, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, weights_dw, bias_dw,
                dst, scratchpad);
    });

    if (pd()->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad(ctx);
}

void jit_avx2_1x1_convolution_fwd_t::execute_forward_thr(const int ithr,
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
    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<data_t>(key_conv_rtus_space)
            : nullptr;

    const int ndims = dst_d.ndims();

    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    const int nb_oc = jcp.nb_load;
    const int nb_ic = jcp.nb_reduce;
    const int nb_ic_blocking = jcp.nb_reduce_blocking;

    auto p = jit_1x1_conv_call_s();
    auto rp = rtus_driver_t<avx2>::call_params_t();

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
    data_t *pbuf;
    size_t row_offset;
    const int nb_buffer = jcp.nb_load_blocking;
    auto jcp_dw = pd()->jcp_dw_;
    std::vector<data_t *> addrs;
    jit_generator *dw_jit_ker = nullptr;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto init_bcast = [&](int iwork, int bcast_end, int &n, int &g,
                              int &bcast_step, int &od, int &oh, int &ow,
                              int &id, int &ih, int &iw) {
        int osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, nb_bcast);

        bcast_step = step(
                nb_bcast_blocking, nb_bcast - osb, nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        const int os_2d = os % (jcp.oh * jcp.ow);
        od = os / (jcp.oh * jcp.ow);
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
        p.load_dim = this_block_size(
                ocb * jcp.oc_block, jcp.oc, load_step * jcp.oc_block);
    };

    auto ker_1x1 = [&](int ocb, int icb, int ocb_start, int n, int g, int od,
                           int oh, int ow, int id, int ih, int iw) {
        const bool is_dst_layout_nxc = utils::one_of(jcp.dst_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        const int oc_off_idx = is_dst_layout_nxc
                ? g * jcp.oc + ocb * jcp.oc_block
                : g * nb_oc + ocb;

        p.output_data = jcp.with_dw_conv
                ? pbuf + (oh % jcp_dw->kh) * row_offset
                : &dst[data_blk_off(dst_d, n, oc_off_idx, od, oh, ow)];
        p.bias_data
                = &bias[oc_off_idx * (is_dst_layout_nxc ? 1 : jcp.oc_block)];

        p.first_last_flag = 0 | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                | (icb + nb_ic_blocking >= nb_ic ? FLAG_REDUCE_LAST : 0);

        p.reduce_dim = this_block_size(
                icb * jcp.ic_block, jcp.ic, nb_ic_blocking * jcp.ic_block);
        rp.icb = p.reduce_dim;

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
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n {0}, g {0}, bcast_step, od, oh, ow, id, ih, iw;
            init_bcast(
                    iwork, bcast_end, n, g, bcast_step, od, oh, ow, id, ih, iw);
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, ocb_end, load_step);
                for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                    ker_1x1(ocb, icb, ocb_start, n, g, od, oh, ow, id, ih, iw);
                }
                ocb += load_step;
            }
            iwork += bcast_step;
        }
    };

    auto ker_dw = [&](int n, int ocb_start, int load_step, int &dw_oh) {
        int oh_1x1 = nstl::max(dw_oh * jcp_dw->stride_h - jcp_dw->t_pad, 0);

        for (int i = 0; i < jcp_dw->kh; ++i)
            addrs[i] = pbuf + ((oh_1x1++) % jcp_dw->kh) * row_offset;

        const auto ocb_end = ocb_start + load_step;
        const auto wch_stride
                = jcp_dw->iw * jcp_dw->nb_ch_blocking * jcp_dw->ch_block;
        const int dil_h = jcp_dw->dilate_h + 1;
        const int str_h = jcp_dw->stride_h;
        const int ch_num = jcp_dw->nb_ch_blocking;
        const int ow = 0;
        const int kw = 0;

        for (int ch = ocb_start; ch < ocb_end; ch += jcp_dw->nb_ch_blocking) {

            const int i_t_overflow
                    = nstl::max(0, (int)(jcp_dw->t_pad - dw_oh * str_h));
            const int i_b_overflow
                    = nstl::max(jcp_dw->ih,
                              (int)(dw_oh * str_h + (jcp_dw->kh - 1) * dil_h
                                      - jcp_dw->t_pad + 1))
                    - jcp_dw->ih;

            const int kh = div_up(i_t_overflow, dil_h);
            const int kh_padding = jcp_dw->kh - div_up(i_t_overflow, dil_h)
                    - div_up(i_b_overflow, dil_h);

            jit_conv_call_s par_conv_dw;

            par_conv_dw.src = addrs.data();
            par_conv_dw.dst = &dst[dst_d.blk_off(n, ch, dw_oh, ow)];

            par_conv_dw.filt
                    = &weights_dw[dw_weights_d.blk_off(ch, 0, 0, kh, kw)];
            if (bias)
                par_conv_dw.bias
                        = &bias_dw[dw_bias_d.blk_off(ch * jcp_dw->ch_block)];

            par_conv_dw.kh_padding = (size_t)nstl::max(0, kh_padding);

            par_conv_dw.ch_blocks = nstl::min(ch + ch_num, jcp_dw->nb_ch) - ch;

            (*dw_jit_ker)(&par_conv_dw);

            for (int i = 0; i < jcp_dw->kh; ++i)
                addrs[i] += wch_stride;
        }
    };

    auto conv_dw = [&]() {
        // Set variables
        memory_tracking::grantor_t dw_scratchpad(
                scratchpad, memory_tracking::names::prefix_fusion);
        auto dw_conv_buffer
                = dw_scratchpad.get<data_t>(key_fusion_inout_buffer);
        dw_jit_ker = kernel_dw_avx2 ? kernel_dw_avx2->ker()
                                    : kernel_dw_sse41->ker();

        const auto dw_conv_buffer_size_
                = (size_t)jcp_dw->kh * jcp.ow * nb_buffer * jcp.oc_block;
        pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;
        row_offset = dw_conv_buffer_size_ / jcp_dw->kh;
        addrs.resize(jcp_dw->kh);

        int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
        balance2D(nthr, ithr, jcp.mb * jcp.ngroups * jcp_dw->oh, bcast_start,
                bcast_end, nb_oc, ocb_start, ocb_end, 1);

        while (ocb_start < ocb_end) {
            int load_step;
            init_load(ocb_start, ocb_end, load_step);

            int oh_1x1 = 0;
            auto bcast_iter = bcast_start;
            while (bcast_iter < bcast_end) {
                int n, g, oh_dw;
                nd_iterator_init(bcast_iter, n, jcp.mb, g, jcp.ngroups, oh_dw,
                        jcp_dw->oh);
                if (oh_dw == 0) oh_1x1 = 0; // Reset over mb boundary
                const int oh_1x1_range
                        = oh_dw * jcp_dw->stride_h - jcp_dw->t_pad;
                const int oh_1x1_begin = nstl::max(oh_1x1_range, 0);
                const int oh_1x1_end
                        = nstl::min(oh_1x1_range + jcp_dw->kh, jcp.oh);
                oh_1x1 = nstl::max(
                        oh_1x1_begin, oh_1x1); // Skip rows computed previously

                // dw_spatial to 1x1 spatial conversion. if jcp.oh != jcp_dw->oh
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
        int start {0}, end {0};
        const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
        balance211(work_amount, nthr, ithr, start, end);
        conv_1x1(start, end, 0, jcp.nb_load);
    }
}

/* convolution backward wtr data */

void jit_avx2_1x1_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const auto &jcp = kernel_->jcp;
    auto rtus_space = pd()->rtus_.reduce_src_
            ? ctx.get_scratchpad_grantor().get<data_t>(key_conv_rtus_space)
            : nullptr;

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1);
    const int ndims = diff_dst_d.ndims();

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

    auto ker = [&](const int ithr, const int nthr) {
        auto p = jit_1x1_conv_call_s();
        auto rp = rtus_driver_t<avx2>::call_params_t();

        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int load_step = 0;
        for (int icb = 0; icb < jcp.nb_load; icb += load_step) {
            load_step = step(jcp.nb_load_blocking, jcp.nb_load - icb,
                    jcp.nb_load_blocking_max);

            p.load_dim = this_block_size(
                    icb * jcp.ic_block, jcp.ic, load_step * jcp.ic_block);
            rp.icb = p.load_dim;

            int bcast_step;
            for (int iwork = start; iwork < end; iwork += bcast_step) {
                int n {0}, g {0}, osb {0};
                nd_iterator_init(
                        iwork, n, jcp.mb, g, jcp.ngroups, osb, jcp.nb_bcast);

                bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                        jcp.nb_bcast_blocking_max);
                bcast_step = nstl::min(bcast_step, end - iwork);

                const int os = osb * os_block;
                p.bcast_dim
                        = this_block_size(os, jcp.os, bcast_step * os_block);
                rp.os = p.bcast_dim;

                const int od = os / (jcp.oh * jcp.ow);
                const int os_2d = os % (jcp.oh * jcp.ow);
                const int oh = os_2d / jcp.ow;
                const int ow = os_2d % jcp.ow;
                const int id = od * stride_d;
                const int ih = oh * stride_h;
                const int iw = ow * stride_w;
                rp.iw_start = iw;

                const bool is_dsrc_layout_nxc = utils::one_of(jcp.src_tag,
                        format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
                const int ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g * nb_ic + icb;
                rp.src = diff_src
                        + data_blk_off(diff_src_d, n, ic_off_idx, id, ih, iw);
                if (pd()->rtus_.reduce_src_) {
                    rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_;
                    p.output_data = rp.ws;
                } else
                    p.output_data = rp.src;

                for (int ocb = 0; ocb < jcp.nb_reduce;
                        ocb += jcp.nb_reduce_blocking) {
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

                    p.reduce_dim = this_block_size(ocb * jcp.oc_block, jcp.oc,
                            nb_oc_blocking * jcp.oc_block);

                    (*kernel_)(&p);
                }

                if (pd()->rtus_.reduce_src_) (*rtus_driver_)(&rp);
            }
        }
    };

    parallel(jcp.nthr, ker);
}

/* convolution backward wtr weights */

status_t jit_avx2_1x1_convolution_bwd_weights_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_,
            new jit_avx2_1x1_conv_kernel_f32(pd()->jcp_, *pd()->attr())));
    CHECK(kernel_->create_kernel());

    CHECK(safe_ptr_assign(reducer_weights_,
            new cpu_reducer_2d_t<data_type::f32>(pd()->reducer_wei_conf_)));
    CHECK(reducer_weights_->create_kernel());

    CHECK(safe_ptr_assign(reducer_bias_,
            new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_)));
    if (pd()->with_bias()) {
        assert(reducer_weights_->balancer().nthr_
                == reducer_bias_->balancer().nthr_);
        CHECK(reducer_bias_->create_kernel());
    }

    CHECK(init_rtus_driver<avx2>(this));
    return status::success;
}

void jit_avx2_1x1_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias_in = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_BIAS);

    auto scratchpad = ctx.get_scratchpad_grantor();

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    const auto &jcp = kernel_->jcp;
    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<data_t>(key_conv_rtus_space)
            : nullptr;

    const bool is_bias_padded
            = pd()->with_bias() && (jcp.oc_without_padding % jcp.oc_block != 0);

    data_t *diff_bias = is_bias_padded
            ? scratchpad.get<data_t>(key_conv_padded_bias)
            : diff_bias_in;

    auto reducer_bia_scratchpad
            = memory_tracking::grantor_t(scratchpad, prefix_reducer_bia);
    auto rb = this->reducer_bias_.get();
    rb->init(reducer_bia_scratchpad);

    auto reducer_wei_scratchpad
            = memory_tracking::grantor_t(scratchpad, prefix_reducer_wei);
    auto rw = this->reducer_weights_.get();
    rw->init(reducer_wei_scratchpad);

    const int ndims = diff_dst_d.ndims();
    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

    const int nb_ic = jcp.nb_bcast;
    const int nb_ic_blocking = jcp.nb_bcast_blocking;
    const int bcast_work = div_up(nb_ic, nb_ic_blocking);

    const int nb_oc = jcp.nb_load;
    const int nb_oc_blocking = jcp.nb_load_blocking;
    const int load_work = div_up(nb_oc, nb_oc_blocking);

    const int sp_dim = jcp.reduce_dim;
    const int mb_sp_work = jcp.mb * sp_dim;

    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    const bool is_src_layout_nxc = utils::one_of(
            jcp.src_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    const bool is_ddst_layout_nxc = utils::one_of(
            jcp.dst_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto oc_ic_sp_loop = [=](int sp_start, int sp_end, bool first_image,
                                 data_t *store_to, size_t store_to_ld,
                                 const data_t *diff_dst, const data_t *src,
                                 int ithr) {
        auto p = jit_1x1_conv_call_s();
        auto rp = rtus_driver_t<avx2>::call_params_t();

        p.output_stride = store_to_ld * sizeof(float);

        int oc_b_step = 0;
        for (int oc_b = 0; oc_b < nb_oc_blocking; oc_b += oc_b_step) {
            oc_b_step = step(nb_oc_blocking, nb_oc_blocking - oc_b,
                    jcp.nb_load_blocking_max);
            p.load_dim = this_block_size(
                    oc_b * jcp.oc_block, jcp.oc, oc_b_step * jcp.oc_block);

            int ic_b_step = 0;
            for (int ic_b = 0; ic_b < nb_ic_blocking; ic_b += ic_b_step) {
                ic_b_step = step(nb_ic_blocking, nb_ic_blocking - ic_b,
                        jcp.nb_bcast_blocking_max);
                p.bcast_dim = this_block_size(
                        ic_b * jcp.ic_block, jcp.ic, ic_b_step * jcp.ic_block);
                rp.icb = p.bcast_dim;

                p.output_data = store_to + oc_b * store_to_ld
                        + ic_b * jcp.ic_block * jcp.oc_block;

                /* spatial reduction */
                int sp_step = 0;
                for (int sp = sp_start; sp < sp_end; sp += sp_step) {
                    sp_step = step(jcp.nb_reduce_blocking, sp_end - sp,
                            jcp.nb_reduce_blocking_max);
                    p.reduce_dim = sp_step * jcp.reduce_block;
                    rp.os = p.reduce_dim;

                    p.first_last_flag = sp == sp_start && first_image
                            ? FLAG_REDUCE_FIRST
                            : 0;

                    p.load_data = diff_dst
                            + (oc_b * jcp.reduce_dim + sp)
                                    * (is_ddst_layout_nxc ? jcp.oc
                                                          : jcp.oc_block);

                    if (pd()->rtus_.reduce_src_) {
                        const int od = sp / (jcp.oh * jcp.ow);
                        const int sp_2d = sp % (jcp.oh * jcp.ow);
                        const int oh = sp_2d / jcp.ow;
                        const int ow = sp_2d % jcp.ow;

                        const int id = od * stride_d;
                        const int ih = oh * stride_h;
                        const int iw = ow * stride_w;
                        rp.iw_start = iw;

                        rp.ws = rtus_space
                                + ithr * pd()->rtus_.space_per_thread_
                                + (ic_b * jcp.is + sp) * jcp.ic_block;
                        size_t src_offset
                                = iw * src_d.blocking_desc().strides[ndims - 1];
                        if (ndims > 3)
                            src_offset += ih
                                    * src_d.blocking_desc().strides[ndims - 2];
                        if (ndims == 5)
                            src_offset += id
                                    * src_d.blocking_desc().strides[ndims - 3];

                        rp.src = src + src_offset;
                        if (oc_b == 0) (*rtus_driver_)(&rp);

                        p.bcast_data = rp.ws;
                    } else
                        p.bcast_data = src
                                + (ic_b * jcp.reduce_dim + sp)
                                        * (is_src_layout_nxc ? jcp.ic
                                                             : jcp.ic_block);

                    (*kernel_)(&p);
                }
            }
        }
    };

    auto ker = [&](const int ithr, const int nthr) {
        assert(nthr == rw->balancer().nthr_);

        const int w_njobs = rw->balancer().ithr_njobs(ithr);
        if (w_njobs == 0) return;

        /* setup: independent work (oc, ic) */
        const int w_job_start = rw->balancer().ithr_job_off(ithr);
        int g {0}, load_i {0}, bcast_i {0};
        nd_iterator_init(w_job_start, g, jcp.ngroups, load_i, load_work,
                bcast_i, bcast_work);

        /* setup: reduction work (mb, sp) */
        int mb_sp_start {0}, mb_sp_end {0};
        balance211(mb_sp_work, rw->balancer().nthr_per_group_,
                rw->balancer().id_in_group(ithr), mb_sp_start, mb_sp_end);
        int img_start {0}, sp_start {0};
        nd_iterator_init(mb_sp_start, img_start, jcp.mb, sp_start, sp_dim);

        /* independent work */
        for (int iwork = 0; iwork < w_njobs; ++iwork) {
            const int oc_b = nb_oc_blocking * load_i;
            const int ic_b = nb_ic_blocking * bcast_i;

            const int oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + oc_b * jcp.oc_block
                    : g * nb_oc + oc_b;
            const int ic_off_idx = is_src_layout_nxc
                    ? g * jcp.ic + ic_b * jcp.ic_block
                    : g * nb_ic + ic_b;

            data_t *store_to;
            size_t store_to_ld;

            if (rw->balancer().nthr_per_group_ == 1) {
                const size_t off = pd()->with_groups()
                        ? diff_weights_d.blk_off(g, oc_b, ic_b)
                        : diff_weights_d.blk_off(oc_b, ic_b);
                store_to = &diff_weights[off];
                store_to_ld = rnd_up(jcp.ic, jcp.ic_block) * jcp.oc_block;
            } else {
                const size_t off = (size_t)iwork * rw->balancer().job_size_;
                store_to
                        = rw->get_local_ptr(ithr, reducer_wei_scratchpad) + off;
                store_to_ld = nb_ic_blocking * jcp.ic_block * jcp.oc_block;
            }

            /* reduction work */
            int img = img_start;
            int sp = sp_start;
            int sp_step = 0;
            for (int mb_sp = mb_sp_start; mb_sp < mb_sp_end; mb_sp += sp_step) {
                sp_step = nstl::min(sp_dim - sp, mb_sp_end - mb_sp);

                const bool first_image = img == img_start;
                if (is_ddst_layout_nxc && first_image
                        && rw->balancer().nthr_per_group_ > 1) {
                    // Zero-pad the scratchpad when nthr > 1 (since most threads
                    // write to scratchpad) so that zero-padding is maintained
                    // for the final output after reduction
                    array_set(rw->get_local_ptr(ithr, reducer_wei_scratchpad)
                                    + iwork * rw->balancer().job_size_,
                            0, rw->balancer().job_size_);
                }
                oc_ic_sp_loop(sp, sp + sp_step, first_image, store_to,
                        store_to_ld,
                        &diff_dst[diff_dst_d.blk_off(img, oc_off_idx)],
                        &src[src_d.blk_off(img, ic_off_idx)], ithr);

                sp = 0;
                img += 1;
            }

            nd_iterator_step(
                    g, jcp.ngroups, load_i, load_work, bcast_i, bcast_work);
        }

        if (dnnl_thr_syncable())
            rw->reduce(ithr, diff_weights, reducer_wei_scratchpad);
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
        nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start, nb_oc);

        for (int img = img_start; img < img_end; ++img) {
            int g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const int oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb * jcp.oc_block
                        : g * nb_oc + ocb;

                const data_t *d_dst
                        = &diff_dst[diff_dst_d.blk_off(img, oc_off_idx)];
                data_t *d_bias = rb->get_local_ptr(ithr, diff_bias,
                                         reducer_bia_scratchpad)
                        + b_job_loc * rb->balancer().job_size_;

                if (img == img_start)
                    for (int o = 0; o < 8; ++o)
                        d_bias[o] = 0.;

                const int spatial_shift
                        = is_ddst_layout_nxc ? jcp.oc : jcp.oc_block;
                const int max_oc = this_block_size(
                        ocb * jcp.oc_block, jcp.oc, jcp.oc_block);
                for (int hw = 0; hw < jcp.os; ++hw) {
                    PRAGMA_OMP_SIMD()
                    for (int o = 0; o < max_oc; ++o)
                        d_bias[o] += d_dst[o];
                    d_dst += spatial_shift;
                }

                nd_iterator_step(g, jcp.ngroups, ocb, nb_oc);
            }
        }

        if (dnnl_thr_syncable())
            rb->reduce(ithr, diff_bias, reducer_bia_scratchpad);
    };

    if (dnnl_thr_syncable()) {
        assert(IMPLICATION(pd()->with_bias(),
                rw->balancer().nthr_ == rb->balancer().nthr_));
        parallel(rw->balancer().nthr_, [&](const int ithr, const int nthr) {
            ker(ithr, nthr);
            if (pd()->with_bias()) ker_bias(ithr, nthr);
        });
    } else {
        parallel(rw->balancer().nthr_,
                [&](int ithr, int nthr) { ker(ithr, nthr); });
        parallel(rw->balancer().nthr_, [&](int ithr, int nthr) {
            assert(nthr == rw->balancer().nthr_);
            MAYBE_UNUSED(nthr);
            if (rw->balancer().ithr_njobs(ithr) == 0) return;
            rw->reduce_nolock(ithr, diff_weights, reducer_wei_scratchpad);
        });
        if (pd()->with_bias()) {
            parallel(rb->balancer().nthr_,
                    [&](int ithr, int nthr) { ker_bias(ithr, nthr); });
            parallel(rb->balancer().nthr_, [&](int ithr, int nthr) {
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
        const int padded_stride = utils::rnd_up(jcp.oc, jcp.oc_block);
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
