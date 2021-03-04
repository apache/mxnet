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

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_avx512_core_bf16_1x1_convolution.hpp"

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::prop_kind;

#define data_blk_off(f, n, c, d, h, w) \
    ((ndims == 3) ? (f).blk_off(n, c, w) \
                  : ((ndims == 4) ? (f).blk_off(n, c, h, w) \
                                  : (f).blk_off(n, c, d, h, w)))

namespace {
/*TODO: investigate why common balance2D defined in common/dnnl_thread.hpp
 * not used here ?*/
template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end, T nx, T &nx_start,
        T &nx_end, T nx_divider) {
    const T grp_size = utils::div_up(nthr, nx_divider);
    const T grp_count = utils::div_up(nthr, grp_size);

    T grp = ithr / grp_size;
    T grp_ithr = ithr % grp_size;
    T grp_nthr = grp_size;
    T first_grps = nthr % grp_count;
    if (first_grps > 0 && grp >= first_grps) {
        ithr -= first_grps * grp_size;
        grp_nthr--;
        grp = ithr / grp_nthr + first_grps;
        grp_ithr = ithr % grp_nthr;
    }
    balance211(nx, grp_count, grp, nx_start, nx_end);
    balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}
} // namespace

/* convolution forward */
template <data_type_t dst_type>
void jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(const char *, DNNL_ARG_DST);
    auto weights_dw = CTX_IN_MEM(
            const dw_wei_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);

    auto scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = kernel_->jcp;
    if (pd()->wants_padded_bias()) {
        const size_t bia_dt_size = pd()->jcp_.typesize_bia;
        auto padded_bias = scratchpad.template get<char>(key_conv_padded_bias);
        utils::array_copy(
                padded_bias, bias, bia_dt_size * jcp.oc_without_padding);
        utils::array_set(padded_bias + bia_dt_size * jcp.oc_without_padding,
                0.f, bia_dt_size * (jcp.oc - jcp.oc_without_padding));
        bias = padded_bias;
    }

    float *bias_dw = nullptr;
    if (pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS)->data_type
            == data_type::bf16) {
        auto jcp_dw = pd()->jcp_dw_;
        memory_tracking::grantor_t dw_scratchpad(
                scratchpad, memory_tracking::names::prefix_fusion);
        auto bias_in = CTX_IN_MEM(
                const src_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
        bias_dw = dw_scratchpad.template get<float>(
                key_conv_bias_bf16_convert_wsp);
        cvt_bfloat16_to_float(bias_dw, bias_in, jcp_dw->oc_without_padding);
        utils::array_set(bias_dw + jcp_dw->oc_without_padding, 0.f,
                jcp_dw->oc - jcp_dw->oc_without_padding);
    } else {
        auto bias_in = CTX_IN_MEM(
                const float *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
        bias_dw = const_cast<float *>(bias_in);
    }

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, weights_dw, bias_dw,
                dst, scratchpad);
    });

    if (pd()->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad(ctx);
}

template <data_type_t dst_type>
void jit_avx512_core_bf16_1x1_convolution_fwd_t<dst_type>::execute_forward_thr(
        const int ithr, const int nthr, const src_data_t *src,
        const wei_data_t *weights, const char *bias,
        const dw_wei_data_t *weights_dw, const float *bias_dw, const char *dst,
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
    float *store_buffer = scratchpad.template get<float>(key_conv_store_wsp);

    const int ndims = src_d.ndims();
    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

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
    dst_data_t *pbuf; //bf16->bf16 fusion
    size_t row_offset;
    const auto jcp_dw = pd()->jcp_dw_;
    const int nb_buffer = jcp.nb_load_blocking;
    std::vector<decltype(pbuf)> addrs;

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

        void *output_data = jcp.with_dw_conv
                ? (void *)(pbuf + (oh % jcp_dw->kh) * row_offset)
                : (void *)(&dst[dst_off * dst_d.data_type_size()]);
        p.output_data = output_data;

        p.bias_data = &bias[jcp.typesize_bia * oc_off_idx
                * (is_dst_layout_nxc ? 1 : jcp.oc_block)];
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

        const size_t grp_count = utils::div_up(
                jcp.nthr, utils::div_up(jcp.nthr, jcp.load_grp_count));
        const size_t max_load_per_thread = is_dst_layout_nxc
                ? jcp.load_dim
                : rnd_up((jcp.load_dim / grp_count), jcp.load_block);
        const size_t str_size = jcp.bcast_dim * max_load_per_thread;
        p.store_buffer = store_buffer + ithr * str_size
                + data_blk_off(dst_d, 0, 0, od, oh, ow);

        (*kernel_)(&p);
    };

    auto conv_1x1 = [&](int bcast_start, int bcast_end, int ocb_start,
                            int ocb_end) {
        if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;
        if (jcp.loop_order == loop_lbr) {
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
        int oh_1x1 = nstl::max(dw_oh * jcp_dw->stride_h - jcp_dw->t_pad, 0);

        for (int i = 0; i < jcp_dw->kh; ++i)
            addrs[i] = pbuf + ((oh_1x1++) % jcp_dw->kh) * row_offset;

        const auto ocb_end = ocb_start + load_step;
        const auto wch_stride
                = jcp_dw->iw * jcp_dw->nb_ch_blocking * jcp_dw->ch_block;

        const int dil_h = jcp_dw->dilate_h + 1;
        const int str_h = jcp_dw->stride_h;
        const int ch_num = jcp_dw->nb_ch_blocking;

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

            const int ow = 0;
            const int kw = 0;
            jit_conv_call_s par_conv_dw;

            par_conv_dw.src = addrs.data();
            par_conv_dw.dst = &dst[dst_d.blk_off(n, ch, dw_oh, ow)
                    * dst_d.data_type_size()];

            par_conv_dw.filt
                    = &weights_dw[dw_weights_d.blk_off(ch, 0, 0, kh, kw)];
            if (bias)
                par_conv_dw.bias
                        = &bias_dw[dw_bias_d.blk_off(ch * jcp_dw->ch_block)];

            par_conv_dw.kh_padding = (size_t)nstl::max(0, kh_padding);

            par_conv_dw.ch_blocks = nstl::min(ch + ch_num, jcp_dw->nb_ch) - ch;

            (*kernel_dw_)(&par_conv_dw);

            for (int i = 0; i < jcp_dw->kh; ++i)
                addrs[i] += wch_stride;
        }
    };

    auto conv_dw = [&]() {
        // Set variables
        memory_tracking::grantor_t dw_scratchpad(
                scratchpad, memory_tracking::names::prefix_fusion);
        const auto dw_conv_buffer
                = dw_scratchpad.get<dst_data_t>(key_fusion_inout_buffer);

        const auto dw_conv_buffer_size_
                = jcp_dw->kh * jcp.ow * nb_buffer * jcp.oc_block;
        pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;
        row_offset = dw_conv_buffer_size_ / jcp_dw->kh;
        addrs.resize(jcp_dw->kh);

        int bcast_start {0}, bcast_end {0}, ocb_start, ocb_end;
        balance2D(nthr, ithr, jcp.mb * jcp.ngroups * jcp_dw->oh, bcast_start,
                bcast_end, nb_oc, ocb_start, ocb_end, jcp.load_grp_count);

        while (ocb_start < ocb_end) {
            int load_step;
            init_load(ocb_start, ocb_end, load_step);

            int oh_1x1 = 0;
            auto bcast_iter = bcast_start;
            while (bcast_iter < bcast_end) {
                int n {0}, g {0}, oh_dw {0};
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
        const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
        int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
                ocb_start, ocb_end, jcp.load_grp_count);

        conv_1x1(bcast_start, bcast_end, ocb_start, ocb_end);
    }
}

template struct jit_avx512_core_bf16_1x1_convolution_fwd_t<data_type::f32>;
template struct jit_avx512_core_bf16_1x1_convolution_fwd_t<data_type::bf16>;

template <data_type_t diff_src_type>
void jit_avx512_core_bf16_1x1_convolution_bwd_data_t<
        diff_src_type>::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);
    auto scratchpad = ctx.get_scratchpad_grantor();
    const auto &jcp = kernel_->jcp;
    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);
        execute_backward_data_thr(
                ithr, nthr, diff_dst, weights, diff_src, scratchpad);
    });
}

template <data_type_t diff_src_type>
void jit_avx512_core_bf16_1x1_convolution_bwd_data_t<
        diff_src_type>::execute_backward_data_thr(const int ithr,
        const int nthr, const diff_dst_data_t *diff_dst,
        const wei_data_t *weights, diff_src_data_t *diff_src,
        const memory_tracking::grantor_t &scratchpad) const {

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const auto &jcp = kernel_->jcp;

    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.template get<diff_src_data_t>(key_conv_rtus_space)
            : nullptr;
    float *store_buffer = scratchpad.template get<float>(key_conv_store_wsp);
    const int ndims = diff_src_d.ndims();
    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();
    const int nb_ic = jcp.nb_load;
    const int nb_oc = jcp.nb_reduce;
    const int os_block = jcp.bcast_block;
    const int nb_oc_blocking = jcp.nb_reduce_blocking;

    int bcast_start {0}, bcast_end {0}, icb_start {0}, icb_end {0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
            icb_start, icb_end, jcp.load_grp_count);

    auto init_bcast = [&](int iwork, int &n, int &g, int &bcast_step, int &od,
                              int &oh, int &ow, int &id, int &ih, int &iw) {
        int osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, jcp.nb_bcast);
        bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                jcp.nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        od = os / (jcp.oh * jcp.ow);
        const int os_2d = os % (jcp.oh * jcp.ow);
        oh = os_2d / jcp.ow;
        ow = os_2d % jcp.ow;
        id = od * stride_d;
        ih = oh * stride_h;
        iw = ow * stride_w;
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](int icb, int &load_step) {
        load_step = step(
                jcp.nb_load_blocking, icb_end - icb, jcp.nb_load_blocking_max);
        const int max_ic = nstl::min(icb_end * jcp.ic_block, jcp.ic);
        p.load_dim = this_block_size(
                icb * jcp.ic_block, max_ic, load_step * jcp.ic_block);
        rp.icb = p.load_dim;
    };

    auto init_reduce = [&](int ocb) {
        const int nb_oc_blocking_step
                = nstl::min(ocb + nb_oc_blocking, nb_oc) - ocb;
        p.first_last_flag = 0 | (ocb == 0 ? FLAG_REDUCE_FIRST : 0)
                | (ocb + nb_oc_blocking_step >= nb_oc ? FLAG_REDUCE_LAST : 0);

        p.reduce_dim = this_block_size(
                ocb * jcp.oc_block, jcp.oc, nb_oc_blocking_step * jcp.oc_block);
    };

    auto inner_ker = [&](int icb, int ocb, int n, int g, int od, int oh, int ow,
                             int id, int ih, int iw) {
        const bool is_dsrc_layout_nxc = utils::one_of(jcp.src_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        const int ic_off_idx = is_dsrc_layout_nxc
                ? g * jcp.ic + icb * jcp.ic_block
                : g * nb_ic + icb;
        const size_t diff_src_off
                = data_blk_off(diff_src_d, n, ic_off_idx, id, ih, iw);

        rp.src = diff_src + diff_src_off;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_;
            p.output_data = rp.ws;
        } else
            p.output_data = rp.src;
        p.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];

        const bool is_ddst_layout_nxc = utils::one_of(jcp.dst_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        const int oc_off_idx = is_ddst_layout_nxc
                ? g * jcp.oc + ocb * jcp.oc_block
                : g * nb_oc + ocb;
        p.bcast_data = diff_dst
                + data_blk_off(diff_dst_d, n, oc_off_idx, od, oh, ow);

        const size_t grp_count = utils::div_up(
                jcp.nthr, utils::div_up(jcp.nthr, jcp.load_grp_count));
        const size_t max_load_per_thread = is_dsrc_layout_nxc
                ? jcp.load_dim
                : rnd_up((jcp.load_dim / grp_count), jcp.load_block);
        const size_t str_size = jcp.bcast_dim * max_load_per_thread;
        p.store_buffer = store_buffer + ithr * str_size
                + data_blk_off(diff_src_d, 0, 0, id, ih, iw);
        (*kernel_)(&p);
        if (pd()->rtus_.reduce_src_) (*rtus_driver_)(&rp);
    };

    if (jcp.loop_order == loop_lbr) {
        int icb = icb_start;
        while (icb < icb_end) {
            int load_step;
            init_load(icb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, od, oh, ow, id, ih, iw;
                init_bcast(iwork, n, g, bcast_step, od, oh, ow, id, ih, iw);
                for (int ocb = 0; ocb < nb_oc; ocb += nb_oc_blocking) {
                    init_reduce(ocb);
                    inner_ker(icb, ocb, n, g, od, oh, ow, id, ih, iw);
                }
                iwork += bcast_step;
            }
            icb += load_step;
        }
    } else {
        assert(!"unsupported loop order");
    }
}

template struct jit_avx512_core_bf16_1x1_convolution_bwd_data_t<data_type::f32>;
template struct jit_avx512_core_bf16_1x1_convolution_bwd_data_t<
        data_type::bf16>;

/* convolution backward wtr weights */

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

template <data_type_t diff_weights_type>
status_t
jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<diff_weights_type>::init(
        engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_,
            new jit_avx512_core_bf16_1x1_conv_kernel(
                    pd()->jcp_, *pd()->attr())));

    CHECK(safe_ptr_assign(
            acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
    CHECK(kernel_->create_kernel());
    CHECK(acc_ker_->create_kernel());

    if (!pd()->jcp_.uses_permw_transposition) {
        const bool is_src_layout_nxc = utils::one_of(pd()->jcp_.src_tag,
                format_tag::ndhwc, format_tag::nhwc, format_tag::nwc);
        const bool is_ddst_layout_nxc = utils::one_of(pd()->jcp_.dst_tag,
                format_tag::ndhwc, format_tag::nhwc, format_tag::nwc);
        if (!is_src_layout_nxc || !is_ddst_layout_nxc) {
            CHECK(safe_ptr_assign(tr_reorder_,
                    new jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t()));
            CHECK(tr_reorder_->create_kernel());
        }
        if (is_src_layout_nxc) {
            int ic = pd()->jcp_.ic * pd()->jcp_.ngroups;
            CHECK(safe_ptr_assign(tr_reorder_nhwc_src_,
                    new jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t(ic)));
            CHECK(tr_reorder_nhwc_src_->create_kernel());
        }
        if (is_ddst_layout_nxc) {
            int oc = pd()->jcp_.oc * pd()->jcp_.ngroups;
            CHECK(safe_ptr_assign(tr_reorder_nhwc_ddst_,
                    new jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t(oc)));
            CHECK(tr_reorder_nhwc_ddst_->create_kernel());
        }
    }

    CHECK(init_rtus_driver<avx512_common>(this));
    return status::success;
}

template <data_type_t diff_weights_type>
void jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<diff_weights_type>::
        execute_backward_weights(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(diff_wei_data_t *, DNNL_ARG_DIFF_WEIGHTS);

    auto scratchpad = ctx.get_scratchpad_grantor();
    const auto &jcp = pd()->jcp_;

    float *diff_bias = nullptr;
    if (jcp.with_bias && pd()->jcp_.bia_dt == data_type::f32) {
        diff_bias = pd()->with_bias() && jcp.oc_without_padding % jcp.oc_block
                ? scratchpad.template get<float>(key_conv_padded_bias)
                : CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);
    }
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    auto rtus_space = scratchpad.template get<src_data_t>(key_conv_rtus_space);
    auto wei_reduction = scratchpad.template get<float>(key_conv_wei_reduction);

    auto tr_src_buffer = !jcp.uses_permw_transposition
            ? scratchpad.template get<src_data_t>(key_conv_tr_src)
            : nullptr;
    auto tr_diff_buffer = !jcp.uses_permw_transposition
            ? scratchpad.template get<diff_dst_data_t>(key_conv_tr_diff_dst)
            : nullptr;

    const int ndims = src_d.ndims();
    const int wei_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block)
            * rnd_up(jcp.ic, jcp.ic_block);
    const int n_wei_buffers
            = jcp.dst_dt == data_type::bf16 ? jcp.nthr_mb : jcp.nthr_mb - 1;
    auto bia_reduction = wei_reduction + n_wei_buffers * wei_size;

    simple_barrier::ctx_t reduction_barrier;
    if (dnnl_thr_syncable()) simple_barrier::ctx_init(&reduction_barrier);

    // TODO (Roma): remove this restriction
    assert(jcp.stride_w == 1 && jcp.stride_h == 1);

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

    const bool is_ddst_layout_nxc = utils::one_of(
            jcp.dst_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    auto ker = [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);

        const int ithr_ic_b = ithr % jcp.nthr_ic_b;
        const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
        const int ithr_g = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
        const int ithr_mb = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b / jcp.nthr_g;

        /* reduction dimension */
        int mb_sp_b_start {0}, mb_sp_b_end {0};
        balance211(
                mb_sp_work, jcp.nthr_mb, ithr_mb, mb_sp_b_start, mb_sp_b_end);

        /* independent dimensions */
        int g_start {0}, oc_b_start {0}, ic_b_start {0};
        int g_end {0}, oc_b_end {0}, ic_b_end {0};

        balance211(jcp.ngroups, jcp.nthr_g, ithr_g, g_start, g_end);
        balance211(jcp.nb_load, jcp.nthr_oc_b, ithr_oc_b, oc_b_start, oc_b_end);
        balance211(
                jcp.nb_bcast, jcp.nthr_ic_b, ithr_ic_b, ic_b_start, ic_b_end);

        float *diff_wei;
        if (diff_weights_type == data_type::bf16) {
            diff_wei = wei_reduction + (ithr_mb)*wei_size;
        } else {
            diff_wei = ithr_mb == 0
                    ? (float *)diff_weights
                    : (float *)wei_reduction + (ithr_mb - 1) * wei_size;
        }

        float *diff_bia = nullptr;
        if (jcp.with_bias) {
            const int bias_size = jcp.ngroups * jcp.nb_load * jcp.oc_block;
            if (jcp.bia_dt == data_type::bf16) {
                diff_bia = bia_reduction + (ithr_mb)*bias_size;
            } else {
                diff_bia = ithr_mb == 0
                        ? (float *)diff_bias
                        : (float *)bia_reduction + (ithr_mb - 1) * bias_size;
            }
        }

        int sp_b_step = 0;
        for (int mb_sp_b = mb_sp_b_start; mb_sp_b < mb_sp_b_end;
                mb_sp_b += sp_b_step) {
            int img {0}, sp_b {0};
            nd_iterator_init(mb_sp_b, img, jcp.mb, sp_b, sp_nb);
            sp_b_step = step(jcp.nb_reduce_blocking,
                    nstl::min(sp_nb - sp_b, mb_sp_b_end - mb_sp_b),
                    jcp.nb_reduce_blocking_max);

            for (int g = g_start; g < g_end; ++g) {
                int load_step = 0;
                int bcast_step = 0;
                for (int ic_b = ic_b_start; ic_b < ic_b_end;
                        ic_b += bcast_step) {
                    bcast_step = step(nb_ic_blocking, ic_b_end - ic_b,
                            jcp.nb_bcast_blocking_max);
                    for (int oc_b = oc_b_start; oc_b < oc_b_end;
                            oc_b += load_step) {
                        load_step = step(nb_oc_blocking, oc_b_end - oc_b,
                                jcp.nb_load_blocking_max);

                        float *store_to;

                        const size_t off
                                = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                        store_to = diff_wei + off;

                        const bool is_src_layout_nxc
                                = utils::one_of(jcp.src_tag, format_tag::nwc,
                                        format_tag::nhwc, format_tag::ndhwc);
                        const int ic_off_idx = is_src_layout_nxc
                                ? g * jcp.ic + ic_b * jcp.ic_block
                                : g * nb_oc + ic_b;
                        const src_data_t *diff_src
                                = &src[src_d.blk_off(img, ic_off_idx)];
                        const int oc_off_idx = is_ddst_layout_nxc
                                ? g * jcp.oc + oc_b * jcp.oc_block
                                : g * nb_oc + oc_b;
                        const diff_dst_data_t *pdiff_dst
                                = &diff_dst[diff_dst_d.blk_off(
                                        img, oc_off_idx)];
                        const src_data_t *local_src = diff_src;

                        auto p = jit_1x1_conv_call_s();
                        auto rp = rtus_driver_t<avx512_common>::call_params_t();

                        p.output_stride = utils::rnd_up(jcp.ic, jcp.oc_block)
                                * jcp.oc_block * jcp.typesize_out;

                        p.load_dim = this_block_size(oc_b * jcp.oc_block,
                                jcp.oc, load_step * jcp.oc_block);

                        p.bcast_dim = this_block_size(ic_b * jcp.ic_block,
                                jcp.ic, bcast_step * jcp.ic_block);
                        rp.icb = p.bcast_dim;
                        p.output_data = store_to;

                        p.reduce_dim = sp_b_step * jcp.reduce_block;
                        if (!jcp.uses_permw_transposition)
                            p.reduce_dim = nstl::min(p.reduce_dim,
                                    (size_t)jcp.reduce_dim
                                            - sp_b * jcp.reduce_block);

                        rp.os = p.reduce_dim;

                        p.first_last_flag = 0
                                | (mb_sp_b == mb_sp_b_start ? FLAG_REDUCE_FIRST
                                                            : 0)
                                | (ic_b == 0 ? FLAG_COMPUTE_BIAS : 0);

                        int sp = sp_b * jcp.reduce_block;
                        int oc_mult = is_ddst_layout_nxc ? jcp.ngroups * jcp.oc
                                                         : jcp.oc_block;
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
                            int ic_mult = is_src_layout_nxc
                                    ? jcp.ngroups * jcp.ic
                                    : jcp.ic_block;
                            p.bcast_data = local_src + sp * ic_mult;
                        }
                        if (!jcp.uses_permw_transposition) {
                            bf16_support::jit_call_t ptr;
                            ptr.nelems = p.reduce_dim;
                            int thr_src_block_size = rnd_up(jcp.reduce_dim, 2)
                                    * jcp.ic_block * jcp.nb_bcast_blocking_max;
                            src_data_t *tr_src
                                    = &tr_src_buffer[ithr * thr_src_block_size];
                            for (int bs = 0; bs < bcast_step; bs++) {
                                size_t src_off = bs * jcp.ic_block
                                        * (is_src_layout_nxc ? 1
                                                             : jcp.reduce_dim);
                                size_t src_tr_off = bs
                                        * rnd_up(jcp.reduce_dim, 2)
                                        * jcp.ic_block;
                                src_data_t *curr_inp = &(
                                        (src_data_t *)p.bcast_data)[src_off];
                                src_data_t *curr_out = &tr_src[src_tr_off];
                                int ch_work = nstl::min<int>(
                                        p.bcast_dim - bs * jcp.bcast_block, 16);
                                assert(ch_work <= 16);
                                ptr.mask = (1 << ch_work) - 1;
                                ptr.inp = (void *)curr_inp;
                                ptr.out = (void *)curr_out;
                                if (is_src_layout_nxc)
                                    (*tr_reorder_nhwc_src_)(&ptr);
                                else
                                    (*tr_reorder_)(&ptr);
                            }

                            p.bcast_data = (void *)tr_src;
                            int thr_dst_block_size = rnd_up(jcp.reduce_dim, 2)
                                    * jcp.oc_block * jcp.nb_load_blocking_max;
                            diff_dst_data_t *tr_diff_dst = &tr_diff_buffer[ithr
                                    * thr_dst_block_size];
                            for (int ls = 0; ls < load_step; ls++) {
                                size_t ddst_off = ls * jcp.oc_block
                                        * (is_ddst_layout_nxc ? 1 : jcp.os);
                                size_t ddst_tr_off = ls
                                        * rnd_up(jcp.reduce_dim, 2)
                                        * jcp.oc_block;
                                diff_dst_data_t *curr_inp
                                        = &((diff_dst_data_t *)
                                                        p.load_data)[ddst_off];
                                diff_dst_data_t *curr_out
                                        = &tr_diff_dst[ddst_tr_off];
                                int ch_work = nstl::min<int>(
                                        p.load_dim - ls * jcp.load_block, 16);
                                ptr.mask = (1 << ch_work) - 1;
                                ptr.inp = (void *)curr_inp;
                                ptr.out = (void *)curr_out;
                                if (is_ddst_layout_nxc)
                                    (*tr_reorder_nhwc_ddst_)(&ptr);
                                else
                                    (*tr_reorder_)(&ptr);
                            }
                            p.load_data = (void *)tr_diff_dst;
                        } //if (!jcp.uses_permw_transposition)

                        p.bias_data = diff_bia
                                ? &diff_bia[oc_off_idx
                                        * (is_ddst_layout_nxc ? 1
                                                              : jcp.oc_block)]
                                : nullptr;
                        (*kernel_)(&p);
                    }
                }
            }
        }
    };

    auto ker_reduce_and_convert_diff_wei_bia = [&](const int ithr,
                                                       const int nthr) {
        assert(nthr == jcp.nthr);

        const int ithr_ic_b = ithr % jcp.nthr_ic_b;
        const int ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
        const int ithr_g = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
        const int ithr_mb = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b / jcp.nthr_g;

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

        const int _start_nthr_mb = 1;
        const bool is_bf16_out = diff_weights_type == data_type::bf16;
        const bool is_bf16_bias
                = jcp.with_bias && jcp.bia_dt == data_type::bf16;
        /* diff_weights[:] += sum(ws_reduction_[thr_mb][:]) */
        if (jcp.nthr_mb > _start_nthr_mb) {
            if (dnnl_thr_syncable())
                simple_barrier::barrier(&reduction_barrier, jcp.nthr);
            const int work = g_work * oc_b_work * ic_b_work;
            int start {0}, end {0};
            balance211(work, jcp.nthr_mb, ithr_mb, start, end);
            if (start == end) return;

            for (int thr_mb = _start_nthr_mb; thr_mb < jcp.nthr_mb; ++thr_mb) {
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
                    float *wei_reduced = is_bf16_out
                            ? wei_reduction + off
                            : (float *)diff_weights + off;

                    int thr_mb_buffer_idx = is_bf16_out ? thr_mb : thr_mb - 1;
                    float *wei_to_reduce = wei_reduction
                            + thr_mb_buffer_idx * wei_size + off;
                    if (is_bf16_out && thr_mb == jcp.nthr_mb - 1)
                        // the last iteration for bfloat16 requires conversion
                        // and store to diff_weights array
                        add_floats_and_cvt_to_bfloat16(
                                (bfloat16_t *)(diff_weights + off), wei_reduced,
                                wei_to_reduce, acc_size);
                    else
                        acc_ker_->accumulate(
                                wei_reduced, wei_to_reduce, acc_size);

                    nd_iterator_jump(w, end, sub_g_start, g_work,
                            sub_oc_b_start, oc_b_work, sub_ic_b_start,
                            ic_b_work);
                }

                if (jcp.with_bias && ithr_ic_b == 0 && ic_b_work > 0
                        && ithr_mb == 0) {
                    for (int g = g_start; g < g_end; g++) {
                        float *bias_reduced
                                = is_bf16_bias ? bia_reduction : diff_bias;
                        int thr_mb_buffer_idx
                                = is_bf16_bias ? thr_mb : thr_mb - 1;
                        int bias_buf_size
                                = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block);
                        float *bias_to_reduce = bia_reduction
                                + thr_mb_buffer_idx * bias_buf_size;
                        const size_t acc_size
                                = this_block_size(oc_b_start * jcp.oc_block,
                                        jcp.oc_without_padding,
                                        (oc_b_end - oc_b_start) * jcp.oc_block);
                        int idx = g * rnd_up(jcp.oc, jcp.oc_block)
                                + oc_b_start * jcp.oc_block;
                        if (is_bf16_bias && thr_mb == jcp.nthr_mb - 1) {
                            // the last iteration for bfloat16 requires conversion and
                            // store to diff_weights array
                            int diff_bias_idx = g * jcp.oc_without_padding
                                    + oc_b_start * jcp.oc_block;
                            bfloat16_t *diff_bias_result
                                    = CTX_OUT_MEM(
                                              bfloat16_t *, DNNL_ARG_DIFF_BIAS)
                                    + diff_bias_idx;
                            add_floats_and_cvt_to_bfloat16(diff_bias_result,
                                    &bias_reduced[idx], &bias_to_reduce[idx],
                                    acc_size);
                        } else {
                            acc_ker_->accumulate(&bias_reduced[idx],
                                    &bias_to_reduce[idx], acc_size);
                        }
                    }
                }
            }
        } else {
            if (is_bf16_out) {
                const auto ic_work = nstl::min(jcp.ic, ic_b_end * jcp.ic_block)
                        - ic_b_start * jcp.ic_block;
                for_(int g = g_start; g < g_end; g++)
                for (int oc_b = oc_b_start; oc_b < oc_b_end; oc_b++) {
                    const size_t acc_size = (size_t)ic_work * jcp.oc_block;
                    const size_t off
                            = wht_blk_off(diff_weights_d, g, oc_b, ic_b_start);

                    cvt_float_to_bfloat16((bfloat16_t *)(diff_weights + off),
                            (const float *)(wei_reduction + off), acc_size);
                }
            }

            if (is_bf16_bias && ithr_ic_b == 0 && ic_b_work > 0) {
                for (int g = g_start; g < g_end; g++) {
                    int result_start_idx = g * jcp.oc_without_padding
                            + oc_b_start * jcp.oc_block;
                    int buffer_start_idx = g * rnd_up(jcp.oc, jcp.oc_block)
                            + oc_b_start * jcp.oc_block;
                    const size_t acc_size = nstl::min(jcp.oc_without_padding,
                                                    oc_b_end * jcp.oc_block)
                            - oc_b_start * jcp.oc_block;
                    bfloat16_t *diff_bias_result
                            = CTX_OUT_MEM(bfloat16_t *, DNNL_ARG_DIFF_BIAS)
                            + result_start_idx;
                    float *buffer = bia_reduction + buffer_start_idx;
                    cvt_float_to_bfloat16(diff_bias_result, buffer, acc_size);
                }
            }
        }
    };

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        assert(nthr == jcp.nthr);
        ker(ithr, jcp.nthr);
        if (dnnl_thr_syncable())
            ker_reduce_and_convert_diff_wei_bia(ithr, jcp.nthr);
    });

    if (!dnnl_thr_syncable()) {
        parallel(jcp.nthr, [&](const int ithr, const int nthr) {
            assert(nthr == jcp.nthr);
            ker_reduce_and_convert_diff_wei_bia(ithr, jcp.nthr);
        });
    }

    if (pd()->jcp_.bia_dt == data_type::f32
            && jcp.oc_without_padding % jcp.oc_block) {
        auto diff_bias_in = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);
        utils::array_copy(diff_bias_in, diff_bias, jcp.oc_without_padding);
    }
}

template struct jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<
        data_type::f32>;
template struct jit_avx512_core_bf16_1x1_convolution_bwd_weights_t<
        data_type::bf16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
