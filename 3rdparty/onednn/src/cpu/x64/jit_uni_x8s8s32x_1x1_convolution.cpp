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

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_uni_x8s8s32x_1x1_convolution.hpp"

#include "cpu/cpu_primitive.hpp"

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
template <cpu_isa_t isa, data_type_t src_type, data_type_t dst_type>
status_t jit_uni_x8s8s32x_1x1_convolution_fwd_t<isa, src_type,
        dst_type>::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    auto weights_dw = CTX_IN_MEM(
            const wei_data_t *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
    auto bias_dw = CTX_IN_MEM(
            const char *, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);

    DEFINE_ZERO_POINTS_BUFFER(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINTS_BUFFER(dst_zero_point, DNNL_ARG_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();

    if (pd()->jcp_.signed_input) {
        auto local_scales
                = scratchpad.template get<float>(key_conv_adjusted_scales);
        auto scales = pd()->attr()->output_scales_.scales_;
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, scales[0] * factor, 8);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = scales[c] * factor;
        }
    }

    if (pd()->jcp_.with_dw_conv) {
        auto jcp_dw = pd()->jcp_dw_;
        if (jcp_dw->signed_input && jcp_dw->ver != ver_vnni) {
            memory_tracking::grantor_t dw_scratchpad(
                    scratchpad, memory_tracking::names::prefix_fusion);
            auto attr_dw = pd()->dw_conv_pd_->attr();

            auto local_scales = dw_scratchpad.template get<float>(
                    key_conv_adjusted_scales);
            auto scales = attr_dw->output_scales_.scales_;
            size_t count = attr_dw->output_scales_.count_;
            float factor = 1.f / jcp_dw->wei_adj_scale;
            if (count == 1) {
                utils::array_set(
                        local_scales, scales[0] * factor, pd()->jcp_.ic_block);
            } else {
                for (size_t c = 0; c < count; c++)
                    local_scales[c] = scales[c] * factor;
            }
        }
    }
    parallel(0, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, weights_dw, bias_dw,
                dst, src_zero_point, dst_zero_point, scratchpad);
    });
    return status::success;
}

template <cpu_isa_t isa, data_type_t src_type, data_type_t dst_type>
void jit_uni_x8s8s32x_1x1_convolution_fwd_t<isa, src_type,
        dst_type>::execute_forward_thr(const int ithr, const int nthr,
        const src_data_t *src, const wei_data_t *weights, const char *bias,
        const wei_data_t *weights_dw, const char *bias_dw, dst_data_t *dst,
        const int32_t *src_zero_point, const int32_t *dst_zero_point,
        const memory_tracking::grantor_t &scratchpad) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper dw_weights_d(
            pd()->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS));

    const auto &jcp = pd()->jcp_;

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;

    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<src_data_t>(key_conv_rtus_space)
            : nullptr;

    auto local_scales = scratchpad.get<float>(key_conv_adjusted_scales);

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    const int ndims = dst_d.ndims();
    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    float *oscales {nullptr};
    if (jcp.signed_input && jcp.ver != ver_vnni)
        oscales = scratchpad.get<float>(key_conv_adjusted_scales);
    else
        oscales = pd()->attr()->output_scales_.scales_;

    auto offset = weights_d.size() - weights_d.additional_buffer_size();
    wei_data_t *w = const_cast<wei_data_t *>(weights);
    const int32_t *compensation = (jcp.signed_input)
            ? reinterpret_cast<int32_t *>(w + offset)
            : nullptr;
    const int32_t *zp_compensation = jcp.src_zero_point
            ? reinterpret_cast<int32_t *>(&w[offset])
                    + (jcp.signed_input ? jcp.ngroups * jcp.oc : 0)
            : nullptr;

    auto p = jit_1x1_conv_call_s();

    auto rp = typename rtus_driver_t<isa>::call_params_t();
    const int nb_oc = jcp.nb_load;
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
    const auto jcp_dw = pd()->jcp_dw_;
    const auto &dw_pd = pd()->dw_conv_pd_;
    memory_tracking::grantor_t dw_scratchpad(
            scratchpad, memory_tracking::names::prefix_fusion);

    const size_t dw_bia_dt_size = jcp_dw && jcp_dw->with_bias
            ? types::data_type_size(dw_pd->desc()->bias_desc.data_type)
            : 0;

    int32_t *compensation_dw {nullptr};
    float *dw_oscales {nullptr};
    if (jcp.with_dw_conv) {
        offset = dw_weights_d.size() - dw_weights_d.additional_buffer_size();
        w = const_cast<wei_data_t *>(weights_dw);
        compensation_dw = (jcp_dw->signed_input)
                ? reinterpret_cast<int32_t *>(w + offset)
                : nullptr;
        if (jcp_dw->signed_input && jcp_dw->ver != ver_vnni)
            dw_oscales = dw_scratchpad.get<float>(key_conv_adjusted_scales);
        else
            dw_oscales = dw_pd->attr()->output_scales_.scales_;
    }

    dst_data_t *pbuf {nullptr};
    size_t row_offset {};
    const int nb_buffer = jcp.nb_load_blocking;
    std::vector<dst_data_t *> addrs;
    // End

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

    auto init_load = [&](int ocb, int ocb_end, int &load_step) {
        load_step = step(nb_load_blocking, ocb_end - ocb, nb_load_blocking_max);
        p.load_dim = this_block_size(ocb * jcp.oc_block, ocb_end * jcp.oc_block,
                load_step * jcp.oc_block);

        if (ocb + load_step >= nb_oc)
            p.first_last_flag |= FLAG_OC_LAST;
        else
            p.first_last_flag &= ~FLAG_OC_LAST;
    };

    auto init_reduce = [&]() {
        p.reduce_dim = this_block_size(
                0, jcp.ic_without_padding, jcp.ic_without_padding);
        rp.icb = p.reduce_dim;
    };

    auto ker_1x1 = [&](int ocb, int ocb_start, int n, int g, int od, int oh,
                           int ow, int id, int ih, int iw) {
        const int icb = 0; // Start from the first IC block
        const int _ocb = g * nb_oc + ocb;
        const int _icb = g;

        p.output_data = jcp.with_dw_conv
                ? pbuf + (oh % jcp_dw->kh) * row_offset
                : &dst[data_blk_off(dst_d, n, _ocb * jcp.oc_block, od, oh, ow)];
        p.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];
        p.bias_data = &bias[_ocb * jcp.oc_block * bia_dt_size];
        p.compensation = (jcp.signed_input) ? &compensation[_ocb * jcp.oc_block]
                                            : nullptr;
        p.zp_compensation = jcp.src_zero_point
                ? zp_compensation + _ocb * jcp.oc_block
                : nullptr;
        p.src_zero_point = jcp.src_zero_point ? src_zero_point : nullptr;
        p.dst_zero_point = jcp.dst_zero_point ? dst_zero_point : nullptr;
        p.scales = (jcp.signed_input)
                ? &local_scales[jcp.is_oc_scale * _ocb * jcp.oc_block]
                : &oscales[jcp.is_oc_scale * _ocb * jcp.oc_block];
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_
                    + _icb * jcp.is * jcp.ic_block;
            if (ocb == ocb_start) {
                rp.src = src
                        + data_blk_off(
                                src_d, n, _icb * jcp.ic_block, id, ih, iw);
                (*rtus_driver_)(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src
                    + data_blk_off(src_d, n, _icb * jcp.ic_block, id, ih, iw);

        (*kernel_)(&p);
    };

    auto conv_1x1 = [&](int bcast_start, int bcast_end, int ocb_start,
                            int ocb_end) {
        if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;
        if (jcp.loop_order == loop_rlb) {
            init_reduce();
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
                    ker_1x1(ocb, ocb_start, n, g, od, oh, ow, id, ih, iw);
                    iwork += bcast_step;
                }
                ocb += load_step;
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
                    init_reduce();
                    ker_1x1(ocb, ocb_start, n, g, od, oh, ow, id, ih, iw);
                    iwork += bcast_step;
                }
                ocb += load_step;
            }
        } else if (jcp.loop_order == loop_rbl) {
            init_reduce();
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
                    ker_1x1(ocb, ocb_start, n, g, od, oh, ow, id, ih, iw);
                    ocb += load_step;
                }
                iwork += bcast_step;
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
                    init_reduce();
                    ker_1x1(ocb, ocb_start, n, g, od, oh, ow, id, ih, iw);
                    ocb += load_step;
                }
                iwork += bcast_step;
            }
        } else {
            assert(!"unsupported loop order");
        }
    };

    auto ker_dw = [&](int n, int ocb_start, int load_step, int &dw_oh) {
        int oh_1x1 = dw_oh * jcp_dw->stride_h - jcp_dw->t_pad;
        int oh_1x1_begin = nstl::max(oh_1x1, 0);

        for (int i = 0; i < jcp_dw->kh; ++i)
            addrs[i] = pbuf + ((oh_1x1_begin++) % jcp_dw->kh) * row_offset;

        const auto ocb_end = ocb_start + load_step;
        const size_t src_ch_stride = jcp_dw->nb_ch_blocking * jcp_dw->ch_block;
        auto par_conv_dw = jit_conv_call_s();

        par_conv_dw.t_overflow = nstl::min(jcp_dw->kh, nstl::max(0, -oh_1x1));
        par_conv_dw.b_overflow = nstl::min(
                jcp_dw->kh, nstl::max(0, oh_1x1 - jcp.oh + jcp_dw->kh));
        par_conv_dw.kh_padding = nstl::max<int>(0,
                jcp_dw->kh - par_conv_dw.t_overflow - par_conv_dw.b_overflow);

        const size_t dst_offset = n * jcp_dw->ngroups * jcp_dw->oh * jcp_dw->ow
                + dw_oh * jcp_dw->ow * jcp_dw->ngroups;

        const auto wht_h_stride = dw_weights_d.blk_off(0, 0, 0, 1);
        const auto wei_stride = (!jcp_dw->signed_input) * par_conv_dw.t_overflow
                * wht_h_stride;
        for (int ocb = ocb_start; ocb < ocb_end;
                ocb += jcp_dw->nb_ch_blocking) {

            par_conv_dw.src = addrs.data();
            par_conv_dw.dst = &dst[(dst_offset + jcp_dw->ch_block * ocb)
                    * jcp_dw->typesize_out];

            par_conv_dw.filt
                    = weights_dw + dw_weights_d.blk_off(ocb, 0) + wei_stride;
            par_conv_dw.bias
                    = &bias_dw[ocb * jcp_dw->ch_block * dw_bia_dt_size];
            par_conv_dw.ur_w = (size_t)(jcp_dw->ow);
            par_conv_dw.owb = jcp_dw->ow;
            par_conv_dw.oc_blocks = ocb;
            par_conv_dw.compensation = compensation_dw
                    ? &compensation_dw[ocb * jcp_dw->ch_block]
                    : nullptr;
            par_conv_dw.scales = dw_oscales
                    ? &dw_oscales[jcp_dw->is_oc_scale * ocb * jcp_dw->ch_block]
                    : nullptr;

            (*kernel_dw_)(&par_conv_dw);

            for (int i = 0; i < jcp_dw->kh; ++i)
                addrs[i] += src_ch_stride;
        }
    };

    auto conv_dw = [&]() {
        auto &jcp_dw = pd()->jcp_dw_;
        auto dw_conv_buffer
                = dw_scratchpad.get<dst_data_t>(key_fusion_inout_buffer);

        const auto dw_conv_buffer_size_
                = (size_t)jcp_dw->kh * jcp.ow * nb_buffer * jcp.oc_block;
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
        int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
                jcp.nb_load / jcp.nb_load_chunk, ocb_start, ocb_end,
                jcp.load_grp_count);
        if (jcp.nb_load_chunk > 1) {
            ocb_start *= jcp.nb_load_chunk;
            ocb_end *= jcp.nb_load_chunk;
        }
        conv_1x1(bcast_start, bcast_end, ocb_start, ocb_end);
    }
}

using namespace data_type;

template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, u8, u8>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, s8, u8>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, u8, s8>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, s8, s8>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, u8, s32>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, s8, s32>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, u8, f32>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<avx2, s8, f32>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, u8, u8>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, s8, u8>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, u8, s8>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, s8, s8>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, u8, s32>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, s8, s32>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, u8, f32>;
template struct jit_uni_x8s8s32x_1x1_convolution_fwd_t<sse41, s8, f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
