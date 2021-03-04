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

#include "cpu/x64/jit_avx512_common_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using jit_conv_ker_t = void (*)(jit_conv_call_s *);

#define PIPELINE(field) \
    do { \
        p.field = p.field##_prf; \
        p.field##_prf = field; \
    } while (0)

inline void jit_conv_ker_pipeline(const jit_conv_ker_t ker, jit_conv_call_s &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int reduce_work, int load_work) {
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    // non-positive value of kh_padding is allowed, in this case kernel must
    // skip computation part and initialize output by zeroes
    PIPELINE(kh_padding);
    PIPELINE(reduce_work);
    PIPELINE(load_work);

    if (p.src) ker(&p);
}
// The special case for the driver with ow-parallelization (FWD)
inline void jit_conv_ker_pipeline_ow_thr(const jit_conv_ker_t ker,
        jit_conv_call_s &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, int owb, int reduce_work,
        int load_work, int flags) {
    PIPELINE(owb);
    PIPELINE(flags);
    jit_conv_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            reduce_work, load_work);
}
// The special case for the driver with iw-parallelization (BWD)
inline void jit_conv_ker_pipeline_iw_thr(const jit_conv_ker_t ker,
        jit_conv_call_s &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, int iwb, int reduce_work,
        int load_work) {
    PIPELINE(iwb);

    jit_conv_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            reduce_work, load_work);
}

inline void jit_conv_3d_ker_pipeline(const jit_conv_ker_t ker,
        jit_conv_call_s &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, int kd_padding,
        int reduce_work, int load_work) {
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    // non-positive value of both kd_padding and kh_padding is allowed, in this
    // case kernel must skip computation part and initialize output by zeroes
    PIPELINE(kh_padding);
    PIPELINE(kd_padding);
    PIPELINE(reduce_work);
    PIPELINE(load_work);

    if (p.src) ker(&p);
}
// The special case for the driver with ow-parallelization (FWD)
// TODO: implement it for BWD_D and BWD_W too
inline void jit_conv_3d_ker_pipeline_ow_thr(const jit_conv_ker_t ker,
        jit_conv_call_s &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, int kd_padding, int owb,
        int reduce_work, int load_work, int flags) {
    PIPELINE(owb);
    PIPELINE(flags);

    jit_conv_3d_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            kd_padding, reduce_work, load_work);
}

inline void jit_conv_ker_pipeline_bwd_w(const jit_conv_ker_t ker,
        jit_conv_call_s &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, size_t reduce_work,
        size_t load_work) {
    jit_conv_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            reduce_work, load_work);
}

void jit_conv_2d_ker_bwd_w_pipeline(const jit_conv_ker_t ker,
        jit_conv_call_s &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int os_index_begin, int os_index_end,
        int kh_padding /* kh_work_size */, size_t kh_offset, size_t reduce_work,
        size_t load_work) {
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    PIPELINE(os_index_begin);
    PIPELINE(os_index_end);
    // non-positive value of kh_padding is allowed, in this case kernel must
    // skip kw loop computation and initialize output by zeroes
    PIPELINE(kh_padding);
    PIPELINE(kh_offset);
    PIPELINE(reduce_work);
    PIPELINE(load_work);

    if (p.src) ker(&p);
}

void jit_conv_3d_ker_bwd_w_pipeline(const jit_conv_ker_t ker,
        jit_conv_call_s &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int os_index_begin, int os_index_end,
        int kd_padding /* kd_work_size */, size_t kd_offset, size_t reduce_work,
        size_t load_work) {
    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    PIPELINE(os_index_begin);
    PIPELINE(os_index_end);
    // non-positive value of kd_padding is allowed, in this case kernel must
    // skip kh loop computation and initialize output by zeroes
    PIPELINE(kd_padding);
    PIPELINE(kd_offset);
    PIPELINE(reduce_work);
    PIPELINE(load_work);

    if (p.src) ker(&p);
}
#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t<src_type, wei_type,
        dst_type>::prepare_padded_bias(const dst_data_t *&bias,
        const memory_tracking::grantor_t &scratchpad) const {
    if (!pd()->wants_padded_bias()) return;

    auto padded_bias
            = scratchpad.template get<dst_data_t>(key_conv_padded_bias);
    utils::array_copy(padded_bias, bias, pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + pd()->jcp_.oc_without_padding, (dst_data_t)0,
            pd()->jcp_.oc - pd()->jcp_.oc_without_padding);
    bias = padded_bias;
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_1d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.nb_ow;
    int nthr = jcp.aligned_threads;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t src_c_stride = src_d.blk_off(0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n {0}, gg {0}, occ {0}, owb {0};

            if (jcp.loop_order == loop_cwgn) {
                int dummy {0};
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == loop_gncw) {
                int dummy {0};
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, dummy, 1);
            } else if (jcp.loop_order == loop_nhwcg) {
                nd_iterator_init(start, n, jcp.mb, owb, jcp.nb_ow, occ,
                        oc_chunks, gg, nb_groups);
            } else {
                assert(!"unsupported loop order");
            }

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g = gg * g_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                int ow_s = owb * jcp.ow_block;
                int iw_s = ow_s * jcp.stride_w;
                const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::nwc;
                const int oc_off_idx = is_dst_layout_nxc
                        ? g * jcp.oc + ocb * jcp.oc_block
                        : g_ocb;
                auto dst_w = dst + dst_d.blk_off(n, oc_off_idx, ow_s);
                const bool is_src_layout_nxc = jcp.src_tag == format_tag::nwc;
                const int ic_off_idx = is_src_layout_nxc
                        ? g * jcp.ic + icb_l2 * jcp.ic_block
                        : g_icb + icb_l2;
                auto src_w = src + src_d.blk_off(n, ic_off_idx, iw_s);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb, icb_l2);
                auto bias_w = bias ? bias
                                + oc_off_idx
                                        * (is_dst_layout_nxc ? 1 : jcp.oc_block)
                                   : nullptr;

                int icb_step = is_src_layout_nxc ? jcp.nb_ic_L2 : 1;
                int icb_end = min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                const int oc_work = utils::this_block_size(ocb * jcp.oc_block,
                        jcp.oc, jcp.nb_oc_blocking * jcp.oc_block);
                int ic_work = icb_step * jcp.ic_block;
                for (int icb = icb_l2; icb < icb_end; icb += icb_step) {
                    int curr_nb_ic = nstl::min(icb_step, icb_end - icb);
                    int flags = 0;
                    if (icb == 0) flags |= FLAG_IC_FIRST;
                    if (icb + curr_nb_ic >= jcp.nb_ic) {
                        flags |= FLAG_IC_LAST;
                        ic_work = utils::this_block_size(icb * jcp.ic_block,
                                jcp.ic, icb_step * jcp.ic_block);
                    }
                    jit_conv_ker_pipeline_ow_thr(jit_ker, par_conv, src_w,
                            dst_w, wht_w, bias_w, icb, 1, owb, ic_work, oc_work,
                            flags);

                    src_w += src_c_stride;
                    wht_w += wht_ic_stride;
                }
                if (jcp.loop_order == loop_cwgn) {
                    int dummy {0};
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            gg, nb_groups, n, jcp.mb, dummy, 1);
                } else if (jcp.loop_order == loop_gncw) {
                    int dummy {0};
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                            oc_chunks, owb, jcp.nb_ow, dummy, 1);
                } else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, owb, jcp.nb_ow, occ, oc_chunks,
                            gg, nb_groups);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_ker_pipeline_ow_thr(
                jit_ker, par_conv, src, dst, weights, bias, 0, 0, 0, 0, 0, 0);
    });
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_2d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh * jcp.nb_ow;
    int nthr = jcp.aligned_threads;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t src_c_stride = src_d.blk_off(0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n {0}, gg {0}, occ {0}, oh_s {0}, owb {0};

            if (jcp.loop_order == loop_cwgn)
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_nhwcg)
                nd_iterator_init(start, n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow,
                        occ, oc_chunks, gg, nb_groups);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g = gg * g_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                int work_rem = end - start;

                int ow_s = owb * jcp.ow_block;
                int iw_s = ow_s * jcp.stride_w;
                int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
                if (jcp.loop_order == loop_nhwcg)
                    oh_e = oh_s + 1; //step instead

                for (int oh_b = oh_s; oh_b < oh_e; oh_b += jcp.h_blocking) {
                    int ih_b = -jcp.t_pad + oh_b * jcp.stride_h;
                    const bool is_dst_layout_nxc
                            = jcp.dst_tag == format_tag::nhwc;
                    const int oc_off_idx = is_dst_layout_nxc
                            ? g * jcp.oc + ocb * jcp.oc_block
                            : g_ocb;
                    auto dst_w = dst + dst_d.blk_off(n, oc_off_idx, oh_b, ow_s);
                    const bool is_src_layout_nxc
                            = jcp.src_tag == format_tag::nhwc;
                    const int ic_off_idx = is_src_layout_nxc
                            ? g * jcp.ic + icb_l2 * jcp.ic_block
                            : g_icb + icb_l2;
                    auto src_w = src + src_d.blk_off(n, ic_off_idx, ih_b, iw_s);
                    auto wht_w
                            = weights + wht_blk_off(weights_d, g, ocb, icb_l2);

                    int icb_step = is_src_layout_nxc ? jcp.nb_ic_L2 : 1;
                    int icb_end = min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                    auto bias_w = bias ? bias
                                    + oc_off_idx
                                            * (is_dst_layout_nxc ? 1
                                                                 : jcp.oc_block)
                                       : nullptr;
                    const int oc_work
                            = utils::this_block_size(ocb * jcp.oc_block, jcp.oc,
                                    jcp.nb_oc_blocking * jcp.oc_block);
                    int ic_work = icb_step * jcp.ic_block;
                    for (int icb = icb_l2; icb < icb_end; icb += icb_step) {
                        int curr_nb_ic = nstl::min(icb_step, icb_end - icb);
                        int flags = 0;
                        if (icb == 0) flags |= FLAG_IC_FIRST;
                        if (icb + curr_nb_ic >= jcp.nb_ic) {
                            flags |= FLAG_IC_LAST;
                            ic_work = utils::this_block_size(icb * jcp.ic_block,
                                    jcp.ic, icb_step * jcp.ic_block);
                        }
                        auto src_c = src_w;
                        auto dst_c = dst_w;
                        for (int oj = oh_b, ij = ih_b;
                                oj < min(oh_e, oh_b + jcp.h_blocking);
                                ++oj, ij += jcp.stride_h) {
                            int dilate_h = jcp.dilate_h + 1;
                            int i_t_overflow = div_up(max(0, -ij), dilate_h);
                            int i_b_overflow = div_up(
                                    max(0,
                                            ij - jcp.ih
                                                    + (jcp.kh - 1) * dilate_h
                                                    + 1),
                                    dilate_h);
                            int kh_padding = nstl::max(
                                    0, jcp.kh - i_t_overflow - i_b_overflow);

                            auto aux_src = src_c
                                    + i_t_overflow * dilate_h * src_h_stride;
                            auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                            jit_conv_ker_pipeline_ow_thr(jit_ker, par_conv,
                                    aux_src, dst_c, aux_wht, bias_w, icb,
                                    kh_padding, owb, ic_work, oc_work, flags);

                            src_c += src_h_stride * jcp.stride_h;
                            dst_c += dst_h_stride;
                        }
                        src_w += src_c_stride;
                        wht_w += wht_ic_stride;
                    }
                }

                if (jcp.loop_order == loop_cwgn)
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            gg, nb_groups, n, jcp.mb, oh_s, jcp.oh);
                else if (jcp.loop_order == loop_gncw)
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                            oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
                else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow,
                            occ, oc_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_ker_pipeline_ow_thr(
                jit_ker, par_conv, src, dst, weights, bias, 0, 0, 0, 0, 0, 0);
    });
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_3d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);

    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount
            = jcp.mb * nb_groups * oc_chunks * jcp.od * jcp.oh * jcp.nb_ow;
    int nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t src_d_stride = src_d.blk_off(0, 0, 1);
        size_t src_h_stride = src_d.blk_off(0, 0, 0, 1);
        size_t src_c_stride = src_d.blk_off(0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 0, 1);
        size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (int icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n {0}, gg {0}, occ {0}, oh_s {0}, od_s {0}, owb {0};

            if (jcp.loop_order == loop_cwgn)
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_nhwcg)
                nd_iterator_init(start, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh,
                        owb, jcp.nb_ow, occ, oc_chunks, gg, nb_groups);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g = gg * g_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                int work_rem = end - start;
                int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
                int ow_s = owb * jcp.ow_block;
                int iw_s = ow_s * jcp.stride_w;
                int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
                if (jcp.loop_order == loop_nhwcg)
                    oh_e = oh_s + 1; //step instead

                int id_s = -jcp.f_pad + od_s * jcp.stride_d;

                int dilate_d = jcp.dilate_d + 1;
                int d_t_overflow = div_up(max(0, -id_s), dilate_d);
                int d_b_overflow = div_up(
                        max(0, id_s - jcp.id + (jcp.kd - 1) * dilate_d + 1),
                        dilate_d);
                int kd_padding
                        = nstl::max(0, jcp.kd - d_t_overflow - d_b_overflow);
                const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
                const int oc_off_idx = is_dst_layout_nxc
                        ? g * jcp.oc + ocb * jcp.oc_block
                        : g_ocb;
                auto dst_w
                        = dst + dst_d.blk_off(n, oc_off_idx, od_s, oh_s, ow_s);
                const bool is_src_layout_nxc = jcp.src_tag == format_tag::ndhwc;
                const int ic_off_idx = is_src_layout_nxc
                        ? g * jcp.ic + icb_l2 * jcp.ic_block
                        : g_icb + icb_l2;
                auto src_w = src
                        + src_d.blk_off(n, ic_off_idx, id_s, ih_s, iw_s)
                        + d_t_overflow * dilate_d * src_d_stride;
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb, icb_l2)
                        + d_t_overflow * wht_d_stride;
                auto bias_w = bias ? bias
                                + oc_off_idx
                                        * (is_dst_layout_nxc ? 1 : jcp.oc_block)
                                   : nullptr;

                const int icb_step = is_src_layout_nxc ? jcp.nb_ic_L2 : 1;
                int icb_end = min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                const int oc_work = utils::this_block_size(ocb * jcp.oc_block,
                        jcp.oc, jcp.nb_oc_blocking * jcp.oc_block);
                int ic_work = icb_step * jcp.ic_block;
                for (int icb = icb_l2; icb < icb_end; icb += icb_step) {
                    int curr_nb_ic = nstl::min(icb_step, icb_end - icb);
                    int flags = 0;
                    if (icb == 0) flags |= FLAG_IC_FIRST;
                    if (icb + curr_nb_ic >= jcp.nb_ic) {
                        flags |= FLAG_IC_LAST;
                        ic_work = utils::this_block_size(icb * jcp.ic_block,
                                jcp.ic, icb_step * jcp.ic_block);
                    }
                    auto src_c = src_w;
                    auto dst_c = dst_w;
                    for (int oj = oh_s, ij = ih_s; oj < oh_e;
                            ++oj, ij += jcp.stride_h) {
                        int dilate_h = jcp.dilate_h + 1;
                        int i_t_overflow = div_up(max(0, -ij), dilate_h);
                        int i_b_overflow = div_up(
                                max(0,
                                        ij - jcp.ih + (jcp.kh - 1) * dilate_h
                                                + 1),
                                dilate_h);
                        int kh_padding = nstl::max(
                                0, jcp.kh - i_t_overflow - i_b_overflow);
                        jit_conv_3d_ker_pipeline_ow_thr(jit_ker, par_conv,
                                src_c + i_t_overflow * dilate_h * src_h_stride,
                                dst_c, wht_w + i_t_overflow * wht_h_stride,
                                bias_w, icb, kh_padding, kd_padding, owb,
                                ic_work, oc_work, flags);

                        src_c += src_h_stride * jcp.stride_h;
                        dst_c += dst_h_stride;
                    }
                    src_w += src_c_stride;
                    wht_w += wht_ic_stride;
                }

                if (jcp.loop_order == loop_cwgn)
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            gg, nb_groups, n, jcp.mb, od_s, jcp.od, oh_s,
                            jcp.oh);
                else if (jcp.loop_order == loop_gncw)
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                            oc_chunks, owb, jcp.nb_ow, od_s, jcp.od, oh_s,
                            jcp.oh);
                else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh, owb,
                            jcp.nb_ow, occ, oc_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_3d_ker_pipeline_ow_thr(jit_ker, par_conv, src, dst, weights,
                bias, 0, 0, 0, 0, 0, 0, 0);
    });
}

template struct jit_avx512_common_convolution_fwd_t<data_type::f32>;

template <data_type_t diff_dst_type, data_type_t wei_type,
        data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
        diff_src_type>::execute_backward_data_1d(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();

    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = nb_groups * jcp.mb * ic_chunks * jcp.nb_iw;
    int nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n {0}, gg {0}, icc {0}, iwb {0};
            if (jcp.loop_order == loop_cwgn) {
                int dummy {0};
                nd_iterator_init(start, icc, ic_chunks, iwb, jcp.nb_iw, gg,
                        nb_groups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == loop_gncw) {
                int dummy {0};
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, iwb, jcp.nb_iw, dummy, 1);
            } else if (jcp.loop_order == loop_nhwcg) {
                nd_iterator_init(start, n, jcp.mb, iwb, jcp.nb_iw, icc,
                        ic_chunks, gg, nb_groups);
            } else {
                assert(!"unsupported loop order");
            }

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g = gg * g_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;
                int iw_s = iwb * jcp.iw_block;
                int ow_s = iw_s / jcp.stride_w;

                const bool is_dsrc_layout_nxc = jcp.src_tag == format_tag::nwc;
                const int ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g_icb;
                auto diff_src_w
                        = diff_src + diff_src_d.blk_off(n, ic_off_idx, iw_s);
                const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::nwc;
                const int oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb_l2 * jcp.oc_block
                        : g_ocb + ocb_l2;
                auto diff_dst_w
                        = diff_dst + diff_dst_d.blk_off(n, oc_off_idx, ow_s);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb);

                int ocb_step = is_ddst_layout_nxc ? jcp.nb_oc_L2 : 1;
                int ocb_end = min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2);
                const int load_work = utils::this_block_size(icb * jcp.ic_block,
                        jcp.ic, jcp.nb_ic_blocking * jcp.ic_block);
                int reduce_work = ocb_step * jcp.oc_block;
                for (int ocb = ocb_l2; ocb < ocb_end; ocb += ocb_step) {
                    int curr_nb_oc = nstl::min(ocb_step, ocb_end - ocb);
                    if (ocb + curr_nb_oc >= jcp.nb_oc) {
                        reduce_work = utils::this_block_size(ocb * jcp.oc_block,
                                jcp.oc, ocb_step * jcp.oc_block);
                    }

                    jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv, diff_src_w,
                            diff_dst_w, wht_w, nullptr, ocb, 1, iwb,
                            reduce_work, load_work);
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == loop_cwgn) {
                    int dummy {0};
                    nd_iterator_jump(start, end, icc, ic_chunks, iwb, jcp.nb_iw,
                            gg, nb_groups, n, jcp.mb, dummy, 1);
                } else if (jcp.loop_order == loop_gncw) {
                    int dummy {0};
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                            ic_chunks, iwb, jcp.nb_iw, dummy, 1);
                } else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, iwb, jcp.nb_iw, icc, ic_chunks,
                            gg, nb_groups);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv, diff_src, diff_dst,
                weights, nullptr, 0, 0, 0, 0, 0);
    });
}

template <data_type_t diff_dst_type, data_type_t wei_type,
        data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
        diff_src_type>::execute_backward_data_2d(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();

    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = nb_groups * jcp.mb * ic_chunks * jcp.ih * jcp.nb_iw;
    int nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        bool is_fast_path = jcp.dilate_h == 0 && jcp.stride_h == 1;

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n {0}, gg {0}, icc {0}, ih_s {0}, iwb {0};

            if (jcp.loop_order == loop_cwgn) {
                nd_iterator_init(start, icc, ic_chunks, iwb, jcp.nb_iw, gg,
                        nb_groups, n, jcp.mb, ih_s, jcp.ih);
            } else if (jcp.loop_order == loop_gncw) {
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, iwb, jcp.nb_iw, ih_s, jcp.ih);
            } else if (jcp.loop_order == loop_nhwcg) {
                nd_iterator_init(start, n, jcp.mb, ih_s, jcp.ih, iwb, jcp.nb_iw,
                        icc, ic_chunks, gg, nb_groups);
            } else
                assert(!"unsupported loop order");

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g = gg * g_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;

                int work_rem = end - start;
                int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
                if (jcp.loop_order == loop_nhwcg)
                    ih_e = ih_s + 1; //step instead
                int iw_s = iwb * jcp.iw_block;
                int ow_s = iw_s / jcp.stride_w;
                const bool is_dsrc_layout_nxc = jcp.src_tag == format_tag::nhwc;
                const int ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g_icb;
                auto diff_src_w
                        = diff_src + diff_src_d.blk_off(n, ic_off_idx, 0, iw_s);
                const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::nhwc;
                const int oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb_l2 * jcp.oc_block
                        : g_ocb + ocb_l2;
                auto diff_dst_w
                        = diff_dst + diff_dst_d.blk_off(n, oc_off_idx, 0, ow_s);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb);

                int ocb_step = is_ddst_layout_nxc ? jcp.nb_oc_L2 : 1;
                int ocb_end = min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2);
                const int load_work = utils::this_block_size(icb * jcp.ic_block,
                        jcp.ic, jcp.nb_ic_blocking * jcp.ic_block);
                int reduce_work = ocb_step * jcp.oc_block;
                for (int ocb = ocb_l2; ocb < ocb_end; ocb += ocb_step) {
                    int curr_nb_oc = nstl::min(ocb_step, ocb_end - ocb);
                    if (ocb + curr_nb_oc >= jcp.nb_oc) {
                        reduce_work = utils::this_block_size(ocb * jcp.oc_block,
                                jcp.oc, ocb_step * jcp.oc_block);
                    }
                    for (int ij = ih_s; ij < ih_e; ++ij) {
                        int oj, k_len, k_lo;
                        if (is_fast_path) { // dilate == 0 && stride == 1
                            int i_t_overflow
                                    = max(0, jcp.kh - 1 - ij - jcp.t_pad);
                            int i_b_overflow
                                    = max(0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else if (jcp.dilate_h != 0) { // stride == 1
                            int dilate_h = jcp.dilate_h + 1;
                            // Note: use div_up to account for "holes" in filter
                            int i_t_overflow
                                    = div_up(max(0,
                                                     (jcp.kh - 1) * dilate_h
                                                             - ij - jcp.t_pad),
                                            dilate_h);
                            int i_b_overflow = div_up(
                                    max(0,
                                            (jcp.kh - 1) * dilate_h + 1 - jcp.ih
                                                    + ij - jcp.b_pad),
                                    dilate_h);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                        } else { // dilate == 0
                            int i_t_overflow = max(0,
                                    (jcp.kh - 1 - ij - jcp.t_pad)
                                            / jcp.stride_h);
                            int i_b_overflow = max(0,
                                    (jcp.kh - jcp.ih + ij - jcp.b_pad)
                                            / jcp.stride_h);
                            int overflow_kh_hi = jcp.kh - 1
                                    - modulo(jcp.ih - 1 + jcp.b_pad - ij,
                                            jcp.stride_h);
                            int overflow_kh_lo
                                    = (ij + jcp.t_pad) % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                            / jcp.stride_h
                                    + 1 - i_t_overflow - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }

                        jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv,
                                diff_src_w + ij * diff_src_h_stride,
                                diff_dst_w + oj * diff_dst_h_stride,
                                wht_w + k_lo * wht_h_stride, nullptr, ocb,
                                k_len, iwb, reduce_work, load_work);
                    }
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == loop_cwgn) {
                    nd_iterator_jump(start, end, icc, ic_chunks, iwb, jcp.nb_iw,
                            gg, nb_groups, n, jcp.mb, ih_s, jcp.ih);
                } else if (jcp.loop_order == loop_gncw) {
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                            ic_chunks, iwb, jcp.nb_iw, ih_s, jcp.ih);
                } else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, ih_s, jcp.ih, iwb, jcp.nb_iw,
                            icc, ic_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv, diff_src, diff_dst,
                weights, nullptr, 0, 0, 0, 0, 0);
    });
}

template <data_type_t diff_dst_type, data_type_t wei_type,
        data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
        diff_src_type>::execute_backward_data_3d(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();

    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int g_blocking = 1;
    int nb_groups = jcp.ngroups / g_blocking;
    int work_amount = nb_groups * jcp.mb * ic_chunks * jcp.id * jcp.ih;
    int nthr = jcp.nthr;

    parallel(nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_s();
        size_t diff_src_h_stride = diff_src_d.blk_off(0, 0, 0, 1);
        size_t diff_src_d_stride = diff_src_d.blk_off(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off(0, 0, 0, 1);
        size_t diff_dst_d_stride = diff_dst_d.blk_off(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off(0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);
        size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        bool is_fast_path_d = jcp.dilate_d == 0 && jcp.stride_d == 1;
        bool is_fast_path_h = jcp.dilate_h == 0 && jcp.stride_h == 1;

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n {0}, gg {0}, icc {0}, ih_s {0}, id_s {0};
            // Input width threading is not currently implemented for 3d, so it
            // is not included in the iterator.
            if (jcp.loop_order == loop_cwgn)
                nd_iterator_init(start, icc, ic_chunks, gg, nb_groups, n,
                        jcp.mb, id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_nhwcg)
                nd_iterator_init(start, n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih,
                        icc, ic_chunks, gg, nb_groups);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g = gg * g_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;

                int work_rem = end - start;
                int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
                if (jcp.loop_order == loop_nhwcg)
                    ih_e = ih_s + 1; //step instead
                int d_len = 0, d_lo = 0, d_oj = 0;
                if (is_fast_path_d) { // dilate == 0 && stride == 1
                    int d_t_overflow = max(0, jcp.kd - 1 - id_s - jcp.f_pad);
                    int d_b_overflow
                            = max(0, jcp.kd - jcp.id + id_s - jcp.back_pad);
                    d_len = jcp.kd - d_t_overflow - d_b_overflow;
                    d_lo = d_b_overflow;
                    d_oj = id_s + jcp.f_pad - d_b_overflow;
                } else if (jcp.dilate_d != 0) { // stride == 1
                    int dilate_d = jcp.dilate_d + 1;
                    // Note: use div_up to account for "holes" in filter
                    int d_t_overflow = div_up(
                            max(0, (jcp.kd - 1) * dilate_d - id_s - jcp.f_pad),
                            dilate_d);
                    int d_b_overflow = div_up(
                            max(0,
                                    (jcp.kd - 1) * dilate_d + 1 - jcp.id + id_s
                                            - jcp.back_pad),
                            dilate_d);
                    d_len = jcp.kd - d_t_overflow - d_b_overflow;
                    d_lo = d_b_overflow;
                    d_oj = id_s + jcp.f_pad - d_b_overflow * dilate_d;
                } else { // dilate == 0
                    int d_t_overflow = max(
                            0, (jcp.kd - 1 - id_s - jcp.f_pad) / jcp.stride_d);
                    int d_b_overflow = max(0,
                            (jcp.kd - jcp.id + id_s - jcp.back_pad)
                                    / jcp.stride_d);
                    int overflow_kd_hi = jcp.kd - 1
                            - modulo(jcp.id - 1 + jcp.back_pad - id_s,
                                    jcp.stride_d);
                    int overflow_kd_lo = (id_s + jcp.f_pad) % jcp.stride_d;

                    d_len = (overflow_kd_hi - overflow_kd_lo) / jcp.stride_d + 1
                            - d_t_overflow - d_b_overflow;
                    d_lo = overflow_kd_lo + d_b_overflow * jcp.stride_d;
                    d_oj = (id_s + jcp.f_pad - d_lo) / jcp.stride_d;
                }

                const bool is_dsrc_layout_nxc
                        = jcp.src_tag == format_tag::ndhwc;
                const int ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g_icb;
                auto diff_src_w = diff_src + diff_src_d.blk_off(n, ic_off_idx)
                        + id_s * diff_src_d_stride;
                const bool is_ddst_layout_nxc
                        = jcp.dst_tag == format_tag::ndhwc;
                const int oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb_l2 * jcp.oc_block
                        : g_ocb + ocb_l2;
                auto diff_dst_w = diff_dst + diff_dst_d.blk_off(n, oc_off_idx)
                        + d_oj * diff_dst_d_stride;
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb)
                        + d_lo * wht_d_stride;

                int ocb_step = is_ddst_layout_nxc ? jcp.nb_oc_L2 : 1;
                int ocb_end = min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2);
                const int load_work = utils::this_block_size(icb * jcp.ic_block,
                        jcp.ic, jcp.nb_ic_blocking * jcp.ic_block);
                int reduce_work = ocb_step * jcp.oc_block;
                for (int ocb = ocb_l2; ocb < ocb_end; ocb += ocb_step) {
                    int curr_nb_oc = nstl::min(ocb_step, ocb_end - ocb);
                    if (ocb + curr_nb_oc >= jcp.nb_oc) {
                        reduce_work = utils::this_block_size(ocb * jcp.oc_block,
                                jcp.oc, ocb_step * jcp.oc_block);
                    }
                    for (int ij = ih_s; ij < ih_e; ++ij) {
                        int oj, k_len, k_lo;
                        if (is_fast_path_h) { // dilate == 0 && stride == 1
                            int i_t_overflow
                                    = max(0, jcp.kh - 1 - ij - jcp.t_pad);
                            int i_b_overflow
                                    = max(0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else if (jcp.dilate_h != 0) { // stride == 1
                            int dilate_h = jcp.dilate_h + 1;
                            // Note: use div_up to account for "holes" in filter
                            int i_t_overflow
                                    = div_up(max(0,
                                                     (jcp.kh - 1) * dilate_h
                                                             - ij - jcp.t_pad),
                                            dilate_h);
                            int i_b_overflow = div_up(
                                    max(0,
                                            (jcp.kh - 1) * dilate_h + 1 - jcp.ih
                                                    + ij - jcp.b_pad),
                                    dilate_h);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                        } else { // dilate == 0
                            int i_t_overflow = max(0,
                                    (jcp.kh - 1 - ij - jcp.t_pad)
                                            / jcp.stride_h);
                            int i_b_overflow = max(0,
                                    (jcp.kh - jcp.ih + ij - jcp.b_pad)
                                            / jcp.stride_h);
                            int overflow_kh_hi = jcp.kh - 1
                                    - modulo(jcp.ih - 1 + jcp.b_pad - ij,
                                            jcp.stride_h);
                            int overflow_kh_lo
                                    = (ij + jcp.t_pad) % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                            / jcp.stride_h
                                    + 1 - i_t_overflow - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }
                        assert(k_len >= 0);

                        jit_conv_3d_ker_pipeline(jit_ker, par_conv,
                                diff_src_w + ij * diff_src_h_stride,
                                diff_dst_w + oj * diff_dst_h_stride,
                                wht_w + k_lo * wht_h_stride, nullptr, ocb,
                                k_len, d_len, reduce_work, load_work);
                    }
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == loop_cwgn)
                    nd_iterator_jump(start, end, icc, ic_chunks, gg, nb_groups,
                            n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih);
                else if (jcp.loop_order == loop_gncw)
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                            ic_chunks, id_s, jcp.id, ih_s, jcp.ih);
                else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih, icc,
                            ic_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }

        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_3d_ker_pipeline(jit_ker, par_conv, diff_src, diff_dst, weights,
                nullptr, 0, 1, 1, 0, 0);
    });
}

template struct jit_avx512_common_convolution_bwd_data_t<data_type::f32>;

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
status_t jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::init(engine_t *engine) {
    const auto &j = pd()->jcp_;

    nthr_ = j.nthr;
    nthr_mb_ = j.nthr_mb;
    nthr_g_ = j.nthr_g;
    nthr_oc_b_ = j.nthr_oc_b;
    nthr_ic_b_ = j.nthr_ic_b;

    CHECK(safe_ptr_assign(
            kernel_, new jit_avx512_common_conv_bwd_weights_kernel_f32(j)));
    CHECK(kernel_->create_kernel());

    if (j.ver == ver_4fma) {
        CHECK(safe_ptr_assign(trans_kernel_, create_trans_src(&j)));
        CHECK(trans_kernel_->create_kernel());
    }

    if (nthr_mb_ > 1) {
        CHECK(safe_ptr_assign(
                acc_ker_, new cpu_accumulator_1d_t<diff_weights_type>()));
        CHECK(acc_ker_->create_kernel());
    }

    CHECK(safe_ptr_assign(reducer_bias_,
            new cpu_reducer_t<diff_weights_type>(pd()->reducer_bia_conf_)));
    CHECK(reducer_bias_->create_kernel());
    return status::success;
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
struct jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::thread_info_t {
    const src_data_t *src;
    const diff_dst_data_t *diff_dst;
    const diff_weights_data_t *diff_weights;
    diff_weights_data_t *diff_bias;

    const memory_tracking::grantor_t scratchpad;

    src_data_t *tr_src;
    simple_barrier::ctx_t *tr_src_bctx;

    diff_dst_data_t *tr_diff_dst;
    simple_barrier::ctx_t *tr_diff_dst_bctx;

    diff_weights_data_t *wei_bia_reduction;
    simple_barrier::ctx_t *wei_bia_reduction_bctx;

    int ithr;
    int ithr_ic_b, ithr_oc_b, ithr_g, ithr_mb;
    int ithr_but_oc;
    int ithr_but_ic;

    int img_start = 0, img_end = 0, img_work;
    int g_start = 0, g_end = 0, g_work;
    int oc_b_start = 0, oc_b_end = 0, oc_b_work;
    int ic_b_start = 0, ic_b_end = 0, ic_b_work;

    thread_info_t(const jit_avx512_common_convolution_bwd_weights_t *self,
            const exec_ctx_t &ctx, int ithr)
        : scratchpad(ctx.get_scratchpad_grantor()), ithr(ithr) {
        diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
        src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
        diff_weights
                = CTX_OUT_MEM(diff_weights_data_t *, DNNL_ARG_DIFF_WEIGHTS);
        const auto &jcp = self->kernel_->jcp;
        const bool is_bias_padded = self->pd()->with_bias()
                && jcp.oc_without_padding % jcp.oc_block != 0;
        diff_bias = is_bias_padded
                ? scratchpad.template get<diff_weights_data_t>(
                        key_conv_padded_bias)
                : CTX_OUT_MEM(diff_weights_data_t *, DNNL_ARG_DIFF_BIAS);

        tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
        tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_tr_src_bctx);

        tr_diff_dst = scratchpad.template get<diff_dst_data_t>(
                key_conv_tr_diff_dst);
        tr_diff_dst_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_tr_diff_dst_bctx);

        wei_bia_reduction = scratchpad.template get<diff_weights_data_t>(
                key_conv_wei_bia_reduction);
        wei_bia_reduction_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx);

        ithr_ic_b = ithr % self->nthr_ic_b_;
        ithr_oc_b = ithr / self->nthr_ic_b_ % self->nthr_oc_b_;
        ithr_g = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ % self->nthr_g_;
        ithr_mb = ithr / self->nthr_ic_b_ / self->nthr_oc_b_ / self->nthr_g_;

        ithr_but_oc = (ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_ic_b_
                + ithr_ic_b;

        ithr_but_ic = (ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_oc_b_
                + ithr_oc_b;

        /* reduction dimension */
        int oh_reduce = jcp.harness == harness_2d_reduction ? jcp.oh : 1;
        balance211(jcp.mb * jcp.od * oh_reduce, self->nthr_mb_, ithr_mb,
                img_start, img_end);
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

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_weights_nxc(const thread_info_t *ti)
        const {
    const auto &jcp = kernel_->jcp;

    const int wei_size
            = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw * jcp.kd;
    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_weights
            : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    auto diff_weights_offset
            = [&](int g, int i_kd, int i_kh, int i_kw, int i_ic, int i_oc) {
                  const int oc_block_size = 1;
                  const int ic_block_size = jcp.oc_block * oc_block_size;
                  const int kw_block_size = jcp.ic_block * ic_block_size;
                  const int kh_block_size = jcp.kw * kw_block_size;
                  const int kd_block_size = jcp.kh * kh_block_size;
                  const int icb_block_size = jcp.kd * kd_block_size;
                  const int ocb_block_size = jcp.nb_ic * icb_block_size;
                  const int g_block_size = jcp.nb_oc * ocb_block_size;

                  int icb = i_ic / jcp.ic_block;
                  int ocb = i_oc / jcp.oc_block;
                  i_ic = i_ic % jcp.ic_block;
                  i_oc = i_oc % jcp.oc_block;

                  return g * g_block_size + ocb * ocb_block_size
                          + icb * icb_block_size + i_kd * kd_block_size
                          + i_kh * kh_block_size + i_kw * kw_block_size
                          + i_ic * ic_block_size + i_oc * oc_block_size;
              };
    auto src_offset
            = [&](int g, int i_mb, int i_id, int i_ih, int i_ic, int i_iw) {
                  const int ic_block_size = 1;
                  const int g_block_size = jcp.ic * ic_block_size;
                  const int iw_block_size = jcp.ngroups * g_block_size;
                  const int ih_block_size = jcp.iw * iw_block_size;
                  const int id_block_size = jcp.ih * ih_block_size;
                  const int mb_block_size = jcp.id * id_block_size;

                  return g * g_block_size + i_mb * mb_block_size
                          + i_id * id_block_size + i_ih * ih_block_size
                          + i_iw * iw_block_size + i_ic * ic_block_size;
              };
    auto diff_dst_offset
            = [&](int g, int i_mb, int i_od, int i_oh, int i_ow, int i_oc) {
                  const int oc_block_size = 1;
                  const int g_block_size = jcp.oc * oc_block_size;
                  const int ow_block_size = jcp.ngroups * g_block_size;
                  const int oh_block_size = jcp.ow * ow_block_size;
                  const int od_block_size = jcp.oh * oh_block_size;
                  const int mb_block_size = jcp.od * od_block_size;

                  return g * g_block_size + i_mb * mb_block_size
                          + i_od * od_block_size + i_oh * oh_block_size
                          + i_ow * ow_block_size + i_oc * oc_block_size;
              };
    auto zero_diff_weights = [&]() {
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < wei_size; i++)
            diff_wei[i] = 0;
    };

    int kd_step = jcp.dilate_d + 1;
    int kh_step = jcp.dilate_h + 1;
    int stride_d = jcp.stride_d;
    int stride_h = jcp.stride_h;
    int f_pad = jcp.f_pad;
    int t_pad = jcp.t_pad;

    dim_t work_amount = jcp.mb * jcp.od * jcp.oh * jcp.nb_ow;
    dim_t i_work {0}, i_work_end {0};
    balance211(work_amount, jcp.nthr_mb, ti->ithr_mb, i_work, i_work_end);

    int i_mb {0}, i_od {0}, i_oh {0}, i_owb {0};
    nd_iterator_init(
            i_work, i_mb, jcp.mb, i_od, jcp.od, i_oh, jcp.oh, i_owb, jcp.nb_ow);

    zero_diff_weights();
    while (i_work < i_work_end) {
        int kd_start = nstl::max(
                0, div_up(jcp.f_pad - jcp.stride_d * i_od, kd_step));
        int kd_end = nstl::min(
                jcp.kd - 1, (jcp.id - 1 + f_pad - stride_d * i_od) / kd_step);
        int i_id_base = stride_d * i_od - f_pad;
        int kh_start = nstl::max(
                0, div_up(jcp.t_pad - jcp.stride_h * i_oh, +kh_step));
        int kh_end = nstl::min(
                jcp.kh - 1, (jcp.ih - 1 + t_pad - stride_h * i_oh) / kh_step);
        int i_ih_base = jcp.stride_h * i_oh + -jcp.t_pad;
        int i_ow_base = i_owb * jcp.ow_block;
        int i_ow_end = nstl::min(jcp.ow, i_ow_base + jcp.ow_block);

        // The kernel is small so these loops produce measurable overhead. Since
        // these are simple loops, the compiler will likely make the loops just
        // as well as we can with the jitted assembly, so there is not
        // necessarily a reason to move these loops into assembly. Avoid placing
        // computationally heavy operations within the loops.
        for_(int i_ow = i_ow_base; i_ow < i_ow_end; i_ow += jcp.ur_ow)
        for_(int i_oc = 0; i_oc < jcp.oc; i_oc += jcp.oc_block)
        for_(int g = 0; g < jcp.ngroups; g++)
        for_(int i_kd = kd_start; i_kd <= kd_end; i_kd++)
        for (int i_kh = kh_start; i_kh <= kh_end; i_kh++) {
            // Some Optimization Observations: It may be
            // worthwhile to move the kd and kh loops below the
            // icb loop in the kernel to further amortize the
            // ddst register loads. Alternatively, these
            // dimensions are independent on the weights kernel,
            // so can be used as a threading dimension that does
            // not require reduction.

            // The compiler seems to do a good job at optimizing these
            // computations. The offset functions likely need to be located
            // so that they will be inlined.
            int i_iw = i_ow * jcp.stride_w - jcp.l_pad;
            int i_id = i_id_base + i_kd * kd_step;
            int i_ih = i_ih_base + i_kh * kh_step;
            int ddst_offset = diff_dst_offset(g, i_mb, i_od, i_oh, i_ow, i_oc);
            int s_off_base = src_offset(g, i_mb, i_id, i_ih, 0, i_iw);
            int dwei_off_base = diff_weights_offset(g, i_kd, i_kh, 0, 0, i_oc);
            // ensure all parameters are 64bit, to comply with windows kernel
            // param access where the params from 5th are passed using stack.
            (*kernel_)(&diff_wei[dwei_off_base], &ti->src[s_off_base],
                    &ti->diff_dst[ddst_offset], (dim_t)i_iw, (dim_t)i_ow);
        }
        nd_iterator_step(
                i_mb, jcp.mb, i_od, jcp.od, i_oh, jcp.oh, i_owb, jcp.nb_ow);
        i_work++;
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_weights(const thread_info_t *ti)
        const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    const int padded_oc = rnd_up(jcp.oc, jcp.oc_block);
    const int wei_size = jcp.ngroups * padded_oc * rnd_up(jcp.ic, jcp.ic_block)
            * jcp.kh * jcp.kw * jcp.kd;

    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_weights
            : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;
    diff_weights_data_t *diff_bia = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_bias
            : ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size
                    + (ti->ithr_mb - 1) * jcp.ngroups * padded_oc;

    const bool is_src_layout_nxc = utils::one_of(
            jcp.src_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    // TODO: use memory descriptor with the same fmt as src (or use a macro :))
    auto tr_src_off = [&](int ithr_mb, int ic, int ij) {
        assert(!is_src_layout_nxc);
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_chn_size = tr_row_size * jcp.ih;
        const size_t tr_img_size = tr_chn_size * jcp.nb_ic * jcp.ngroups;

        return ti->ithr_mb * tr_img_size + ic * tr_chn_size + ij * tr_row_size;
    };

    auto uker_trans = [&](int img) {
        assert(!is_src_layout_nxc);
        const int work_amount = ti->g_work * ti->ic_b_work * jcp.ih;

        int start {0}, end {0};
        balance211(work_amount, nthr_oc_b_, ti->ithr_oc_b, start, end);
        const int my_work = end - start;

        int g {0}, ic_b {0}, j {0};
        nd_iterator_init(start, g, ti->g_work, ic_b, ti->ic_b_work, j, jcp.ih);
        g += ti->g_start;
        ic_b += ti->ic_b_start;

        const int _ic = g * jcp.nb_ic + ic_b;
        src_data_t *src1 = (src_data_t *)&ti->src[src_d.blk_off(img, _ic, j)];
        src_data_t *tr_src1 = &ti->tr_src[tr_src_off(ti->ithr_mb, _ic, j)];

        assert(jcp.ic_block == 16);
        const int src_stride = jcp.iw * jcp.ic_block;
        const int tr_src_stride = jcp.tr_iw * jcp.ic_block;

        const int pf_depth = 2;
        struct {
            src_data_t *src, *tr_src;
        } pf_circ_buf[pf_depth];

        for (int iwork = 0; iwork < my_work + pf_depth - 1; iwork++) {
            pf_circ_buf[iwork % pf_depth] = {src1, tr_src1};

            if (iwork >= pf_depth - 1) {
                int old_idx = (iwork - pf_depth + 1) % pf_depth;
                auto ctx = jit_trans_src_t::ctx_t();
                ctx.src = pf_circ_buf[old_idx].src;
                ctx.tr_src = pf_circ_buf[old_idx].tr_src;
                ctx.src_prf = src1;
                ctx.tr_src_prf = tr_src1;
                (*trans_kernel_)(&ctx);
            }
            src1 += src_stride;
            tr_src1 += tr_src_stride;
        }
#if 0
        // reference transposition
        const int l_pad = jcp.l_pad;
        const int iwlp = l_pad + jcp.iw;
        const int tr_iw = jcp.tr_iw;

        for (size_t iwork = start; iwork < end; iwork++) {
            PRAGMA_OMP_SIMD()
#pragma unroll
            for (int i = 0; i < l_pad; i++)
                for (int j = 0; j < jcp.ic_block; j++)
                    tr_src1[j * jcp.tr_iw + i] = (src_data_t)0.0;

            PRAGMA_OMP_SIMD()
#pragma unroll
            for (int i = l_pad; i < iwlp; i++)
                for (int j = 0; j < jcp.ic_block; j++)
                    tr_src1[j * jcp.tr_iw + i]
                        = (src_data_t)src1[(i - l_pad) * 16 + j];

            PRAGMA_OMP_SIMD()
#pragma unroll
            for (int i = iwlp; i < tr_iw; i++)
                for (int j = 0; j < jcp.ic_block; j++)
                    tr_src1[j * jcp.tr_iw + i] = (src_data_t)0.0;

             src1 += src_stride;
             tr_src1 += tr_src_stride;
         }
#endif
    };

    if (jcp.is_1stconv && jcp.ver == ver_4fma) {
        assert(!is_src_layout_nxc);
        /* prepare contexts */
        auto tr_ctx = jit_trans_src_t::ctx_t();
        tr_ctx.tr_src = ti->tr_src
                + ti->ithr_but_oc * jcp.ih * jcp.stride_w * jcp.tr_ld;

        assert(IMPLICATION(!dnnl_thr_syncable(), nthr_oc_b_ == 1));
        tr_ctx.nthr_oc_b = nthr_oc_b_;
        int ih_start {0}, ih_end {0};
        balance211(jcp.ih, nthr_oc_b_, ti->ithr_oc_b, ih_start, ih_end);
        tr_ctx.tr_src_ih_start = ih_start;
        tr_ctx.tr_src_ih_end = ih_end;
        tr_ctx.tr_src_bctx = ti->tr_src_bctx + ti->ithr_but_oc;

        auto p = jit_conv_call_s();
        p.src = tr_ctx.tr_src;

        /* zero diff_bias if applicable */
        if (jcp.with_bias && ti->ithr_ic_b == 0) {
            assert(jcp.oc_block == 16);
            for (int oc_b = ti->ic_b_start; oc_b < ti->oc_b_end; ++oc_b) {
                diff_weights_data_t *db = &diff_bia[oc_b * 16];
                for (int o = 0; o < 16; ++o)
                    db[o] = 0;
            }
        }

        for (int img = ti->img_start; img < ti->img_end; ++img) {
            p.flags = (img == ti->img_start) * FLAG_MB_FIRST;

            for_(int g = ti->g_start; g < ti->g_end; ++g)
            for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end; ++ic_b) {
                const int _ic = g * jcp.nb_ic + ic_b;
                tr_ctx.src = &ti->src[src_d.blk_off(img, _ic)];

                (*trans_kernel_)(&tr_ctx);

                if (ic_b == 0)
                    p.flags |= FLAG_IC_FIRST;
                else
                    p.flags &= ~FLAG_IC_FIRST;

                for (int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b) {
                    const int _oc = g * jcp.nb_oc + oc_b;
                    p.dst = &ti->diff_dst[diff_dst_d.blk_off(img, _oc)];

                    const size_t off
                            = wht_blk_off(diff_weights_d, g, oc_b, ic_b);
                    p.filt = diff_wei + off;
                    p.bias = diff_bia + _oc * jcp.oc_block;

                    (*kernel_)(&p);
                }
            }
        }
    } else {
        int ic_b_step = jcp.nb_ic_blocking_max;
        int icb_work = ti->ic_b_end - ti->ic_b_start;
        if (ic_b_step > 1 && icb_work > ic_b_step && icb_work < 2 * ic_b_step)
            ic_b_step = utils::div_up(icb_work, 2);

        for (int img = ti->img_start; img < ti->img_end; ++img) {
            auto p = jit_conv_call_s();

            if (jcp.ver == ver_4fma) {
                /* tr_src[nb_ic][ih][16][~iw~] <- src[nb_ic][ih][iw][16] */
                using simple_barrier::barrier;
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
                uker_trans(img);
                if (nthr_oc_b_ > 1)
                    barrier(&ti->tr_src_bctx[ti->ithr_but_oc], nthr_oc_b_);
            }

            const int max_oc = nstl::min(ti->oc_b_end * jcp.oc_block, jcp.oc);
            const int max_ic = nstl::min(ti->ic_b_end * jcp.ic_block, jcp.ic);
            const bool is_ddst_layout_nxc = utils::one_of(jcp.dst_tag,
                    format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
            for_(int g = ti->g_start; g < ti->g_end; ++g)
            for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
            for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                    ic_b += ic_b_step) {
                const int _oc = g * jcp.nb_oc + oc_b;
                const int _ic = g * jcp.nb_ic + ic_b;
                const int ic_to_compute = this_block_size(
                        ic_b * jcp.ic_block, max_ic, ic_b_step * jcp.ic_block);
                const int oc_to_compute = this_block_size(
                        oc_b * jcp.oc_block, max_oc, jcp.oc_block);

                const int ic_off_idx = is_src_layout_nxc
                        ? g * jcp.ic + ic_b * jcp.ic_block
                        : _ic;
                const int oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + oc_b * jcp.oc_block
                        : _oc;

                jit_conv_ker_pipeline_bwd_w(jit_ker, p,
                        jcp.ver == ver_4fma
                                ? &ti->tr_src[tr_src_off(ti->ithr_mb, _ic, 0)]
                                : &ti->src[src_d.blk_off(img, ic_off_idx)],
                        &ti->diff_dst[diff_dst_d.blk_off(img, oc_off_idx)],
                        diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                        nullptr, (img == ti->img_start), 0, ic_to_compute,
                        oc_to_compute);
            }

            const int _oc = ti->g_start * jcp.nb_oc + ti->oc_b_start;
            const int _ic = ti->g_start * jcp.nb_ic + ti->ic_b_start;
            const int ic_off_idx = is_src_layout_nxc
                    ? ti->g_start * jcp.ic + ti->ic_b_start * jcp.ic_block
                    : _ic;
            const int oc_off_idx = is_ddst_layout_nxc
                    ? ti->g_start * jcp.oc + ti->oc_b_start * jcp.oc_block
                    : _oc;
            // This call is required only to finalize pipeline with paramaters
            // set on the last iteration of loop above. Only valid pointers make
            // sense here as call parameters to avoid execution of prefetch
            // instructions with nullptr, other parameters are not used in real
            // jit call here
            jit_conv_ker_pipeline_bwd_w(jit_ker, p,
                    jcp.ver == ver_4fma
                            ? &ti->tr_src[tr_src_off(ti->ithr_mb, _ic, 0)]
                            : &ti->src[src_d.blk_off(img + 1, ic_off_idx)],
                    &ti->diff_dst[diff_dst_d.blk_off(img + 1, oc_off_idx)],
                    diff_wei
                            + wht_blk_off(diff_weights_d, ti->g_start,
                                    ti->oc_b_start, ti->ic_b_start),
                    nullptr, 0, 0, 0, 0);
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_weights_2d(const thread_info_t *ti)
        const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    const int padded_oc = rnd_up(jcp.oc, jcp.oc_block);
    const int wei_size = jcp.ngroups * padded_oc * rnd_up(jcp.ic, jcp.ic_block)
            * jcp.kh * jcp.kw;

    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_weights
            : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;
    diff_weights_data_t *diff_bia = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_bias
            : ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size
                    + (ti->ithr_mb - 1) * jcp.ngroups * padded_oc;

    int img {0}, oh_s {0};
    int img_start = ti->img_start, img_end = ti->img_end;
    nd_iterator_init(img_start, img, jcp.mb, oh_s, jcp.oh);
    const int img_first = img;

    int ic_b_step = jcp.nb_ic_blocking_max;
    int icb_work = ti->ic_b_end - ti->ic_b_start;
    if (ic_b_step > 1 && icb_work > ic_b_step && icb_work < 2 * ic_b_step)
        ic_b_step = utils::div_up(icb_work, 2);
    while (img_start < img_end) {
        auto p = jit_conv_call_s();

        int work_rem = img_end - img_start;
        const int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
        const int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
        const int kh_top_overflow = nstl::max(0, -ih_s);
        const int kh_bottom_overflow = nstl::max(0, ih_s - jcp.ih + jcp.kh);
        int kh_padding = jcp.kh - kh_top_overflow - kh_bottom_overflow;
        int kh_padding_offset = nstl::min(jcp.kh - 1, kh_top_overflow) * jcp.kw
                * jcp.ic_block * jcp.oc_block * jcp.typesize_out;
        auto src_h = ti->src + src_d.blk_off(img, 0, ih_s + kh_top_overflow);
        auto diff_dst_h = ti->diff_dst + diff_dst_d.blk_off(img, 0, oh_s);

        const bool is_src_layout_nxc = jcp.src_tag == format_tag::nhwc;
        const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::nhwc;
        const int max_oc = nstl::min(ti->oc_b_end * jcp.oc_block, jcp.oc);
        const int max_ic = nstl::min(ti->ic_b_end * jcp.ic_block, jcp.ic);
        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += ic_b_step) {
            const int _oc = g * jcp.nb_oc + oc_b;
            const int _ic = g * jcp.nb_ic + ic_b;
            const int ic_to_compute = this_block_size(
                    ic_b * jcp.ic_block, max_ic, ic_b_step * jcp.ic_block);
            const int oc_to_compute = this_block_size(
                    oc_b * jcp.oc_block, max_oc, jcp.oc_block);
            const int ic_off_idx = is_src_layout_nxc
                    ? g * jcp.ic + ic_b * jcp.ic_block
                    : _ic;
            const int oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + oc_b * jcp.oc_block
                    : _oc;
            auto src = src_h + src_d.blk_off(0, ic_off_idx);
            auto diff_dst = diff_dst_h + diff_dst_d.blk_off(0, oc_off_idx);

            jit_conv_2d_ker_bwd_w_pipeline(jit_ker, p, src, diff_dst,
                    diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                    diff_bia + _oc * jcp.oc_block, (img == img_first), oh_s,
                    oh_e, kh_padding, kh_padding_offset, ic_to_compute,
                    oc_to_compute);

            p.flags = ic_b == 0 ? 0 : 1;
        }

        const int _oc = ti->g_start * jcp.nb_oc + ti->oc_b_start;
        const int _ic = ti->g_start * jcp.nb_ic + ti->ic_b_start;
        const int ic_off_idx = is_src_layout_nxc
                ? ti->g_start * jcp.ic + ti->ic_b_start * jcp.ic_block
                : _ic;
        const int oc_off_idx = is_ddst_layout_nxc
                ? ti->g_start * jcp.oc + ti->oc_b_start * jcp.oc_block
                : _oc;
        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_2d_ker_bwd_w_pipeline(jit_ker, p,
                ti->src + src_d.blk_off(img + 1, ic_off_idx),
                ti->diff_dst + diff_dst_d.blk_off(img + 1, oc_off_idx),
                diff_wei
                        + wht_blk_off(diff_weights_d, ti->g_start,
                                ti->oc_b_start, ti->ic_b_start),
                diff_bia + _oc * jcp.oc_block, 0, 0, 0, 0, 0, 0, 0);
        nd_iterator_jump(img_start, img_end, img, jcp.mb, oh_s, jcp.oh);
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_weights_3d(const thread_info_t *ti)
        const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    const int padded_oc = rnd_up(jcp.oc, jcp.oc_block);
    const int wei_size = jcp.ngroups * padded_oc * rnd_up(jcp.ic, jcp.ic_block)
            * jcp.kh * jcp.kw * jcp.kd;

    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_weights
            : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;
    diff_weights_data_t *diff_bia = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_bias
            : ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size
                    + (ti->ithr_mb - 1) * jcp.ngroups * padded_oc;

    const bool is_src_layout_nxc = jcp.src_tag == format_tag::ndhwc;
    const int inp_mult = is_src_layout_nxc
            ? jcp.ngroups * jcp.ic
            : (jcp.is_1stconv ? 1 : jcp.ic_block);
    const int input_step = jcp.ih * jcp.iw * inp_mult;
    const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
    const int output_step = jcp.ow * jcp.oh
            * (is_ddst_layout_nxc ? jcp.ngroups * jcp.oc : jcp.oc_block);
    int img {0}, od_s {0};
    int img_start = ti->img_start, img_end = ti->img_end;
    nd_iterator_init(img_start, img, jcp.mb, od_s, jcp.od);
    const int img_first = img;

    int ic_b_step = jcp.nb_ic_blocking_max;
    int icb_work = ti->ic_b_end - ti->ic_b_start;
    if (ic_b_step > 1 && icb_work > ic_b_step && icb_work < 2 * ic_b_step)
        ic_b_step = utils::div_up(icb_work, 2);

    while (img_start < img_end) {
        auto p = jit_conv_call_s();

        int work_rem = img_end - img_start;
        const int od_e = od_s + work_rem > jcp.od ? jcp.od : od_s + work_rem;
        const int id_s = od_s * jcp.stride_d;
        const int ik_overlap = nstl::max(0, id_s - jcp.f_pad);
        const int kd_front_pad = nstl::max(0, jcp.f_pad - id_s);
        const int kd_back_pad
                = nstl::max(0, id_s - jcp.f_pad - jcp.id + jcp.kd);
        int kd_pad_off = nstl::min(jcp.kd - 1, kd_front_pad) * jcp.kh * jcp.kw
                * jcp.ic_block * jcp.oc_block * jcp.typesize_out;

        const int max_oc = nstl::min(ti->oc_b_end * jcp.oc_block, jcp.oc);
        const int max_ic = nstl::min(ti->ic_b_end * jcp.ic_block, jcp.ic);

        for_(int g = ti->g_start; g < ti->g_end; ++g)
        for_(int oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (int ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += ic_b_step) {
            const int _oc = g * jcp.nb_oc + oc_b;
            const int _ic = g * jcp.nb_ic + ic_b;

            const int ic_to_compute = this_block_size(
                    ic_b * jcp.ic_block, max_ic, ic_b_step * jcp.ic_block);
            const int oc_to_compute = this_block_size(
                    oc_b * jcp.oc_block, max_oc, jcp.oc_block);

            const int ic_off_idx = is_src_layout_nxc
                    ? g * jcp.ic + ic_b * jcp.ic_block
                    : _ic;
            const int oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + oc_b * jcp.oc_block
                    : _oc;
            auto src = &ti->src[src_d.blk_off(img, ic_off_idx)
                    + ik_overlap * input_step];
            auto dst = &ti->diff_dst[diff_dst_d.blk_off(img, oc_off_idx)
                    + od_s * output_step];

            jit_conv_3d_ker_bwd_w_pipeline(jit_ker, p, src, dst,
                    diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                    diff_bia + _oc * 16, (img == img_first), od_s, od_e,
                    jcp.kd - kd_front_pad - kd_back_pad, kd_pad_off,
                    ic_to_compute, oc_to_compute);

            p.flags = ic_b == 0 ? 0 : 1;
        }

        const int _oc = ti->g_start * jcp.nb_oc + ti->oc_b_start;
        const int _ic = ti->g_start * jcp.nb_ic + ti->ic_b_start;
        const int ic_off_idx = is_src_layout_nxc
                ? ti->g_start * jcp.ic + ti->ic_b_start * jcp.ic_block
                : _ic;
        const int oc_off_idx = is_ddst_layout_nxc
                ? ti->g_start * jcp.oc + ti->oc_b_start * jcp.oc_block
                : _oc;
        // This call is required only to finalize pipeline with paramaters set
        // on the last iteration of loop above. Only valid pointers make sense
        // here as call parameters to avoid execution of prefetch instructions
        // with nullptr, other parameters are not used in real jit call here
        jit_conv_3d_ker_bwd_w_pipeline(jit_ker, p,
                &ti->src[src_d.blk_off(img + 1, ic_off_idx)],
                &ti->diff_dst[diff_dst_d.blk_off(img + 1, oc_off_idx)],
                diff_wei
                        + wht_blk_off(diff_weights_d, ti->g_start,
                                ti->oc_b_start, ti->ic_b_start),
                diff_bia, 0, 0, 0, 0, 0, 0, 0);
        nd_iterator_jump(img_start, img_end, img, jcp.mb, od_s, jcp.od);
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::reduce_diff_weights(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const int padded_oc = rnd_up(jcp.oc, jcp.oc_block);
    const int wei_size = jcp.ngroups * padded_oc * rnd_up(jcp.ic, jcp.ic_block)
            * jcp.kh * jcp.kw;
    const int bia_size = jcp.ngroups * padded_oc;
    const diff_weights_data_t *diff_bias_ws
            = ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size;

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    if (dnnl_thr_syncable())
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const int ic_b_kh_work = ti->ic_b_work * jcp.kh;
    const int work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int start {0}, end {0};
    balance211(work, nthr_mb_, ti->ithr_mb, start, end);
    if (start == end) return;

    for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
        int w = start;
        int sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_kh_start {0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const int g = ti->g_start + sub_g_start;
            const int oc_b = ti->oc_b_start + sub_oc_b_start;
            const int ic_b = ti->ic_b_start + sub_ic_b_kh_start / jcp.kh;
            const int kh = sub_ic_b_kh_start % jcp.kh;

            const int acc_size
                    = nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start)
                    * jcp.kw * jcp.ic_block * jcp.oc_block;

            const size_t off = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kh);

            diff_weights_data_t *d
                    = (diff_weights_data_t *)ti->diff_weights + off;
            diff_weights_data_t *s
                    = ti->wei_bia_reduction + (thr_mb - 1) * wei_size + off;

            acc_ker_->accumulate(d, s, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }

        if (jcp.with_bias && jcp.is_1stconv && jcp.ver == ver_4fma) {
            if (ti->ithr == 0)
                acc_ker_->accumulate((diff_weights_data_t *)ti->diff_bias,
                        diff_bias_ws, bia_size);
            diff_bias_ws += bia_size;
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::reduce_diff_weights_3d(const thread_info_t *ti)
        const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const int wei_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block)
            * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw * jcp.kd;

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    if (dnnl_thr_syncable())
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const int ic_b_kh_work = ti->ic_b_work * jcp.kd;
    const int work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int start {0}, end {0};
    balance211(work, nthr_mb_, ti->ithr_mb, start, end);
    if (start == end) return;

    for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
        int w = start;
        int sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_kh_start {0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const int g = ti->g_start + sub_g_start;
            const int oc_b = ti->oc_b_start + sub_oc_b_start;
            const int ic_b = ti->ic_b_start + sub_ic_b_kh_start / jcp.kd;
            const int kd = sub_ic_b_kh_start % jcp.kd;

            const int acc_size
                    = nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start)
                    * jcp.kw * jcp.ic_block * jcp.oc_block * jcp.kh;

            const size_t off = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kd);
            diff_weights_data_t *d
                    = (diff_weights_data_t *)ti->diff_weights + off;
            diff_weights_data_t *s
                    = ti->wei_bia_reduction + (thr_mb - 1) * wei_size + off;
            acc_ker_->accumulate(d, s, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_bias(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    auto rb = this->reducer_bias_.get();
    assert(nthr_ == rb->balancer().nthr_);

    const auto reducer_bia_scratchpad
            = memory_tracking::grantor_t(ti->scratchpad, prefix_reducer_bia);

    const auto &jcp = kernel_->jcp;

    if (jcp.with_bias && jcp.is_1stconv && jcp.ver == ver_4fma) return;

    const int b_job_start = rb->balancer().ithr_job_off(ti->ithr);
    const int b_njobs = rb->balancer().ithr_njobs(ti->ithr);

    if (b_njobs == 0) return;

    /* reduction dimension */
    int img_start {0}, img_end {0};
    balance211(jcp.mb, rb->balancer().nthr_per_group_,
            rb->balancer().id_in_group(ti->ithr), img_start, img_end);

    /* jobs */
    int g_start {0}, ocb_start {0};
    nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_oc);
    for (int img = img_start; img < img_end; ++img) {
        int g = g_start, ocb = ocb_start;
        for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
            const size_t _oc = g * jcp.nb_oc + ocb;
            const int max_oc
                    = this_block_size(ocb * jcp.oc_block, jcp.oc, jcp.oc_block);

            const bool is_ddst_layout_nxc = utils::one_of(jcp.dst_tag,
                    format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
            const int oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + ocb * jcp.oc_block
                    : _oc;
            const diff_dst_data_t *d_dst
                    = &ti->diff_dst[diff_dst_d.blk_off(img, oc_off_idx)];
            diff_weights_data_t *d_bias
                    = rb->get_local_ptr(
                              ti->ithr, ti->diff_bias, reducer_bia_scratchpad)
                    + b_job_loc * rb->balancer().job_size_;

            if (img == img_start)
                for (int o = 0; o < jcp.oc_block; ++o)
                    d_bias[o] = 0;
            for (int hw = 0; hw < jcp.oh * jcp.ow * jcp.od; ++hw) {
                PRAGMA_OMP_SIMD()
                for (int o = 0; o < max_oc; ++o)
                    d_bias[o] += d_dst[o];
                d_dst += is_ddst_layout_nxc ? jcp.ngroups * jcp.oc
                                            : jcp.oc_block;
            }

            nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc);
        }
    }

    if (dnnl_thr_syncable())
        rb->reduce(ti->ithr, ti->diff_bias, reducer_bia_scratchpad);
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::reduce_diff_bias(const thread_info_t *ti) const {
    const auto &jcp = kernel_->jcp;

    const size_t wei_size = (size_t)jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block)
            * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw * jcp.kd;
    const int bia_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block);
    const diff_weights_data_t *diff_bias_ws
            = ti->wei_bia_reduction + (size_t)(nthr_mb_ - 1) * wei_size;

    if (dnnl_thr_syncable() && nthr_mb_ > 1) dnnl_thr_barrier();

    if (ti->ithr == 0) {
        for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
            acc_ker_->accumulate(ti->diff_bias, diff_bias_ws, bia_size);
            diff_bias_ws += bia_size;
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::prepare_scratchpad_data(const exec_ctx_t &ctx)
        const {
    const auto &j = pd()->jcp_;
    auto scratchpad = ctx.get_scratchpad_grantor();

    if (j.ver == ver_4fma) {
        if (!j.is_1stconv) {
            // XXX: See the comment about tr_iw and guarding elements in
            // jit_avx512_common_conv_bwd_weights_kernel_f32::init_conf()
            const int max_nthr = j.nthr_mb * j.ngroups * j.nb_ic;
            const int min_tr_src_size_per_thr = j.ih * j.ic_block * j.tr_iw;

            auto tr_src = scratchpad.template get<src_data_t>(key_conv_tr_src);
            /* to avoid NaNs in computations we zero tail num_guard_elems for
             * each possible thread group */

            for (int ithr = 1; ithr <= max_nthr; ++ithr) {
                src_data_t *ts = &tr_src[ithr * min_tr_src_size_per_thr];
                for (int i = 0; i < j.tr_src_num_guard_elems; ++i)
                    ts[i] = 0;
            }
        }

        if (j.nthr_oc_b > 1) {
            const int tr_src_bctx_size = j.nthr / j.nthr_oc_b;
            auto tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_src_bctx);
            for (int i = 0; i < tr_src_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_src_bctx[i]);
        }
    }

    if (dnnl_thr_syncable() && nthr_mb_ > 1) {
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx));
    }

    const auto reducer_bia_scratchpad
            = memory_tracking::grantor_t(scratchpad, prefix_reducer_bia);
    auto rb = this->reducer_bias_.get();
    rb->init(reducer_bia_scratchpad);
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::execute_backward_weights(const exec_ctx_t &ctx)
        const {
    prepare_scratchpad_data(ctx);

#if DNNL_THR_SYNC == 1
    parallel(nthr_, [&](const int ithr, const int nthr) {
        assert(nthr_ == nthr);

        thread_info_t thread_info(this, ctx, ithr);

        switch (pd()->jcp_.harness) {
            case harness_2d_reduction:
                compute_diff_weights_2d(&thread_info);
                if (nthr_mb_ > 1) reduce_diff_weights(&thread_info);
                if (pd()->with_bias()) reduce_diff_bias(&thread_info);
                break;
            case harness_3d_reduction:
                compute_diff_weights_3d(&thread_info);
                if (nthr_mb_ > 1) reduce_diff_weights_3d(&thread_info);
                if (pd()->with_bias()) reduce_diff_bias(&thread_info);
                break;
            case harness_mb_reduction:
                compute_diff_weights(&thread_info);
                if (nthr_mb_ > 1) reduce_diff_weights(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info);
                break;
            case harness_nxc:
                compute_diff_weights_nxc(&thread_info);
                if (nthr_mb_ > 1) reduce_diff_weights_3d(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info);
                break;
            default: assert(!"Invalid harness type");
        }
    });
#else
    parallel(nthr_, [&](const int ithr, const int nthr) {
        thread_info_t thread_info(this, ctx, ithr);
        switch (pd()->jcp_.harness) {
            case harness_nxc:
                compute_diff_weights_nxc(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info);
                break;
            case harness_2d_reduction:
                compute_diff_weights_2d(&thread_info);
                break;
            case harness_3d_reduction:
                compute_diff_weights_3d(&thread_info);
                break;
            case harness_mb_reduction:
                compute_diff_weights(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info);
                break;
            default: assert(!"Invalid harness type");
        }
    });

    parallel(nthr_, [&](const int ithr, const int nthr) {
        thread_info_t thread_info(this, ctx, ithr);
        if (nthr_mb_ > 1) {
            switch (pd()->jcp_.harness) {
                case harness_mb_reduction:
                case harness_2d_reduction:
                    reduce_diff_weights(&thread_info);
                    break;
                case harness_nxc:
                case harness_3d_reduction:
                    reduce_diff_weights_3d(&thread_info);
                    break;
                default: assert(!"Invalid harness type");
            }
        }
        if (pd()->with_bias()) {
            switch (pd()->jcp_.harness) {
                case harness_2d_reduction:
                case harness_3d_reduction:
                    reduce_diff_bias(&thread_info);
                    break;
                case harness_nxc:
                case harness_mb_reduction: {
                    auto rb = this->reducer_bias_.get();
                    assert(nthr == rb->balancer().nthr_);
                    if (rb->balancer().ithr_njobs(ithr) == 0) return;
                    const auto reducer_bia_scratchpad
                            = memory_tracking::grantor_t(
                                    thread_info.scratchpad, prefix_reducer_bia);
                    rb->reduce_nolock(thread_info.ithr, thread_info.diff_bias,
                            reducer_bia_scratchpad);
                } break;
                default: assert(!"Invalid harness type");
            }
        }
    });
#endif

    /* TODO: put that into compute_diff_bias() */
    auto &jcp = pd()->jcp_;
    if (pd()->with_bias() && jcp.oc_without_padding % jcp.oc_block != 0) {
        auto diff_bias = ctx.get_scratchpad_grantor()
                                 .template get<const diff_weights_data_t>(
                                         key_conv_padded_bias);
        auto diff_bias_in
                = CTX_OUT_MEM(diff_weights_data_t *, DNNL_ARG_DIFF_BIAS);
        const int padded_stride = rnd_up(jcp.oc, jcp.oc_block);
        const int stride = jcp.oc_without_padding;
        for (int g = 0; g < jcp.ngroups; ++g) {
            utils::array_copy(diff_bias_in + g * stride,
                    diff_bias + g * padded_stride, stride);
        }
    }
}

template struct jit_avx512_common_convolution_bwd_weights_t<data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
