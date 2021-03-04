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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/ref_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

static inline dim_t get_offset(
        const memory_desc_wrapper &mdw, int n, int c, int d, int h, int w) {
    switch (mdw.ndims()) {
        case 3: return mdw.off(n, c, w);
        case 4: return mdw.off(n, c, h, w);
        case 5: return mdw.off(n, c, d, h, w);
        default: assert(!"Invalid tensor dimension in pooling");
    }
    return 0;
}

using namespace nstl;

template <data_type_t data_type, data_type_t acc_type>
void ref_pooling_fwd_t<data_type, acc_type>::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(unsigned char *, DNNL_ARG_WORKSPACE);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper ws_d(pd()->workspace_md());

    auto alg = pd()->desc()->alg_kind;
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    if (ws) assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);

    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();
    const int DD = pd()->DD();
    const int DH = pd()->DH();
    const int DW = pd()->DW();

    auto set_ws = [=](int mb, int oc, int od, int oh, int ow, int value) {
        if (ws) {
            const auto off = get_offset(ws_d, mb, oc, od, oh, ow);
            if (ws_dt == data_type::u8) {
                assert(0 <= value
                        && value <= numeric_limits<typename prec_traits<
                                        data_type::u8>::type>::max());
                ws[off] = value;
            } else
                reinterpret_cast<int *>(ws)[off] = value;
        }
    };

    auto ker_max = [=](float &d, int mb, int oc, int od, int oh, int ow) {
        for (int kd = 0; kd < KD; ++kd) {
            const int id = od * SD - padF + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (int kh = 0; kh < KH; ++kh) {
                const int ih = oh * SH - padT + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    const int iw = ow * SW - padL + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    const auto off = get_offset(src_d, mb, oc, id, ih, iw);
                    auto s = src[off];
                    if (s > d) {
                        d = s;
                        set_ws(mb, oc, od, oh, ow, (kd * KH + kh) * KW + kw);
                    }
                }
            }
        }
    };

    auto ker_avg = [=](float &d, int mb, int oc, int od, int oh, int ow) {
        for (int kd = 0; kd < KD; ++kd) {
            const int id = od * SD - padF + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (int kh = 0; kh < KH; ++kh) {
                const int ih = oh * SH - padT + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    const int iw = ow * SW - padL + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    const auto off = get_offset(src_d, mb, oc, id, ih, iw);
                    d += src[off];
                }
            }
        }
        int num_summands;
        if (alg == alg_kind::pooling_avg_include_padding)
            num_summands = KW * KH * KD;
        else {
            auto id_start = od * SD - padF;
            auto ih_start = oh * SH - padT;
            auto iw_start = ow * SW - padL;
            auto id_end = od * SD - padF + (KD - 1) * DD + KD;
            auto ih_end = oh * SH - padT + (KH - 1) * DH + KH;
            auto iw_end = ow * SW - padL + (KW - 1) * DW + KW;

            auto id_start_excluded
                    = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
            auto ih_start_excluded
                    = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
            auto iw_start_excluded
                    = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
            auto id_end_excluded
                    = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
            auto ih_end_excluded
                    = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
            auto iw_end_excluded
                    = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

            num_summands = (KD - id_start_excluded - id_end_excluded)
                    * (KH - ih_start_excluded - ih_end_excluded)
                    * (KW - iw_start_excluded - iw_end_excluded);
        }
        d /= num_summands;
    };

    const int MB = pd()->MB();
    const int OC = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    if (alg == alg_kind::pooling_max) {
        parallel_nd(MB, OC, OD, OH, OW,
                [&](int mb, int oc, int od, int oh, int ow) {
                    auto data_p_off = get_offset(dst_d, mb, oc, od, oh, ow);
                    auto data_l_off
                            = (((mb * OC + oc) * OD + od) * OH + oh) * OW + ow;
                    float res = numeric_limits<data_t>::lowest();
                    set_ws(mb, oc, od, oh, ow, 0);
                    ker_max(res, mb, oc, od, oh, ow);

                    ref_post_ops_t::args_t args;
                    args.ctx = &ctx;
                    args.l_offset = data_l_off;
                    args.dst_md = pd()->dst_md();
                    ref_post_ops->execute(res, args);

                    dst[data_p_off] = cpu::saturate_and_round<data_t>(res);
                });
    } else {
        parallel_nd(MB, OC, OD, OH, OW,
                [&](int mb, int oc, int od, int oh, int ow) {
                    auto data_p_off = get_offset(dst_d, mb, oc, od, oh, ow);
                    auto data_l_off
                            = (((mb * OC + oc) * OD + od) * OH + oh) * OW + ow;
                    float res = 0.f;
                    ker_avg(res, mb, oc, od, oh, ow);

                    ref_post_ops_t::args_t args;
                    args.ctx = &ctx;
                    args.l_offset = data_l_off;
                    args.dst_md = pd()->dst_md();
                    ref_post_ops->execute(res, args);

                    dst[data_p_off] = cpu::saturate_and_round<data_t>(res);
                });
    }
}

template <data_type_t data_type>
void ref_pooling_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {

    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const unsigned char *, DNNL_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper ws_d(pd()->workspace_md());

    const auto alg = pd()->desc()->alg_kind;

    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();
    const int DD = pd()->DD();
    const int DH = pd()->DH();
    const int DW = pd()->DW();

    auto ker_zero = [=](int mb, int oc) {
        for_(int id = 0; id < ID; ++id)
        for_(int ih = 0; ih < IH; ++ih)
        for (int iw = 0; iw < IW; ++iw) {
            const auto off = get_offset(diff_src_d, mb, oc, id, ih, iw);
            diff_src[off] = data_type_t(0);
        }
    };

    auto ker_max
            = [=](const data_t *d, int mb, int oc, int od, int oh, int ow) {
                  const auto ws_off = get_offset(ws_d, mb, oc, od, oh, ow);
                  const int index = ws_d.data_type() == data_type::u8
                          ? (int)ws[ws_off]
                          : ((int *)ws)[ws_off];
                  const int kd = (index / KW) / KH;
                  const int kh = (index / KW) % KH;
                  const int kw = index % KW;
                  const int id = od * SD - padF + kd * (DD + 1);
                  const int ih = oh * SH - padT + kh * (DH + 1);
                  const int iw = ow * SW - padL + kw * (DW + 1);

                  // If padding area could fit the kernel,
                  // then input displacement would be out of bounds.
                  // No need to back propagate there as padding is
                  // virtual in pooling_max case.
                  if (id < 0 || id >= ID) return;
                  if (ih < 0 || ih >= IH) return;
                  if (iw < 0 || iw >= IW) return;

                  const auto off = get_offset(diff_src_d, mb, oc, id, ih, iw);
                  diff_src[off] += d[0];
              };

    auto ker_avg = [=](const data_t *d, int mb, int oc, int od, int oh,
                           int ow) {
        int num_summands;
        if (alg == alg_kind::pooling_avg_include_padding)
            num_summands = KW * KH * KD;
        else {
            auto id_start = od * SD - padF;
            auto ih_start = oh * SH - padT;
            auto iw_start = ow * SW - padL;
            auto id_end = od * SD - padF + (KD - 1) * DD + KD;
            auto ih_end = oh * SH - padT + (KH - 1) * DH + KH;
            auto iw_end = ow * SW - padL + (KW - 1) * DW + KW;

            auto id_start_excluded
                    = id_start < 0 ? (0 - id_start - 1) / (DD + 1) + 1 : 0;
            auto ih_start_excluded
                    = ih_start < 0 ? (0 - ih_start - 1) / (DH + 1) + 1 : 0;
            auto iw_start_excluded
                    = iw_start < 0 ? (0 - iw_start - 1) / (DW + 1) + 1 : 0;
            auto id_end_excluded
                    = id_end > ID ? (id_end - ID - 1) / (DD + 1) + 1 : 0;
            auto ih_end_excluded
                    = ih_end > IH ? (ih_end - IH - 1) / (DH + 1) + 1 : 0;
            auto iw_end_excluded
                    = iw_end > IW ? (iw_end - IW - 1) / (DW + 1) + 1 : 0;

            num_summands = (KD - id_start_excluded - id_end_excluded)
                    * (KH - ih_start_excluded - ih_end_excluded)
                    * (KW - iw_start_excluded - iw_end_excluded);
        }
        for (int kd = 0; kd < KD; ++kd) {
            const int id = od * SD - padF + kd * (DD + 1);
            if (id < 0 || id >= ID) continue;
            for (int kh = 0; kh < KH; ++kh) {
                const int ih = oh * SH - padT + kh * (DH + 1);
                if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    const int iw = ow * SW - padL + kw * (DW + 1);
                    if (iw < 0 || iw >= IW) continue;

                    const auto off = get_offset(diff_src_d, mb, oc, id, ih, iw);
                    diff_src[off] += d[0] / num_summands;
                }
            }
        }
    };

    const int MB = pd()->MB();
    const int OC = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    int ow_start = max(0, utils::div_up(padL - ((KW - 1) * DW + KW) + 1, SW));
    int ow_end = min(OW, 1 + (padL + IW - 1) / SW);

    int oh_start = max(0, utils::div_up(padT - ((KH - 1) * DH + KH) + 1, SH));
    int oh_end = min(OH, 1 + (padT + IH - 1) / SH);

    int od_start = max(0, utils::div_up(padF - ((KD - 1) * DD + KD) + 1, SD));
    int od_end = min(OD, 1 + (padF + ID - 1) / SD);

    if (alg == alg_kind::pooling_max) {
        parallel_nd(MB, OC, [&](int mb, int oc) {
            ker_zero(mb, oc);
            for_(int od = od_start; od < od_end; ++od)
            for_(int oh = oh_start; oh < oh_end; ++oh)
            for (int ow = ow_start; ow < ow_end; ++ow) {
                const data_t *d
                        = &diff_dst[get_offset(diff_dst_d, mb, oc, od, oh, ow)];
                ker_max(d, mb, oc, od, oh, ow);
            }
        });
    } else {
        parallel_nd(MB, OC, [&](int mb, int oc) {
            ker_zero(mb, oc);
            for_(int od = od_start; od < od_end; ++od)
            for_(int oh = oh_start; oh < oh_end; ++oh)
            for (int ow = ow_start; ow < ow_end; ++ow) {
                const data_t *d
                        = &diff_dst[get_offset(diff_dst_d, mb, oc, od, oh, ow)];
                ker_avg(d, mb, oc, od, oh, ow);
            }
        });
    }
}

template struct ref_pooling_fwd_t<data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s32>;
template struct ref_pooling_fwd_t<data_type::bf16, data_type::f32>;
template struct ref_pooling_fwd_t<data_type::s8, data_type::s32>;
template struct ref_pooling_fwd_t<data_type::u8, data_type::s32>;

template struct ref_pooling_bwd_t<data_type::f32>;
template struct ref_pooling_bwd_t<data_type::s32>;
template struct ref_pooling_bwd_t<data_type::bf16>;
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
