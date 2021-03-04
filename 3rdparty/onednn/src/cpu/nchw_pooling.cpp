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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/nchw_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace nstl;

template <data_type_t d_type>
void nchw_pooling_fwd_t<d_type>::execute_forward(const exec_ctx_t &ctx) const {
    auto alg = pd()->desc()->alg_kind;

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(unsigned char *, DNNL_ARG_WORKSPACE);

    const memory_desc_wrapper ws_d(pd()->workspace_md());
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
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

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto set_ws = [=](int mb, int c, int od, int oh, int ow, int value) {
        if (ws) {
            assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
            size_t ws_offset = (size_t)OW * OH * OD * C * mb
                    + (size_t)OW * OH * OD * c + (size_t)OW * OH * od
                    + (size_t)OW * oh + (size_t)ow;
            if (ws_dt == data_type::u8) {
                assert(0 <= value
                        && value <= numeric_limits<typename prec_traits<
                                        data_type::u8>::type>::max());
                ws[ws_offset] = value;
            } else
                reinterpret_cast<int *>(ws)[ws_offset] = value;
        }
    };

    auto ker_max = [=](data_t *d, int mb, int c, int od, int oh, int ow) {
        for_(int kd = 0; kd < KD; ++kd)
        for_(int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            const int id = od * SD - padF + kd;
            const int ih = oh * SH - padT + kh;
            const int iw = ow * SW - padL + kw;

            if (id < 0 || id >= ID) continue;
            if (ih < 0 || ih >= IH) continue;
            if (iw < 0 || iw >= IW) continue;

            auto src_offset = (size_t)IW * IH * ID * C * mb
                    + (size_t)IW * IH * ID * c + (size_t)IW * IH * id
                    + (size_t)IW * ih + (size_t)iw;
            auto s = src[src_offset];
            if (s > d[0]) {
                d[0] = s;
                set_ws(mb, c, od, oh, ow, kd * KH * KW + kh * KW + kw);
            }
        }
    };

    auto ker_avg = [=](data_t *d, int mb, int c, int od, int oh, int ow) {
        auto id_start = apply_offset(od * SD, padF);
        auto ih_start = apply_offset(oh * SH, padT);
        auto iw_start = apply_offset(ow * SW, padL);
        auto id_end = min(od * SD - padF + KD, ID);
        auto ih_end = min(oh * SH - padT + KH, IH);
        auto iw_end = min(ow * SW - padL + KW, IW);

        auto num_summands = (alg == alg_kind::pooling_avg_include_padding)
                ? KD * KW * KH
                : (id_end - id_start) * (ih_end - ih_start)
                        * (iw_end - iw_start);

        for_(int id = id_start; id < id_end; ++id)
        for_(int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            auto src_offset = (size_t)IW * IH * ID * C * mb
                    + (size_t)IW * IH * ID * c + (size_t)IW * IH * id
                    + (size_t)IW * ih + (size_t)iw;
            d[0] += src[src_offset];
        }

        d[0] = out_round<data_t>((float)d[0] / num_summands);
    };

    if (alg == alg_kind::pooling_max) {
        parallel_nd(
                MB, C, OD, OH, OW, [&](int mb, int c, int od, int oh, int ow) {
                    size_t dst_offset = (size_t)OW * OH * OD * C * mb
                            + (size_t)OW * OH * OD * c + (size_t)OW * OH * od
                            + (size_t)OW * oh + (size_t)ow;
                    data_t *d = &dst[dst_offset];
                    d[0] = numeric_limits<data_t>::lowest();
                    set_ws(mb, c, od, oh, ow, 0);
                    ker_max(d, mb, c, od, oh, ow);
                });
    } else {
        parallel_nd(
                MB, C, OD, OH, OW, [&](int mb, int c, int od, int oh, int ow) {
                    size_t dst_offset = (size_t)OW * OH * OD * C * mb
                            + (size_t)OW * OH * OD * c + (size_t)OW * OH * od
                            + (size_t)OW * oh + (size_t)ow;
                    data_t *d = &dst[dst_offset];
                    d[0] = 0;
                    ker_avg(d, mb, c, od, oh, ow);
                });
    }
}

template <>
void nchw_pooling_fwd_t<data_type::bf16>::execute_forward(
        const exec_ctx_t &ctx) const {

    auto alg = pd()->desc()->alg_kind;

    auto src = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(bfloat16_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(unsigned char *, DNNL_ARG_WORKSPACE);

    auto scratchpad = ctx.get_scratchpad_grantor();
    float *bf16cvt_wsp = scratchpad.template get<float>(
            memory_tracking::names::key_pool_src_bf16cvt);

    const memory_desc_wrapper ws_d(pd()->workspace_md());
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
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

    const size_t simd_w = 16;
    const size_t src_size = MB * C * ID * IH * IW;
    const size_t blocked_size = src_size / simd_w;
    const size_t tail_size = src_size % simd_w;

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto set_ws = [=](int mb, int c, int od, int oh, int ow, int value) {
        if (ws) {
            assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
            size_t ws_offset = (size_t)OW * OH * OD * C * mb
                    + (size_t)OW * OH * OD * c + (size_t)OW * OH * od
                    + (size_t)OW * oh + (size_t)ow;
            if (ws_dt == data_type::u8) {
                assert(0 <= value
                        && value <= numeric_limits<typename prec_traits<
                                        data_type::u8>::type>::max());
                ws[ws_offset] = value;
            } else
                reinterpret_cast<int *>(ws)[ws_offset] = value;
        }
    };

    auto ker_max = [=](float *d, int mb, int c, int od, int oh, int ow) {
        for_(int kd = 0; kd < KD; ++kd)
        for_(int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            const int id = od * SD - padF + kd;
            const int ih = oh * SH - padT + kh;
            const int iw = ow * SW - padL + kw;

            if (id < 0 || id >= ID) continue;
            if (ih < 0 || ih >= IH) continue;
            if (iw < 0 || iw >= IW) continue;

            auto src_offset = (size_t)IW * IH * ID * C * mb
                    + (size_t)IW * IH * ID * c + (size_t)IW * IH * id
                    + (size_t)IW * ih + (size_t)iw;
            auto s = bf16cvt_wsp[src_offset];

            if (s > d[0]) {
                d[0] = s;
                set_ws(mb, c, od, oh, ow, kd * KH * KW + kh * KW + kw);
            }
        }
    };

    auto ker_avg = [=](float *d, int mb, int c, int od, int oh, int ow) {
        auto id_start = apply_offset(od * SD, padF);
        auto ih_start = apply_offset(oh * SH, padT);
        auto iw_start = apply_offset(ow * SW, padL);
        auto id_end = min(od * SD - padF + KD, ID);
        auto ih_end = min(oh * SH - padT + KH, IH);
        auto iw_end = min(ow * SW - padL + KW, IW);

        auto num_summands = (alg == alg_kind::pooling_avg_include_padding)
                ? KD * KW * KH
                : (id_end - id_start) * (ih_end - ih_start)
                        * (iw_end - iw_start);

        for_(int id = id_start; id < id_end; ++id)
        for_(int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            auto src_offset = (size_t)IW * IH * ID * C * mb
                    + (size_t)IW * IH * ID * c + (size_t)IW * IH * id
                    + (size_t)IW * ih + (size_t)iw;
            d[0] += bf16cvt_wsp[src_offset];
        }

        d[0] = out_round<float>((float)d[0] / num_summands);
    };
    parallel_nd(blocked_size, [&](size_t i) {
        cvt_bfloat16_to_float(
                &bf16cvt_wsp[i * simd_w], &src[i * simd_w], simd_w);
    });
    if (tail_size)
        cvt_bfloat16_to_float(&bf16cvt_wsp[blocked_size * simd_w],
                &src[blocked_size * simd_w], tail_size);
    if (alg == alg_kind::pooling_max) {
        parallel_nd(
                MB, C, OD, OH, OW, [&](int mb, int c, int od, int oh, int ow) {
                    size_t dst_offset = (size_t)OW * OH * OD * C * mb
                            + (size_t)OW * OH * OD * c + (size_t)OW * OH * od
                            + (size_t)OW * oh + (size_t)ow;
                    float d_fp32 = numeric_limits<bfloat16_t>::lowest();

                    set_ws(mb, c, od, oh, ow, 0);

                    ker_max(&d_fp32, mb, c, od, oh, ow);

                    dst[dst_offset] = (bfloat16_t)d_fp32;
                });
    } else {
        parallel_nd(
                MB, C, OD, OH, OW, [&](int mb, int c, int od, int oh, int ow) {
                    size_t dst_offset = (size_t)OW * OH * OD * C * mb
                            + (size_t)OW * OH * OD * c + (size_t)OW * OH * od
                            + (size_t)OW * oh + (size_t)ow;
                    float d_fp32 = 0.0f;

                    ker_avg(&d_fp32, mb, c, od, oh, ow);

                    dst[dst_offset] = (bfloat16_t)d_fp32;
                });
    }
}

template <data_type_t d_type>
void nchw_pooling_bwd_t<d_type>::execute_backward(const exec_ctx_t &ctx) const {
    auto alg = pd()->desc()->alg_kind;
    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;
    const bool is_2d = pd()->desc()->diff_src_desc.ndims == 4;

    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const unsigned char *, DNNL_ARG_WORKSPACE);

    const memory_desc_wrapper ws_d(pd()->workspace_md());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
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

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_zero = [=](int mb, int c) {
        size_t diff_src_offset
                = (size_t)mb * C * ID * IH * IW + (size_t)c * ID * IH * IW;
        for_(int id = 0; id < ID; ++id)
        for_(int ih = 0; ih < IH; ++ih)
        for (int iw = 0; iw < IW; ++iw) {
            diff_src[diff_src_offset++] = 0;
        }
    };

    auto ker_max = [=](const data_t *d, int mb, int c, int od, int oh, int ow) {
        auto b_c = ws_d.blocking_desc().inner_nblks == 0
                ? 1
                : ws_d.blocking_desc().inner_blks[0];
        auto ws_offset = (is_3d ? ws_d.blk_off(mb, c / b_c, od, oh, ow)
                                : is_2d ? ws_d.blk_off(mb, c / b_c, oh, ow)
                                        : ws_d.blk_off(mb, c / b_c, ow))
                + c % b_c;

        const int index = ws_d.data_type() == data_type::u8
                ? (int)ws[ws_offset]
                : ((const int *)ws)[ws_offset];
        const int kw = index % KW;
        const int kh = (index / KW) % KH;
        const int kd = (index / KW) / KH;

        const int id = od * SD - padF + kd;
        const int ih = oh * SH - padT + kh;
        const int iw = ow * SW - padL + kw;

        // If padding area could fit the kernel,
        // then input displacement would be out of bounds.
        // No need to back propagate there as padding is
        // virtual in pooling_max case.
        if (id < 0 || id >= ID) return;
        if (ih < 0 || ih >= IH) return;
        if (iw < 0 || iw >= IW) return;

        size_t diff_src_offset = (size_t)mb * C * ID * IH * IW
                + (size_t)c * ID * IH * IW + (size_t)id * IH * IW
                + (size_t)ih * IW + (size_t)iw;
        diff_src[diff_src_offset] += d[0];
    };

    auto ker_avg = [=](const data_t *d, int mb, int c, int od, int oh, int ow) {
        auto id_start = apply_offset(od * SD, padF);
        auto ih_start = apply_offset(oh * SH, padT);
        auto iw_start = apply_offset(ow * SW, padL);
        auto id_end = min(od * SD - padF + KD, ID);
        auto ih_end = min(oh * SH - padT + KH, IH);
        auto iw_end = min(ow * SW - padL + KW, IW);

        size_t num_summands = (alg == alg_kind::pooling_avg_include_padding)
                ? (size_t)KW * KH * KD
                : (size_t)(id_end - id_start) * (ih_end - ih_start)
                        * (iw_end - iw_start);

        for_(int id = id_start; id < id_end; ++id)
        for_(int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            size_t diff_src_offset = (size_t)mb * C * ID * IH * IW
                    + (size_t)c * ID * IH * IW + (size_t)id * IH * IW
                    + (size_t)ih * IW + (size_t)iw;
            diff_src[diff_src_offset] += d[0] / num_summands;
        }
    };

    int ow_start = max(0, utils::div_up(padL - KW + 1, SW));
    int ow_end = min(OW, 1 + (padL + IW - 1) / SW);

    int oh_start = max(0, utils::div_up(padT - KH + 1, SH));
    int oh_end = min(OH, 1 + (padT + IH - 1) / SH);

    int od_start = max(0, utils::div_up(padF - KD + 1, SD));
    int od_end = min(OD, 1 + (padF + ID - 1) / SD);

    if (alg == alg_kind::pooling_max) {
        parallel_nd(MB, C, [&](int mb, int c) {
            size_t diff_dst_offset_b
                    = (size_t)mb * C * OD * OH * OW + (size_t)c * OD * OH * OW;
            ker_zero(mb, c);
            for_(int od = od_start; od < od_end; ++od)
            for (int oh = oh_start; oh < oh_end; ++oh) {
                size_t diff_dst_offset = diff_dst_offset_b
                        + (size_t)od * OH * OW + (size_t)oh * OW;
                for (int ow = ow_start; ow < ow_end; ++ow) {
                    const data_t *d = &diff_dst[diff_dst_offset + ow];
                    ker_max(d, mb, c, od, oh, ow);
                }
            }
        });
    } else {
        parallel_nd(MB, C, [&](int mb, int c) {
            size_t diff_dst_offset_b
                    = (size_t)mb * C * OD * OH * OW + (size_t)c * OD * OH * OW;
            ker_zero(mb, c);
            for_(int od = od_start; od < od_end; ++od)
            for (int oh = oh_start; oh < oh_end; ++oh) {
                size_t diff_dst_offset = diff_dst_offset_b
                        + (size_t)od * OH * OW + (size_t)oh * OW;
                for (int ow = ow_start; ow < ow_end; ++ow) {
                    const data_t *d = &diff_dst[diff_dst_offset + ow];
                    ker_avg(d, mb, c, od, oh, ow);
                }
            }
        });
    }
}

template <>
void nchw_pooling_bwd_t<data_type::bf16>::execute_backward(
        const exec_ctx_t &ctx) const {

    auto alg = pd()->desc()->alg_kind;
    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;
    const bool is_2d = pd()->desc()->diff_src_desc.ndims == 4;

    auto diff_src = CTX_OUT_MEM(bfloat16_t *, DNNL_ARG_DIFF_SRC);
    auto diff_dst = CTX_IN_MEM(const bfloat16_t *, DNNL_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const unsigned char *, DNNL_ARG_WORKSPACE);

    auto scratchpad = ctx.get_scratchpad_grantor();
    float *bf16cvt_src = scratchpad.template get<float>(
            memory_tracking::names::key_pool_src_bf16cvt);
    float *bf16cvt_dst = scratchpad.template get<float>(
            memory_tracking::names::key_pool_dst_bf16cvt);

    const memory_desc_wrapper ws_d(pd()->workspace_md());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
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

    const size_t dst_sp_size = pd()->OD() * pd()->OH() * pd()->OW();
    const size_t src_sp_size = pd()->ID() * pd()->IH() * pd()->IW();

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_zero = [=](float *diff_src, int c_block_size) {
        size_t diff_src_offset = 0;
        for_(int c = 0; c < c_block_size; ++c)
        for_(int id = 0; id < ID; ++id)
        for_(int ih = 0; ih < IH; ++ih)
        for (int iw = 0; iw < IW; ++iw) {
            diff_src[diff_src_offset++] = 0.0f;
        }
    };

    auto ker_max = [=](const float *d, float *diff_src, int mb, int c, int od,
                           int oh, int ow) {
        auto b_c = ws_d.blocking_desc().inner_nblks == 0
                ? 1
                : ws_d.blocking_desc().inner_blks[0];
        auto ws_offset = (is_3d ? ws_d.blk_off(mb, c / b_c, od, oh, ow)
                                : is_2d ? ws_d.blk_off(mb, c / b_c, oh, ow)
                                        : ws_d.blk_off(mb, c / b_c, ow))
                + c % b_c;

        const int index = ws_d.data_type() == data_type::u8
                ? (int)ws[ws_offset]
                : ((const int *)ws)[ws_offset];
        const int kw = index % KW;
        const int kh = (index / KW) % KH;
        const int kd = (index / KW) / KH;

        const int id = od * SD - padF + kd;
        const int ih = oh * SH - padT + kh;
        const int iw = ow * SW - padL + kw;

        // If padding area could fit the kernel,
        // then input displacement would be out of bounds.
        // No need to back propagate there as padding is
        // virtual in pooling_max case.
        if (id < 0 || id >= ID) return;
        if (ih < 0 || ih >= IH) return;
        if (iw < 0 || iw >= IW) return;

        size_t diff_src_offset
                = (size_t)id * IH * IW + (size_t)ih * IW + (size_t)iw;
        diff_src[diff_src_offset] += d[0];
    };

    auto ker_avg = [=](const float *d, float *diff_src, int mb, int c, int od,
                           int oh, int ow) {
        auto id_start = apply_offset(od * SD, padF);
        auto ih_start = apply_offset(oh * SH, padT);
        auto iw_start = apply_offset(ow * SW, padL);
        auto id_end = min(od * SD - padF + KD, ID);
        auto ih_end = min(oh * SH - padT + KH, IH);
        auto iw_end = min(ow * SW - padL + KW, IW);

        size_t num_summands = (alg == alg_kind::pooling_avg_include_padding)
                ? (size_t)KW * KH * KD
                : (size_t)(id_end - id_start) * (ih_end - ih_start)
                        * (iw_end - iw_start);

        for_(int id = id_start; id < id_end; ++id)
        for_(int ih = ih_start; ih < ih_end; ++ih)
        for (int iw = iw_start; iw < iw_end; ++iw) {
            size_t diff_src_offset
                    = (size_t)id * IH * IW + (size_t)ih * IW + (size_t)iw;
            diff_src[diff_src_offset] += d[0] / num_summands;
        }
    };

    int ow_start = max(0, utils::div_up(padL - KW + 1, SW));
    int ow_end = min(OW, 1 + (padL + IW - 1) / SW);

    int oh_start = max(0, utils::div_up(padT - KH + 1, SH));
    int oh_end = min(OH, 1 + (padT + IH - 1) / SH);

    int od_start = max(0, utils::div_up(padF - KD + 1, SD));
    int od_end = min(OD, 1 + (padF + ID - 1) / SD);

    dim_t c_blk = pd()->channel_block_size_;
    int c_blk_tail = C % c_blk;
    if (alg == alg_kind::pooling_max) {
        parallel_nd_ext(0, MB, utils::div_up(C, c_blk),
                [&](int ithr, int, int mb, int cb) {
                    bool is_last_c_block
                            = c_blk_tail > 0 && (cb + 1) * c_blk > C;
                    int curr_c_block = is_last_c_block ? c_blk_tail : c_blk;
                    size_t diff_dst_offset_b
                            = ((size_t)mb * C + (size_t)cb * c_blk) * OD * OH
                            * OW;
                    size_t diff_src_offset
                            = ((size_t)mb * C + (size_t)cb * c_blk) * ID * IH
                            * IW;
                    float *diff_dst_fp32
                            = &bf16cvt_dst[ithr * dst_sp_size * c_blk];
                    float *diff_src_fp32
                            = &bf16cvt_src[ithr * src_sp_size * c_blk];

                    ker_zero(diff_src_fp32, curr_c_block);

                    cvt_bfloat16_to_float(diff_dst_fp32,
                            &diff_dst[diff_dst_offset_b],
                            dst_sp_size * curr_c_block);

                    for_(int c = 0; c < curr_c_block; ++c)
                    for_(int od = od_start; od < od_end; ++od)
                    for (int oh = oh_start; oh < oh_end; ++oh) {
                        size_t diff_dst_offset = (size_t)c * OD * OH * OW
                                + (size_t)od * OH * OW + (size_t)oh * OW;
                        for (int ow = ow_start; ow < ow_end; ++ow) {
                            const float *d
                                    = &diff_dst_fp32[diff_dst_offset + ow];
                            ker_max(d, &diff_src_fp32[c * ID * IH * IW], mb,
                                    cb * c_blk + c, od, oh, ow);
                        }
                    }
                    cvt_float_to_bfloat16(&diff_src[diff_src_offset],
                            diff_src_fp32, src_sp_size * curr_c_block);
                });
    } else {
        parallel_nd_ext(0, MB, utils::div_up(C, c_blk),
                [&](int ithr, int, int mb, int cb) {
                    bool is_last_c_block
                            = c_blk_tail > 0 && (cb + 1) * c_blk > C;
                    int curr_c_block = is_last_c_block ? c_blk_tail : c_blk;
                    size_t diff_dst_offset_b = (size_t)mb * C * OD * OH * OW
                            + (size_t)cb * c_blk * OD * OH * OW;
                    float *diff_dst_fp32
                            = &bf16cvt_dst[ithr * dst_sp_size * c_blk];
                    size_t diff_src_offset = (size_t)mb * C * ID * IH * IW
                            + (size_t)cb * c_blk * ID * IH * IW;
                    float *diff_src_fp32
                            = &bf16cvt_src[ithr * src_sp_size * c_blk];

                    ker_zero(diff_src_fp32, curr_c_block);

                    cvt_bfloat16_to_float(diff_dst_fp32,
                            &diff_dst[diff_dst_offset_b],
                            dst_sp_size * curr_c_block);
                    for_(int c = 0; c < curr_c_block; ++c)
                    for_(int od = od_start; od < od_end; ++od)
                    for (int oh = oh_start; oh < oh_end; ++oh) {
                        size_t diff_dst_offset = (size_t)c * OD * OH * OW
                                + (size_t)od * OH * OW + (size_t)oh * OW;
                        for (int ow = ow_start; ow < ow_end; ++ow) {
                            const float *d
                                    = &diff_dst_fp32[diff_dst_offset + ow];
                            ker_avg(d, &diff_src_fp32[c * ID * IH * IW], mb,
                                    cb * c_blk + c, od, oh, ow);
                        }
                    }
                    cvt_float_to_bfloat16(&diff_src[diff_src_offset],
                            diff_src_fp32, src_sp_size * curr_c_block);
                });
    }
}
template struct nchw_pooling_fwd_t<data_type::f32>;
template struct nchw_pooling_bwd_t<data_type::f32>;
template struct nchw_pooling_fwd_t<data_type::bf16>;
template struct nchw_pooling_bwd_t<data_type::bf16>;
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
