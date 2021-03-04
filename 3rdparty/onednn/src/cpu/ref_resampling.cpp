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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/resampling_utils.hpp"

#include "cpu/ref_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

static inline dim_t get_offset(
        const memory_desc_wrapper &data_d, int n, int c, int d, int h, int w) {
    if (data_d.ndims() == 5)
        return data_d.off(n, c, d, h, w);
    else if (data_d.ndims() == 4)
        return data_d.off(n, c, h, w);
    else
        return data_d.off(n, c, w);
}

using namespace resampling_utils;

template <impl::data_type_t data_type>
void ref_resampling_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    if (this->pd()->has_zero_dim_memory()) return;

    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto alg = pd()->desc()->alg_kind;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    auto lin_interp = [&](float c0, float c1, float w) {
        return c0 * w + c1 * (1 - w);
    };
    auto bilin_interp = [&](float c00, float c01, float c10, float c11,
                                float w0, float w1) {
        return lin_interp(
                lin_interp(c00, c10, w0), lin_interp(c01, c11, w0), w1);
    };
    auto trilin_interp = [&](float c000, float c001, float c010, float c011,
                                 float c100, float c101, float c110, float c111,
                                 float w0, float w1, float w2) {
        return lin_interp(bilin_interp(c000, c010, c100, c110, w0, w1),
                bilin_interp(c001, c011, c101, c111, w0, w1), w2);
    };
    parallel_nd(MB, C, OD, OH, OW,
            [&](dim_t mb, dim_t ch, dim_t od, dim_t oh, dim_t ow) {
                if (alg == alg_kind::resampling_nearest) {
                    const dim_t id = nearest_idx(od, OD, ID);
                    const dim_t ih = nearest_idx(oh, OH, IH);
                    const dim_t iw = nearest_idx(ow, OW, IW);
                    dst[get_offset(dst_d, mb, ch, od, oh, ow)]
                            = src[get_offset(src_d, mb, ch, id, ih, iw)];
                } else if (alg == alg_kind::resampling_linear) {
                    // Trilinear interpolation (linear interpolation on a 3D spatial
                    // tensor) can be expressed as linear interpolation along
                    // dimension x followed by interpolation along dimension y and z
                    //      C011--C11--C111
                    //     -          - |
                    //   -          -   |
                    //C001--C01--C111   |
                    // -     .C   -    C110
                    // -          -    -
                    // -          -  -
                    //C000--C00--C100
                    auto id = linear_coeffs_t(od, OD, ID);
                    auto iw = linear_coeffs_t(ow, OW, IW);
                    auto ih = linear_coeffs_t(oh, OH, IH);
                    data_t src_l[8] = {0};
                    for_(int i = 0; i < 2; i++)
                    for_(int j = 0; j < 2; j++)
                    for (int k = 0; k < 2; k++) {
                        src_l[4 * i + 2 * j + k] = src[get_offset(src_d, mb, ch,
                                id.idx[i], ih.idx[j], iw.idx[k])];
                    }
                    dst[get_offset(dst_d, mb, ch, od, oh, ow)]
                            = trilin_interp(src_l[0], src_l[1], src_l[2],
                                    src_l[3], src_l[4], src_l[5], src_l[6],
                                    src_l[7], id.wei[0], ih.wei[0], iw.wei[0]);
                }
            });
}

template struct ref_resampling_fwd_t<data_type::f32>;
template struct ref_resampling_fwd_t<data_type::bf16>;

template <impl::data_type_t data_type>
void ref_resampling_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    if (this->pd()->has_zero_dim_memory()) return;

    const auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const auto alg = pd()->desc()->alg_kind;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();

    if (alg == alg_kind::resampling_nearest) {
        parallel_nd(MB, C, ID, IH, IW,
                [&](dim_t mb, dim_t ch, dim_t id, dim_t ih, dim_t iw) {
                    const dim_t od_start
                            = ceil_idx(((float)id * OD / ID) - 0.5f);
                    const dim_t oh_start
                            = ceil_idx(((float)ih * OH / IH) - 0.5f);
                    const dim_t ow_start
                            = ceil_idx(((float)iw * OW / IW) - 0.5f);

                    const dim_t od_end
                            = ceil_idx(((id + 1.f) * OD / ID) - 0.5f);
                    const dim_t oh_end
                            = ceil_idx(((ih + 1.f) * OH / IH) - 0.5f);
                    const dim_t ow_end
                            = ceil_idx(((iw + 1.f) * OW / IW) - 0.5f);

                    float ds = 0;
                    for_(dim_t od = od_start; od < od_end; od++)
                    for_(dim_t oh = oh_start; oh < oh_end; oh++)
                    for (dim_t ow = ow_start; ow < ow_end; ow++)
                        ds += diff_dst[get_offset(
                                diff_dst_d, mb, ch, od, oh, ow)];
                    diff_src[get_offset(diff_src_d, mb, ch, id, ih, iw)] = ds;
                });
        return;
    } else {
        parallel_nd(MB, C, ID, IH, IW,
                [&](dim_t mb, dim_t ch, dim_t id, dim_t ih, dim_t iw) {
                    bwd_linear_coeffs_t d(id, OD, ID);
                    bwd_linear_coeffs_t h(ih, OH, IH);
                    bwd_linear_coeffs_t w(iw, OW, IW);

                    float ds = 0;
                    for_(int i = 0; i < 2; i++)
                    for_(int j = 0; j < 2; j++)
                    for_(int k = 0; k < 2; k++)
                    for_(dim_t od = d.start[i]; od < d.end[i]; od++)
                    for_(dim_t oh = h.start[j]; oh < h.end[j]; oh++)
                    for (dim_t ow = w.start[k]; ow < w.end[k]; ow++) {
                        const float weight_d = linear_weight(i, od, OD, ID);
                        const float weight_h = linear_weight(j, oh, OH, IH);
                        const float weight_w = linear_weight(k, ow, OW, IW);

                        float dd = diff_dst[get_offset(
                                diff_dst_d, mb, ch, od, oh, ow)];
                        ds += dd * weight_d * weight_h * weight_w;
                    }
                    diff_src[get_offset(diff_src_d, mb, ch, id, ih, iw)] = ds;
                });
    }
}

template struct ref_resampling_bwd_t<data_type::f32>;
template struct ref_resampling_bwd_t<data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
