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

#include "tests/test_thread.hpp"

#include "lrn/lrn.hpp"

namespace lrn {

float fast_powf(float omega, float beta) {
    if (beta == 0.75f) return 1.0f / sqrtf(sqrtf(omega) * omega);
    return 1.0f / powf(omega, beta);
}

float get_omega(const prb_t *prb, const dnn_mem_t &src, int64_t mb, int64_t c,
        int64_t d, int64_t h, int64_t w) {
    const int size = prb->ls;
    const int half_size = (size - 1) / 2;
    const int summands = compute_n_summands(prb);

    float sum = 0;
    if (prb->alg == ACROSS) {
        const int64_t c_st = MAX2(c - half_size + 0, 0);
        const int64_t c_en = MIN2(c + half_size + 1, prb->ic);

        for (int64_t cs = c_st; cs < c_en; ++cs) {
            const auto off = data_off(prb, mb, cs, d, h, w);
            const float s = src.get_elem(off);
            sum += s * s;
        }
    } else if (prb->alg == WITHIN) {
        const int64_t d_st = MAX2(d - half_size + 0, 0);
        const int64_t d_en = MIN2(d + half_size + 1, prb->id);
        const int64_t h_st = MAX2(h - half_size + 0, 0);
        const int64_t h_en = MIN2(h + half_size + 1, prb->ih);
        const int64_t w_st = MAX2(w - half_size + 0, 0);
        const int64_t w_en = MIN2(w + half_size + 1, prb->iw);

        for_(int64_t ds = d_st; ds < d_en; ++ds)
        for_(int64_t hs = h_st; hs < h_en; ++hs)
        for (int64_t ws = w_st; ws < w_en; ++ws) {
            const auto off = data_off(prb, mb, c, ds, hs, ws);
            const float s = src.get_elem(off);
            sum += s * s;
        }
    }

    return (float)(prb->k + prb->alpha * sum / summands);
}

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src, dnn_mem_t &dst) {
    dnnl::impl::parallel_nd(prb->mb, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) {
                const auto off = data_off(prb, mb, c, d, h, w);
                const float omega = get_omega(prb, src, mb, c, d, h, w);
                const float omega_in_beta = fast_powf(omega, prb->beta);
                dst.set_elem(off, src.get_elem(off) * omega_in_beta);
            });
}

void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &d_dst, dnn_mem_t &d_src) {
    const int size = prb->ls;
    const int half_size = (size - 1) / 2;
    const int summands = compute_n_summands(prb);

    dnnl::impl::parallel_nd(prb->mb, prb->ic, prb->id, prb->ih, prb->iw,
            [&](int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) {
                float A = 0, B = 0;
                if (prb->alg == ACROSS) {
                    const int64_t c_st = MAX2(c - half_size + 0, 0);
                    const int64_t c_en = MIN2(c + half_size + 1, prb->ic);

                    for (int64_t cs = c_st; cs < c_en; ++cs) {
                        const auto off = data_off(prb, mb, cs, d, h, w);
                        const float omega
                                = get_omega(prb, src, mb, cs, d, h, w);
                        const float omega_in_beta = fast_powf(omega, prb->beta);
                        const float tmp = omega_in_beta * d_dst.get_elem(off);
                        if (cs == c) A = tmp;
                        B += (tmp / omega * src.get_elem(off));
                    }
                } else if (prb->alg == WITHIN) {
                    const int64_t d_st = MAX2(d - half_size + 0, 0);
                    const int64_t d_en = MIN2(d + half_size + 1, prb->id);
                    const int64_t h_st = MAX2(h - half_size + 0, 0);
                    const int64_t h_en = MIN2(h + half_size + 1, prb->ih);
                    const int64_t w_st = MAX2(w - half_size + 0, 0);
                    const int64_t w_en = MIN2(w + half_size + 1, prb->iw);

                    for_(int64_t ds = d_st; ds < d_en; ++ds)
                    for_(int64_t hs = h_st; hs < h_en; ++hs)
                    for (int64_t ws = w_st; ws < w_en; ++ws) {
                        const auto off = data_off(prb, mb, c, ds, hs, ws);
                        const float omega
                                = get_omega(prb, src, mb, c, ds, hs, ws);
                        const float omega_in_beta = fast_powf(omega, prb->beta);
                        const float tmp = omega_in_beta * d_dst.get_elem(off);
                        if (ds == d && hs == h && ws == w) A = tmp;
                        B += (tmp / omega * src.get_elem(off));
                    }
                }
                const auto off = data_off(prb, mb, c, d, h, w);
                B *= (2.0f * prb->alpha * prb->beta * src.get_elem(off)
                        / summands);
                d_src.set_elem(off, A - B);
            });
}

} // namespace lrn
