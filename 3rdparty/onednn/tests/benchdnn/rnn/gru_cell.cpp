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

#include <stdlib.h>

#include "tests/test_thread.hpp"

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"

#include "rnn/cells.hpp"

namespace rnn {

template <typename T>
void gru_fwd_postgemm_part1_template(T func1, const prb_t &prb, float *gates_,
        const float *src_iter_, const float *bias_, float *dst_layer_) {
    AOC<const float> bias(bias_, prb.n_gates(), prb.dhc);
    AOC<const float> src_iter(src_iter_, prb.mb, prb.wc);
    AOC<float> dst_layer(dst_layer_, prb.mb, prb.wc);
    AOC<float> gates(gates_, prb.mb, prb.n_gates(), prb.dhc);

    for (int64_t i = 0; i < prb.mb; i++)
        for (int64_t k = 0; k < prb.dhc; k++) {
            gates(i, GRU_U, k) = func1(prb.linear_scales[GRU_U],
                    maybe_deq(prb, gates(i, GRU_U, k), GRU_U * prb.dhc + k)
                            + bias(GRU_U, k));
            gates(i, GRU_R, k) = func1(prb.linear_scales[GRU_R],
                    maybe_deq(prb, gates(i, GRU_R, k), GRU_R * prb.dhc + k)
                            + bias(GRU_R, k));
            dst_layer(i, k) = maybe_q(
                    prb, (maybe_deq(prb, src_iter(i, k)) * gates(i, GRU_R, k)));
        }
}

void gru_fwd_postgemm_part1(const prb_t &prb, float *gates_,
        const float *src_iter_, const float *bias_, float *dst_layer_) {
    if (prb.skip_nonlinear)
        gru_fwd_postgemm_part1_template(
                [](float scale, float a) { return scale * a; }, prb, gates_,
                src_iter_, bias_, dst_layer_);
    else
        gru_fwd_postgemm_part1_template(
                [](float scale, float a) { return logistic(a); }, prb, gates_,
                src_iter_, bias_, dst_layer_);
}

template <typename T>
void gru_fwd_postgemm_part2_template(T func1, const prb_t &prb, float *gates_,
        const float *src_iter_, const float *bias_, float *dst_layer_) {
    AOC<const float> bias(bias_, prb.n_gates(), prb.dhc);
    AOC<const float> src_iter(src_iter_, prb.mb, prb.wc);
    AOC<float> dst_layer(dst_layer_, prb.mb, prb.wc);
    AOC<float> gates(gates_, prb.mb, prb.n_gates(), prb.dhc);
    for (int64_t i = 0; i < prb.mb; i++)
        for (int64_t k = 0; k < prb.dhc; k++) {
            double U = gates(i, GRU_U, k);
            double O = func1(prb.linear_scales[GRU_O],
                    maybe_deq(prb, gates(i, GRU_O, k), GRU_O * prb.dhc + k)
                            + bias(GRU_O, k));
            dst_layer(i, k) = maybe_q(prb,
                    (float)(U * maybe_deq(prb, src_iter(i, k))
                            + (1.0 - U) * O));

            gates(i, GRU_O, k) = O;
        }
}

void gru_fwd_postgemm_part2(const prb_t &prb, float *gates_,
        const float *src_iter_, const float *bias_, float *dst_layer_) {
    if (prb.skip_nonlinear)
        gru_fwd_postgemm_part2_template(
                [](float scale, float a) { return scale * a; }, prb, gates_,
                src_iter_, bias_, dst_layer_);
    else
        gru_fwd_postgemm_part2_template(
                [](float scale, float a) { return tanhf(a); }, prb, gates_,
                src_iter_, bias_, dst_layer_);
}

void gru_fwd(const prb_t &prb, float *dst_layer_, float *gates_,
        const float *weights_layer_, const float *weights_iter_,
        const float *bias_, const float *src_layer_, const float *src_iter_) {
    AOC<const float> weights_iter(
            weights_iter_, prb.sic, prb.n_gates(), prb.dhc);
    AOC<float> gates(gates_, prb.mb, prb.n_gates(), prb.dhc);

    gemm("C", "N", "N", prb.mb, prb.n_gates() * prb.dhc, prb.slc, 1.0,
            src_layer_, prb.wc, weights_layer_, prb.n_gates() * prb.dhc, 0.0,
            gates_, prb.n_gates() * prb.dhc);
    gemm("C", "N", "N", prb.mb, (prb.n_gates() - 1) * prb.dhc, prb.sic, 1.0,
            src_iter_, prb.wc, weights_iter_, prb.n_gates() * prb.dhc, 1.0,
            gates_, prb.n_gates() * prb.dhc);

    gru_fwd_postgemm_part1(prb, gates_, src_iter_, bias_, dst_layer_);

    gemm("C", "N", "N", prb.mb, prb.dhc, prb.sic, 1.0, dst_layer_, prb.wc,
            &(weights_iter(0, GRU_O, 0)), prb.n_gates() * prb.dhc, 1.0,
            &(gates(0, GRU_O, 0)), prb.n_gates() * prb.dhc);

    gru_fwd_postgemm_part2(prb, gates_, src_iter_, bias_, dst_layer_);
}

void gru_bwd_pregemm_part1(const prb_t &prb, const float *src_iter_,
        const float *diff_dst_layer_, const float *diff_dst_iter_,
        const float *gates_, float *diff_src_iter_, float *b_gates_) {
    AOC<const float> src_iter(src_iter_, prb.mb, prb.wc);
    AOC<const float> diff_dst_layer(diff_dst_layer_, prb.mb, prb.wc);
    AOC<const float> diff_dst_iter(diff_dst_iter_, prb.mb, prb.wc);
    AOC<const float> gates(gates_, prb.mb, prb.n_gates(), prb.dhc);

    AOC<float> diff_src_iter(diff_src_iter_, prb.mb, prb.wc);
    AOC<float> b_gates(b_gates_, prb.mb, prb.n_gates(), prb.dhc);

    // do = (1 - u) * dh; do^ = one_m_square(o) * do;
    // du = (h - u) * dh; du^ = x_m_square(u) * du;
    for (int64_t ib = 0; ib < prb.mb; ib++)
        for (int64_t ih = 0; ih < prb.dhc; ih++) {
            float h = src_iter(ib, ih);
            float o = gates(ib, GRU_O, ih);
            float u = gates(ib, GRU_U, ih);
            float dh = diff_dst_layer(ib, ih) + diff_dst_iter(ib, ih);
            float du = (h - o) * dh;
            float dO = (1.0f - u) * dh;
            b_gates(ib, GRU_U, ih) = x_m_square(u) * du;
            b_gates(ib, GRU_O, ih) = one_m_square(o) * dO;
            diff_src_iter(ib, ih) = dh * u;
        }
}

void gru_bwd_pregemm_part2(const prb_t &prb, const float *src_iter_,
        const float *gates_, const float *dhr_, float *diff_src_iter_,
        float *b_gates_, float *hr_) {
    AOC<const float> src_iter(src_iter_, prb.mb, prb.wc);
    AOC<const float> gates(gates_, prb.mb, prb.n_gates(), prb.dhc);
    AOC<const float> dhr(dhr_, prb.mb, prb.dhc);
    AOC<float> diff_src_iter(diff_src_iter_, prb.mb, prb.wc);
    AOC<float> b_gates(b_gates_, prb.mb, prb.n_gates(), prb.dhc);
    AOC<float> hr(hr_, prb.mb, prb.dhc);

    // dhr = Wo do^;
    // dr = h * dhr; dr^ = x_m_square(r) * dr;
    for (int64_t ib = 0; ib < prb.mb; ib++)
        for (int64_t ih = 0; ih < prb.dhc; ih++) {
            float h = src_iter(ib, ih);
            float r = gates(ib, GRU_R, ih);
            float dr = h * dhr(ib, ih);
            hr(ib, ih) = h * r;
            diff_src_iter(ib, ih) += dhr(ib, ih) * r;
            b_gates(ib, GRU_R, ih) = x_m_square(r) * dr;
        }
}

void gru_bwd(const prb_t &prb, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_, const float *bias_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_,
        float *cell_scratchpad_) {
    AOC<const float> weights_iter(
            weights_iter_, prb.sic, prb.n_gates(), prb.dhc);

    AOC<float> diff_weights_iter(
            diff_weights_iter_, prb.sic, prb.n_gates(), prb.dhc);
    AOC<float> b_gates(b_gates_, prb.mb, prb.n_gates(), prb.dhc);

    assert(prb.dhc == prb.sic);
    float *dhr_ = cell_scratchpad_;
    float *hr_ = cell_scratchpad_ + prb.mb * prb.dhc;

    gru_bwd_pregemm_part1(prb, src_iter_, diff_dst_layer_, diff_dst_iter_,
            gates_, diff_src_iter_, b_gates_);

    gemm("C", "N", "T", prb.mb, prb.sic, prb.dhc, 1.0, &(b_gates(0, GRU_O, 0)),
            prb.n_gates() * prb.dhc, &(weights_iter(0, GRU_O, 0)),
            prb.n_gates() * prb.dhc, 0.0, dhr_, prb.dhc);

    gru_bwd_pregemm_part2(
            prb, src_iter_, gates_, dhr_, diff_src_iter_, b_gates_, hr_);

    // dWx += xdu^ | xdr^ | xdo^
    // dWh += hdu^ | ddr^ | (h * r)do^
    gemm("C", "T", "N", prb.sic, (prb.n_gates() - 1) * prb.dhc, prb.mb, 1.0,
            src_iter_, prb.wc, b_gates_, prb.n_gates() * prb.dhc, 1.0,
            diff_weights_iter_, prb.n_gates() * prb.dhc);
    gemm("C", "T", "N", prb.sic, prb.dhc, prb.mb, 1.0, hr_, prb.dhc,
            &(b_gates(0, GRU_O, 0)), prb.n_gates() * prb.dhc, 1.0,
            &(diff_weights_iter(0, GRU_O, 0)), prb.n_gates() * prb.dhc);
    gemm("C", "T", "N", prb.slc, prb.n_gates() * prb.dhc, prb.mb, 1.0,
            src_layer_, prb.wc, b_gates_, prb.n_gates() * prb.dhc, 1.0,
            diff_weights_layer_, prb.n_gates() * prb.dhc);

    // dx_next = Wxudu^ + Wxrdr^ + Wxodo^
    // dh_next = dh * u + Whudu^ + Whzdz^ + r * Whodo^
    gemm("C", "N", "T", prb.mb, prb.sic, (prb.n_gates() - 1) * prb.dhc, 1.0,
            b_gates_, prb.n_gates() * prb.dhc, weights_iter_,
            prb.n_gates() * prb.dhc, 1.0, diff_src_iter_, prb.wc);
    gemm("C", "N", "T", prb.mb, prb.slc, prb.n_gates() * prb.dhc, 1.0, b_gates_,
            prb.n_gates() * prb.dhc, weights_layer_, prb.n_gates() * prb.dhc,
            0.0, diff_src_layer_, prb.wc);

    gates_reduction(prb, b_gates_, diff_bias_);
}

} // namespace rnn
