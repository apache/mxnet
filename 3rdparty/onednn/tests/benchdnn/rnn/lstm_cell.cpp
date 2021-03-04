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

template <typename T1, typename T2>
void lstm_fwd_postgemm_template(T1 func1, T2 func2, const prb_t &prb,
        float *gates_, const float *weights_peephole_, const float *bias_,
        const float *src_iter_c_, float *dst_layer_, float *dst_iter_c_) {
    AOC<float> gates(gates_, prb.mb, prb.n_gates(), prb.dhc);
    AOC<const float> weights_peephole(weights_peephole_, 3, prb.dhc);
    AOC<const float> bias(bias_, prb.n_gates(), prb.dhc);
    AOC<const float> src_iter_c(src_iter_c_, prb.mb, prb.wc);
    AOC<float> dst_layer(dst_layer_, prb.mb, prb.wc);
    AOC<float> dst_iter_c(dst_iter_c_, prb.mb, prb.wc);

    // run the eltwise
    dnnl::impl::parallel_nd(prb.mb, [&](int64_t ib) {
        for (int64_t ih = 0; ih < prb.dhc; ih++) {
            float peephole_extra_i = 0, peephole_extra_f = 0;
            if (prb.is_lstm_peephole()) {
                peephole_extra_i = weights_peephole(0, ih) * src_iter_c(ib, ih);
                peephole_extra_f = weights_peephole(1, ih) * src_iter_c(ib, ih);
            }

            gates(ib, LSTM_I, ih) = func1(prb.linear_scales[LSTM_I],
                    maybe_deq(prb, gates(ib, LSTM_I, ih), LSTM_I * prb.dhc + ih)
                            + peephole_extra_i + bias(LSTM_I, ih));
            gates(ib, LSTM_F, ih) = func1(prb.linear_scales[LSTM_F],
                    maybe_deq(prb, gates(ib, LSTM_F, ih), LSTM_F * prb.dhc + ih)
                            + peephole_extra_f + bias(LSTM_F, ih));

            gates(ib, LSTM_C, ih) = func2(prb.linear_scales[LSTM_C],
                    maybe_deq(prb, gates(ib, LSTM_C, ih), LSTM_C * prb.dhc + ih)
                            + bias(LSTM_C, ih));

            // compute C_t_l and H_t_l
            float tmp = gates(ib, LSTM_F, ih) * src_iter_c(ib, ih)
                    + gates(ib, LSTM_I, ih) * gates(ib, LSTM_C, ih);
            dst_iter_c(ib, ih) = tmp;

            float peephole_extra_o = 0;
            if (prb.is_lstm_peephole())
                peephole_extra_o = weights_peephole(2, ih) * tmp;

            gates(ib, LSTM_O, ih) = func1(prb.linear_scales[LSTM_O],
                    maybe_deq(prb, gates(ib, LSTM_O, ih), LSTM_O * prb.dhc + ih)
                            + peephole_extra_o + bias(LSTM_O, ih));

            dst_layer(ib, ih) = maybe_q(
                    prb, gates(ib, LSTM_O, ih) * func2(prb.linear_cscale, tmp));

            for (int64_t ig = 0; ig < 4; ig++) {
                BENCHDNN_PRINT(80,
                        "activation 1 a[" IFMT "][" IFMT "][" IFMT "] = %.7f\n",
                        ib, ig, ih, gates(ib, ig, ih));
            }
            BENCHDNN_PRINT(80, "recomp tmp(%a) cin(%a) ht(%a)\n", tmp,
                    src_iter_c(ib, ih), dst_layer(ib, ih));
        }
    });
}

void lstm_fwd_postgemm(const prb_t &prb, float *gates_,
        const float *weights_peephole_, const float *bias_,
        const float *src_iter_c_, float *dst_layer_, float *dst_iter_c_) {
    if (prb.skip_nonlinear)
        lstm_fwd_postgemm_template(
                [](float scale, float val) { return scale * val; },
                [](float scale, float val) { return scale * val; }, prb, gates_,
                weights_peephole_, bias_, src_iter_c_, dst_layer_, dst_iter_c_);
    else
        lstm_fwd_postgemm_template(
                [](float scale, float val) { return logistic(val); },
                [](float scale, float val) { return tanhf(val); }, prb, gates_,
                weights_peephole_, bias_, src_iter_c_, dst_layer_, dst_iter_c_);
}

void lstm_fwd(const prb_t &prb, float *dst_layer_, float *dst_iter_,
        float *dst_iter_c_, float *gates_, float *ht_,
        const float *weights_layer_, const float *weights_iter_,
        const float *weights_peephole_, const float *weights_projection_,
        const float *bias_, const float *src_layer_, const float *src_iter_,
        const float *src_iter_c_) {

    gemm("C", "N", "N", prb.mb, prb.n_gates() * prb.dhc, prb.slc, 1.0,
            src_layer_, prb.wc, weights_layer_, prb.n_gates() * prb.dhc, 0.0,
            gates_, prb.n_gates() * prb.dhc);
    gemm("C", "N", "N", prb.mb, prb.n_gates() * prb.dhc, prb.sic, 1.0,
            src_iter_, prb.wc, weights_iter_, prb.n_gates() * prb.dhc, 1.0,
            gates_, prb.n_gates() * prb.dhc);

    // if lstmp, we use the workspace to write the postgemm output
    auto dst_postgemm = prb.is_lstm_projection() ? ht_ : dst_layer_;
    lstm_fwd_postgemm(prb, gates_, weights_peephole_, bias_, src_iter_c_,
            dst_postgemm, dst_iter_c_);

    assert(dst_layer_ == dst_iter_);
    if (prb.is_lstm_projection()) {
        gemm("C", "N", "N", prb.mb, prb.dic, prb.dhc, 1.0, dst_postgemm, prb.wc,
                weights_projection_, prb.dic, 0.0, dst_layer_, prb.wc);
    } else {
        assert(prb.dic == prb.dhc);
    }
}

template <typename T1>
void lstm_bwd_pregemm_template(T1 func1, const prb_t &prb,
        const float *src_iter_c_, const float *dst_iter_c_,
        const float *weights_peephole_, const float *diff_hidden_state_,
        const float *diff_dst_iter_c_, const float *gates_,
        float *diff_src_iter_c_, float *b_gates_) {
    AOC<const float> src_iter_c(src_iter_c_, prb.mb, prb.wc);
    AOC<const float> dst_iter_c(dst_iter_c_, prb.mb, prb.wc);
    AOC<const float> weights_peephole(weights_peephole_, 3, prb.dhc);
    AOC<const float> diff_hidden_state(diff_hidden_state_, prb.mb, prb.dhc);
    AOC<const float> diff_dst_iter_c(diff_dst_iter_c_, prb.mb, prb.wc);
    AOC<const float> gates(gates_, prb.mb, prb.n_gates(), prb.dhc);
    AOC<float> diff_src_iter_c(diff_src_iter_c_, prb.mb, prb.wc);
    AOC<float> b_gates(b_gates_, prb.mb, prb.n_gates(), prb.dhc);

    for (int64_t ib = 0; ib < prb.mb; ib++)
        for (int64_t ih = 0; ih < prb.dhc; ih++) {
            BENCHDNN_PRINT(80, "rnn_single_bwd: ib = " IFMT " ih = " IFMT "\n",
                    ib, ih);
            float hi = gates(ib, LSTM_I, ih);
            float hf = gates(ib, LSTM_F, ih);
            float hc = gates(ib, LSTM_C, ih);
            float ho = gates(ib, LSTM_O, ih);

            float dh = diff_hidden_state(ib, ih);

            float tanhC = func1(prb.linear_cscale, dst_iter_c(ib, ih));
            float dho = tanhC * dh;
            b_gates(ib, LSTM_O, ih) = x_m_square(ho) * dho;

            float dc = diff_dst_iter_c(ib, ih);
            dc += ho * dh * one_m_square(tanhC);

            if (prb.is_lstm_peephole())
                dc += b_gates(ib, LSTM_O, ih) * weights_peephole(2, ih);

            float dc_tm1 = hf * dc;

            float c_old = src_iter_c(ib, ih);
            float dhf = c_old * dc;
            b_gates(ib, LSTM_F, ih) = x_m_square(hf) * dhf;

            float dhi = hc * dc;
            b_gates(ib, LSTM_I, ih) = x_m_square(hi) * dhi;

            float dhc = hi * dc;
            b_gates(ib, LSTM_C, ih) = one_m_square(hc) * dhc;

            if (prb.is_lstm_peephole()) {
                dc_tm1 += b_gates(ib, LSTM_F, ih) * weights_peephole(1, ih);
                dc_tm1 += b_gates(ib, LSTM_I, ih) * weights_peephole(0, ih);
            }

            diff_src_iter_c(ib, ih) = dc_tm1;
        }
}

void lstm_bwd_pregemm(const prb_t &prb, const float *src_iter_c_,
        const float *dst_iter_c_, const float *weights_peephole_,
        const float *diff_hidden_state_, const float *diff_dst_iter_c_,
        const float *gates_, float *diff_src_iter_c_, float *b_gates_) {
    if (prb.skip_nonlinear)
        lstm_bwd_pregemm_template(
                [](float scale, float val) { return scale * val; }, prb,
                src_iter_c_, dst_iter_c_, weights_peephole_, diff_hidden_state_,
                diff_dst_iter_c_, gates_, diff_src_iter_c_, b_gates_);

    else
        lstm_bwd_pregemm_template(
                [](float scale, float val) { return tanhf(val); }, prb,
                src_iter_c_, dst_iter_c_, weights_peephole_, diff_hidden_state_,
                diff_dst_iter_c_, gates_, diff_src_iter_c_, b_gates_);
}

void lstm_bwd_weights_peephole(const prb_t &prb, const float *src_iter_c_,
        const float *dst_iter_c_, const float *b_gates_,
        float *diff_weights_peephole_) {
    AOC<const float> src_iter_c(src_iter_c_, prb.mb, prb.wc);
    AOC<const float> dst_iter_c(dst_iter_c_, prb.mb, prb.wc);
    AOC<const float> b_gates(b_gates_, prb.mb, prb.n_gates(), prb.dhc);
    AOC<float> diff_weights_peephole(diff_weights_peephole_, 3, prb.dhc);

    for_(int64_t ib = 0; ib < prb.mb; ++ib)
    for (int64_t ih = 0; ih < prb.dhc; ++ih)
        diff_weights_peephole(2, ih)
                += b_gates(ib, LSTM_O, ih) * dst_iter_c(ib, ih);

    for_(int64_t ib = 0; ib < prb.mb; ++ib)
    for_(int64_t j = 0; j < 2; ++j)
    for (int64_t ih = 0; ih < prb.dhc; ++ih)
        diff_weights_peephole(j, ih) += b_gates(ib, j, ih) * src_iter_c(ib, ih);
}

void lstm_bwd(const prb_t &prb, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_src_iter_c_, float *diff_weights_layer_,
        float *diff_weights_iter_, float *diff_weights_peephole_,
        float *diff_weights_projection_, float *diff_bias_, float *b_gates_,
        const float *src_layer_, const float *src_iter_,
        const float *src_iter_c_, const float *weights_layer_,
        const float *weights_iter_, const float *weights_peephole_,
        const float *weights_projection_, const float *bias_,
        const float *dst_layer_, const float *dst_iter_c_, const float *gates_,
        const float *ht_, const float *diff_dst_layer_,
        const float *diff_dst_iter_, const float *diff_dst_iter_c_,
        float *cell_scratchpad_) {
    float *diff_hidden_state_ = cell_scratchpad_;

    AOC<float> diff_hidden_state(diff_hidden_state_, prb.mb, prb.dhc);
    AOC<const float> diff_dst_layer(diff_dst_layer_, prb.mb, prb.wc);
    AOC<const float> diff_dst_iter(diff_dst_iter_, prb.mb, prb.wc);

    if (prb.is_lstm_projection()) {
        float *diff_dst
                = (float *)zmalloc(prb.mb * prb.dic * sizeof(float), 64);
        DNN_SAFE_V(diff_dst == nullptr ? dnnl_out_of_memory : dnnl_success);

        // The loop below relies on this property
        assert(prb.dic == prb.dlc(CELL));
        for_(int64_t ib = 0; ib < prb.mb; ib++)
        for (int64_t ih = 0; ih < prb.dic; ih++)
            diff_dst[ib * prb.dic + ih]
                    = diff_dst_layer(ib, ih) + diff_dst_iter(ib, ih);

        gemm("C", "T", "N", prb.dhc, prb.dic, prb.mb, 1.0, ht_, prb.wc,
                diff_dst, prb.dic, 1.0, diff_weights_projection_, prb.dic);
        gemm("C", "N", "T", prb.mb, prb.dhc, prb.dic, 1.0, diff_dst, prb.dic,
                weights_projection_, prb.dic, 0.0, diff_hidden_state_, prb.dhc);
        zfree(diff_dst);
    } else {
        for_(int64_t ib = 0; ib < prb.mb; ib++)
        for (int64_t ih = 0; ih < prb.dhc; ih++)
            diff_hidden_state(ib, ih)
                    = diff_dst_layer(ib, ih) + diff_dst_iter(ib, ih);
    }

    lstm_bwd_pregemm(prb, src_iter_c_, dst_iter_c_, weights_peephole_,
            diff_hidden_state_, diff_dst_iter_c_, gates_, diff_src_iter_c_,
            b_gates_);

    gemm("C", "T", "N", prb.sic, prb.n_gates() * prb.dhc, prb.mb, 1.0,
            src_iter_, prb.wc, b_gates_, prb.n_gates() * prb.dhc, 1.0,
            diff_weights_iter_, prb.n_gates() * prb.dhc);
    gemm("C", "T", "N", prb.slc, prb.n_gates() * prb.dhc, prb.mb, 1.0,
            src_layer_, prb.wc, b_gates_, prb.n_gates() * prb.dhc, 1.0,
            diff_weights_layer_, prb.n_gates() * prb.dhc);

    gemm("C", "N", "T", prb.mb, prb.sic, prb.n_gates() * prb.dhc, 1.0, b_gates_,
            prb.n_gates() * prb.dhc, weights_iter_, prb.n_gates() * prb.dhc,
            0.0, diff_src_iter_, prb.wc);
    gemm("C", "N", "T", prb.mb, prb.slc, prb.n_gates() * prb.dhc, 1.0, b_gates_,
            prb.n_gates() * prb.dhc, weights_layer_, prb.n_gates() * prb.dhc,
            0.0, diff_src_layer_, prb.wc);

    if (prb.is_lstm_peephole())
        lstm_bwd_weights_peephole(prb, src_iter_c_, dst_iter_c_, b_gates_,
                diff_weights_peephole_);

    gates_reduction(prb, b_gates_, diff_bias_);
}

} // namespace rnn
