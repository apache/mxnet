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

float activation(const prb_t &prb, float x, bool is_fwd = true) {
    float result = 0.0f;
    if (prb.skip_nonlinear)
        result = prb.linear_scales[0] * x;
    else
        switch (prb.activation) {
            case RELU:
                result = is_fwd ? relu(x, prb.alpha) : drelu(x, prb.alpha);
                break;
            case LOGISTIC: result = is_fwd ? logistic(x) : x_m_square(x); break;
            case TANH: result = is_fwd ? tanhf(x) : one_m_square(x); break;
            default: assert(!"unknown activation");
        }
    return result;
}

void rnn_fwd_postgemm(const prb_t &prb, const float *bias_, float *gates_,
        float *dst_layer_) {
    AOC<float> dst_layer(dst_layer_, prb.mb, prb.n_gates(), prb.wc);
    AOC<const float> bias(bias_, prb.n_gates(), prb.dhc);
    AOC<float> gates(gates_, prb.mb, prb.n_gates(), prb.dhc);

    for (int64_t i = 0; i < prb.mb; i++)
        for (int64_t j = 0; j < prb.n_gates(); j++)
            for (int64_t k = 0; k < prb.dhc; k++) {
                const auto tmp = activation(prb, gates(i, j, k) + bias(j, k));
                gates(i, j, k) = tmp;
                dst_layer(i, j, k) = tmp;
            }
}

void rnn_fwd(const prb_t &prb, float *dst_layer_, float *gates_,
        const float *weights_layer_, const float *weights_iter_,
        const float *bias_, const float *src_layer_, const float *src_iter_) {
    gemm("C", "N", "N", prb.mb, prb.n_gates() * prb.dhc, prb.slc, 1.0,
            src_layer_, prb.wc, weights_layer_, prb.n_gates() * prb.dhc, 0.0,
            gates_, prb.n_gates() * prb.dhc);
    gemm("C", "N", "N", prb.mb, prb.n_gates() * prb.dhc, prb.sic, 1.0,
            src_iter_, prb.wc, weights_iter_, prb.n_gates() * prb.dhc, 1.0,
            gates_, prb.n_gates() * prb.dhc);
    rnn_fwd_postgemm(prb, bias_, gates_, dst_layer_);
}

void rnn_bwd_pregemm(const prb_t &prb, const float *diff_dst_layer_,
        const float *diff_dst_iter_, const float *gates_, float *b_gates_) {
    AOC<const float> diff_dst_layer(diff_dst_layer_, prb.mb, prb.wc);
    AOC<const float> diff_dst_iter(diff_dst_iter_, prb.mb, prb.wc);
    AOC<const float> gates(gates_, prb.mb, prb.n_gates(), prb.dhc);
    AOC<float> b_gates(b_gates_, prb.mb, prb.n_gates(), prb.dhc);

    for (int64_t b = 0; b < prb.mb; ++b)
        for (int64_t h = 0; h < prb.dhc; ++h) {
            const float g = gates(b, 0, h);
            const float dd = diff_dst_layer(b, h) + diff_dst_iter(b, h);
            b_gates(b, 0, h) = activation(prb, g, false) * dd;
        }
}

void rnn_bwd(const prb_t &prb, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_, const float *bias_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_) {
    AOC<float> b_gates(b_gates_, prb.mb, prb.n_gates(), prb.dhc);

    rnn_bwd_pregemm(prb, diff_dst_layer_, diff_dst_iter_, gates_, b_gates_);

    gemm("C", "T", "N", prb.sic, prb.n_gates() * prb.dhc, prb.mb, 1.0,
            src_iter_, prb.wc, b_gates_, prb.n_gates() * prb.dhc, 1.0,
            diff_weights_iter_, prb.n_gates() * prb.dhc);
    gemm("C", "T", "N", prb.slc, prb.n_gates() * prb.dhc, prb.mb, 1.0,
            src_layer_, prb.wc, b_gates_, prb.n_gates() * prb.dhc, 1.0,
            diff_weights_layer_, prb.n_gates() * prb.dhc);
    for (int64_t b = 0; b < prb.mb; ++b)
        copy(prb.n_gates(), prb.dhc, prb.dhc, prb.dhc, &b_gates(b, 0, 0),
                diff_bias_, action_sum);

    gemm("C", "N", "T", prb.mb, prb.slc, prb.n_gates() * prb.dhc, 1.0, b_gates_,
            prb.n_gates() * prb.dhc, weights_layer_, prb.n_gates() * prb.dhc,
            0.0, diff_src_layer_, prb.wc);
    gemm("C", "N", "T", prb.mb, prb.sic, prb.n_gates() * prb.dhc, 1.0, b_gates_,
            prb.n_gates() * prb.dhc, weights_iter_, prb.n_gates() * prb.dhc,
            0.0, diff_src_iter_, prb.wc);
}

} // namespace rnn
