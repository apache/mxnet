/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

/*
 * Cell execution LSTM
 */

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"

#include "cpu/rnn/postgemm_dispatcher.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;
#define AOC array_offset_calculator

template <typename T1, typename T2, typename T3, typename src_data_t,
        typename scratch_data_t>
void gru_lbr_fwd_postgemm_template(T1 func1, T2 func2, T3 to_src,
        const float *scales, const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *dst_layer_,
        src_data_t *dst_iter_, const src_data_t *src_iter_, float *bias_,
        src_data_t *ws_grid_, scratch_data_t *scratch_cell_) {

    auto src_iter_ld = rnn.src_iter_ld(cell_position);
    auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
    auto dst_iter_ld = rnn.dst_iter_ld(cell_position);

    ws_states_layer_aoc<src_data_t> dst_layer(rnn, dst_layer_, dst_layer_ld);
    ws_states_iter_aoc<src_data_t> dst_iter(rnn, dst_iter_, dst_iter_ld);
    ws_states_iter_aoc<const src_data_t> src_iter(rnn, src_iter_, src_iter_ld);
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    scratch_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_gates_aoc<scratch_data_t> scratch_cell(rnn, scratch_cell_);
    AOC<src_data_t, 2> ws_Wh_b(ws_grid_, rnn.mb, rnn.dhc);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dhc; j++) {
            float Wh_b = scratch_cell(i, 2, j) + bias(3, j);
            auto G0 = func1(scales, // default func1 is sigmoid
                    scratch_gates(i, 0, j) + scratch_cell(i, 0, j)
                            + bias(0, j));
            auto G1 = func1(scales + 1, // default func1 is sigmoid
                    scratch_gates(i, 1, j) + scratch_cell(i, 1, j)
                            + bias(1, j));
            auto G2 = func2(scales + 2, // default func2 is tanh
                    scratch_gates(i, 2, j) + G1 * Wh_b + bias(2, j));
            auto tmp = to_src(src_iter(i, j) * G0 + (1.0f - G0) * G2);
            if (dst_layer_ != nullptr) dst_layer(i, j) = tmp;
            if (dst_iter_ != nullptr) dst_iter(i, j) = tmp;
            if (rnn.is_training) {
                ws_gates(i, 0, j) = to_src(G0);
                ws_gates(i, 1, j) = to_src(G1);
                ws_gates(i, 2, j) = to_src(G2);
                ws_Wh_b(i, j) = to_src(Wh_b);
            }
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::gru_lbr_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;

    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };
    auto to_src = [](float a) { return a; };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        gru_lbr_fwd_postgemm_template(logistic_f, tanh_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_, ws_grid_, scratch_cell_);
    else
        gru_lbr_fwd_postgemm_template(linear_f, linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_, ws_grid_, scratch_cell_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_bf16_t::gru_lbr_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;

    auto linear_f = [](const float *scale, float a) { return *scale * a; };
    auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };
    auto to_src = [](float a) { return bfloat16_t(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        gru_lbr_fwd_postgemm_template(logistic_f, tanh_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_, ws_grid_, scratch_cell_);
    else
        gru_lbr_fwd_postgemm_template(linear_f, linear_f, to_src, scales, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_layer_, dst_iter_,
                src_iter_, bias_, ws_grid_, scratch_cell_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::gru_lbr_postgemm) {
    assert(!"GRU LBR int8 is not supported");
}

template <typename T1, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void gru_lbr_bwd_postgemm_template(T1 to_src, const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, const src_data_t *src_iter_,
        acc_data_t *diff_src_iter_, acc_data_t *diff_dst_iter_,
        acc_data_t *diff_dst_layer_, scratch_data_t *scratch_cell_,
        src_data_t *ws_grid_) {
    auto src_iter_ld = rnn.src_iter_ld(cell_position);
    ws_states_iter_aoc<const src_data_t> src_iter(rnn, src_iter_, src_iter_ld);
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    ws_diff_states_iter_aoc<acc_data_t> diff_src_iter(rnn, diff_src_iter_);
    ws_diff_states_iter_aoc<acc_data_t> diff_dst_iter(rnn, diff_dst_iter_);
    ws_diff_states_layer_aoc<acc_data_t> diff_dst_layer(rnn, diff_dst_layer_);
    ws_gates_aoc<scratch_data_t> scratch_gates_r(rnn, scratch_cell_);
    AOC<src_data_t, 2> ws_Wh_b(ws_grid_, rnn.mb, rnn.dhc);

    // 1. calculate dG1 dG2 dG3
    // dG0 = (dht - G2) * dht * (1 - G0) * G0
    // dG1 = (W*h + b) * dG2 * (1 - G1) * G1
    // dG2 = (1 - G0) * dht * (1 - G2*G2)
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dhc; j++) {
            float h = src_iter(i, j);
            float dHt = diff_dst_iter(i, j) + diff_dst_layer(i, j);
            float dG0 = (h - ws_gates(i, 2, j)) * dHt
                    * x_m_square(ws_gates(i, 0, j));
            float dG2 = (1.0f - ws_gates(i, 0, j))
                    * one_m_square(ws_gates(i, 2, j)) * dHt;
            float dG1 = ws_Wh_b(i, j) * dG2 * x_m_square(ws_gates(i, 1, j));

            diff_src_iter(i, j) = dHt * ws_gates(i, 0, j);
            scratch_gates(i, 2, j) = to_src(dG2);
            scratch_gates_r(i, 2, j) = to_src(dG2 * ws_gates(i, 1, j));
            scratch_gates(i, 0, j) = scratch_gates_r(i, 0, j) = to_src(dG0);
            scratch_gates(i, 1, j) = scratch_gates_r(i, 1, j) = to_src(dG1);
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::gru_lbr_postgemm) {
    auto to_src = [&](float a) { return a; };
    gru_lbr_bwd_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, src_iter_, diff_src_iter_, diff_dst_iter_,
            diff_dst_layer_, scratch_cell_, ws_grid_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::gru_lbr_postgemm) {
    auto to_src = [&](float a) { return bfloat16_t(a); };
    gru_lbr_bwd_postgemm_template(to_src, rnn, cell_position, ws_gates_,
            scratch_gates_, src_iter_, diff_src_iter_, diff_dst_iter_,
            diff_dst_layer_, scratch_cell_, ws_grid_);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
