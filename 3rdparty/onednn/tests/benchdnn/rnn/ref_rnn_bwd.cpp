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

#include <cmath>

#include "tests/test_thread.hpp"

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"

#include "rnn/cells.hpp"

namespace rnn {

void prepare_ws_bwd(const prb_t &prb, std::vector<float> &ws_bwd_buffer,
        AOC<float> &ws_diff_src_layer, AOC<float> &ws_diff_src_iter,
        AOC<float> &ws_diff_src_iter_c) {
    bool is_lstm = prb.alg == VANILLA_LSTM;

    ws_diff_src_layer = AOC<float>(nullptr, prb.n_layer + 2, prb.n_dir(),
            prb.n_iter + 2, prb.mb, prb.wc);
    ws_diff_src_iter = AOC<float>(nullptr, prb.n_layer + 2, prb.n_dir(),
            prb.n_iter + 2, prb.mb, prb.wc);
    ws_diff_src_iter_c = AOC<float>(nullptr, prb.n_layer + 2, prb.n_dir(),
            prb.n_iter + 2, prb.mb, prb.wc);

    int64_t size = ws_diff_src_layer.nelems() + ws_diff_src_iter.nelems()
            + is_lstm * ws_diff_src_iter_c.nelems();
    ws_bwd_buffer.resize(size, 0);

    ws_diff_src_layer.set_base_ptr(ws_bwd_buffer.data());
    ws_diff_src_iter.set_base_ptr(
            ws_bwd_buffer.data() + ws_diff_src_layer.nelems());
    ws_diff_src_iter_c.set_base_ptr(ws_bwd_buffer.data()
            + ws_diff_src_layer.nelems() + ws_diff_src_iter.nelems());
}

/******************************************************************************/
/******************************* Copy Routines ********************************/
/******************************************************************************/

void copy_init_bwd(const prb_t &prb, const AOC<float> &ws_diff_src_layer,
        const AOC<float> &ws_diff_src_iter,
        const AOC<float> &ws_diff_src_iter_c, const float *diff_dst_layer_,
        const float *diff_dst_iter_, const float *diff_dst_iter_c_,
        rnn_iter_direction_t iter_dir, rnn_layer_direction_t lay_dir,
        int64_t dir_val) {
    AOC<const float> diff_dst_layer(
            diff_dst_layer_, prb.n_iter, prb.mb * prb.dlc(PRIMITIVE));
    AOC<const float> diff_dst_iter(
            diff_dst_iter_, prb.n_layer, prb.n_dir(), prb.mb * prb.dic);
    AOC<const float> diff_dst_iter_c(
            diff_dst_iter_c_, prb.n_layer, prb.n_dir(), prb.mb * prb.dhc);

    const bool is_concat = prb.direction == dnnl_bidirectional_concat;
    int64_t lay_dest = (lay_dir == bottom2top) ? 0 : prb.n_layer + 1;
    int64_t it_dest = (iter_dir == left2right) ? 0 : prb.n_iter + 1;

    for (int64_t it = 0; it < prb.n_iter; it++)
        copy(prb.mb, prb.dlc(CELL), prb.dlc(PRIMITIVE), prb.wc,
                &diff_dst_layer(it, dir_val * is_concat * prb.dlc(CELL)),
                &ws_diff_src_layer(lay_dest, dir_val, it + 1, 0, 0));

    for (int64_t lay = 0; lay < prb.n_layer; lay++) {
        copy(prb.mb, prb.dic, prb.dic, prb.wc, &diff_dst_iter(lay, dir_val, 0),
                &ws_diff_src_iter(lay + 1, dir_val, it_dest, 0, 0));
        if (prb.alg == VANILLA_LSTM) {
            copy(prb.mb, prb.dhc, prb.dhc, prb.wc,
                    &diff_dst_iter_c(lay, dir_val, 0),
                    &ws_diff_src_iter_c(lay + 1, dir_val, it_dest, 0, 0));
        }
    }
}

void copy_res_bwd(const prb_t &prb, float *diff_src_layer_,
        float *diff_src_iter_, float *diff_src_iter_c_,
        const AOC<const float> &ws_diff_src_layer,
        const AOC<const float> &ws_diff_src_iter,
        const AOC<const float> &ws_diff_src_iter_c,
        rnn_iter_direction_t iter_dir, rnn_layer_direction_t lay_dir,
        int64_t dir_val, rnn_action_t action) {
    AOC<float> diff_src_iter(
            diff_src_iter_, prb.n_layer, prb.n_dir(), prb.mb, prb.sic);
    AOC<float> diff_src_iter_c(
            diff_src_iter_c_, prb.n_layer, prb.n_dir(), prb.mb, prb.dhc);
    AOC<float> diff_src_layer(diff_src_layer_, prb.n_iter, prb.mb, prb.slc);

    for (int64_t it = 0; it < prb.n_iter; it++) {
        for (int64_t nb = 0; nb < prb.mb; nb++) {
            auto from = &ws_diff_src_layer(1, dir_val, it + 1, nb, 0);
            auto to = &diff_src_layer(it, nb, 0);

            copy(1, prb.slc, prb.wc, prb.slc, from, to, action);
        }
    }

    int64_t it_source = (iter_dir == left2right) ? prb.n_iter : 1;

    for (int64_t lay = 0; lay < prb.n_layer; lay++) {
        if (prb.alg == VANILLA_LSTM) {
            copy(prb.mb, prb.dhc, prb.wc, prb.dhc,
                    &ws_diff_src_iter_c(lay + 1, dir_val, it_source, 0, 0),
                    &diff_src_iter_c(lay, dir_val, 0, 0));
        }
        copy(prb.mb, prb.sic, prb.wc, prb.sic,
                &ws_diff_src_iter(lay + 1, dir_val, it_source, 0, 0),
                &diff_src_iter(lay, dir_val, 0, 0));
    }
}

/******************************************************************************/
/*************************** Computation Routines *****************************/
/******************************************************************************/
void gates_reduction(
        const prb_t &prb, const float *b_gates_, float *diff_bias_) {
    AOC<const float> b_gates(b_gates_, prb.mb, prb.n_gates(), prb.dhc);
    for (int64_t i = 0; i < prb.mb; i++)
        for (int64_t j = 0; j < prb.n_gates(); j++)
            for (int64_t k = 0; k < prb.dhc; k++)
                diff_bias_[j * prb.dhc + k] += b_gates(i, j, k);
}

void rnn_cell_bwd(const prb_t &prb, float *diff_src_layer, float *diff_src_iter,
        float *diff_src_iter_c, float *diff_weights_layer,
        float *diff_weights_iter, float *diff_weights_peephole,
        float *diff_weights_projection, float *diff_bias, float *b_gates,
        const float *src_layer, const float *src_iter, const float *src_iter_c,
        const float *weights_layer, const float *weights_iter,
        const float *weights_peephole, const float *weights_projection,
        const float *bias, const float *dst_layer, const float *dst_iter_c,
        const float *gates, const float *ht, const float *diff_dst_layer,
        const float *diff_dst_iter, const float *diff_dst_iter_c,
        float *cell_scratchpad_) {

    switch (prb.alg) {
        case VANILLA_LSTM:
            lstm_bwd(prb, diff_src_layer, diff_src_iter, diff_src_iter_c,
                    diff_weights_layer, diff_weights_iter,
                    diff_weights_peephole, diff_weights_projection, diff_bias,
                    b_gates, src_layer, src_iter, src_iter_c, weights_layer,
                    weights_iter, weights_peephole, weights_projection, bias,
                    dst_layer, dst_iter_c, gates, ht, diff_dst_layer,
                    diff_dst_iter, diff_dst_iter_c, cell_scratchpad_);
            break;
        case VANILLA_RNN:
            rnn_bwd(prb, diff_src_layer, diff_src_iter, diff_weights_layer,
                    diff_weights_iter, diff_bias, b_gates, src_layer, src_iter,
                    weights_layer, weights_iter, bias, gates, diff_dst_layer,
                    diff_dst_iter);
            break;
        case VANILLA_GRU:
            gru_bwd(prb, diff_src_layer, diff_src_iter, diff_weights_layer,
                    diff_weights_iter, diff_bias, b_gates, src_layer, src_iter,
                    weights_layer, weights_iter, bias, gates, diff_dst_layer,
                    diff_dst_iter, cell_scratchpad_);
            break;
        case LBR_GRU:
            lbr_gru_bwd(prb, diff_src_layer, diff_src_iter, diff_weights_layer,
                    diff_weights_iter, diff_bias, b_gates, src_layer, src_iter,
                    weights_layer, weights_iter, bias, gates, diff_dst_layer,
                    diff_dst_iter, cell_scratchpad_);
        default: break;
    }
}

void rnn_linear_bwd(const prb_t &prb, const float *diff_dst_layer_,
        const float *diff_dst_iter_, const float *diff_dst_iter_c_,
        const float *weights_layer_, const float *weights_iter_,
        const float *weights_peephole_, const float *weights_projection_,
        const float *bias_, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_src_iter_c_, float *diff_weights_layer_,
        float *diff_weights_iter_, float *diff_weights_peephole_,
        float *diff_weights_projection_, float *diff_bias_,
        const AOC<const float> &ws_src_layer,
        const AOC<const float> &ws_src_iter,
        const AOC<const float> &ws_src_iter_c, const AOC<const float> &ws_gates,
        const AOC<const float> &ws_ht) {
    bool is_lbr = prb.alg == LBR_GRU;

    AOC<const float> weights_layer(weights_layer_, prb.n_layer, prb.n_dir(),
            prb.n_gates() * prb.dhc, prb.slc);
    AOC<const float> weights_iter(weights_iter_, prb.n_layer, prb.n_dir(),
            prb.n_gates() * prb.dhc, prb.sic);

    AOC<float> diff_weights_layer(diff_weights_layer_, prb.n_layer, prb.n_dir(),
            prb.n_gates() * prb.dhc, prb.slc);
    AOC<float> diff_weights_iter(diff_weights_iter_, prb.n_layer, prb.n_dir(),
            prb.n_gates() * prb.dhc, prb.sic);

    AOC<const float> weights_peephole(
            weights_peephole_, prb.n_layer, prb.n_dir(), 3 * prb.dhc);
    AOC<float> diff_weights_peephole(
            diff_weights_peephole_, prb.n_layer, prb.n_dir(), 3 * prb.dhc);

    AOC<const float> weights_projection(
            weights_projection_, prb.n_layer, prb.n_dir(), prb.dhc * prb.dic);
    AOC<float> diff_weights_projection(diff_weights_projection_, prb.n_layer,
            prb.n_dir(), prb.dhc * prb.dic);

    AOC<const float> bias(
            bias_, prb.n_layer, prb.n_dir(), prb.n_gates() + is_lbr, prb.dhc);
    AOC<float> diff_bias(diff_bias_, prb.n_layer, prb.n_dir(),
            prb.n_gates() + is_lbr, prb.dhc);

    std::vector<float> ws_bwd_buffer;
    AOC<float> ws_diff_src_layer, ws_diff_src_iter, ws_diff_src_iter_c;
    prepare_ws_bwd(prb, ws_bwd_buffer, ws_diff_src_layer, ws_diff_src_iter,
            ws_diff_src_iter_c);

    int64_t b_gates_size = prb.mb * prb.n_gates() * prb.dhc;
    auto *b_gates = new float[b_gates_size];
    for (int i = 0; i < b_gates_size; i++) {
        b_gates[i] = NAN;
    }

    int64_t cell_scratchpad_size = 0;
    switch (prb.alg) {
        case VANILLA_LSTM: cell_scratchpad_size = prb.mb * prb.dhc; break;
        case LBR_GRU:
            cell_scratchpad_size = prb.mb * (prb.n_gates() + 1) * prb.dhc;
            break;
        case VANILLA_GRU: cell_scratchpad_size = 2 * prb.mb * prb.dhc; break;
        default: cell_scratchpad_size = 0;
    }
    float *cell_scratchpad_ = new float[cell_scratchpad_size];
    for (int i = 0; i < cell_scratchpad_size; i++) {
        cell_scratchpad_[i] = NAN;
    }

    auto process_direction = [&](rnn_iter_direction_t iter_dir,
                                     rnn_layer_direction_t lay_dir,
                                     int64_t dir_val, rnn_action_t action) {
        // we first need to copy the initial diff_dst_layer and
        // diff_dst_iter{,_c} into ws to simplify the logic of the code
        copy_init_bwd(prb, ws_diff_src_layer, ws_diff_src_iter,
                ws_diff_src_iter_c, diff_dst_layer_, diff_dst_iter_,
                diff_dst_iter_c_, iter_dir, lay_dir, dir_val);

        // We run the grid of computation
        for (int64_t j = prb.n_layer - 1; j >= 0; j--) {
            for (int64_t i = 0; i < prb.n_iter; i++) {
                int64_t iter
                        = (iter_dir == left2right) ? i + 1 : prb.n_iter - i;
                int64_t prev_iter
                        = (iter_dir == left2right) ? iter - 1 : iter + 1;
                int64_t lay = j + 1;
                int64_t prev_lay = lay + 1;

                int64_t ws_iter = iter;
                int64_t ws_prev_iter
                        = (iter_dir == left2right) ? iter + 1 : iter - 1;

                rnn_cell_bwd(prb, &ws_diff_src_layer(lay, dir_val, iter, 0, 0),
                        &ws_diff_src_iter(lay, dir_val, iter, 0, 0),
                        &ws_diff_src_iter_c(lay, dir_val, iter, 0, 0),
                        &diff_weights_layer(lay - 1, dir_val, 0, 0),
                        &diff_weights_iter(lay - 1, dir_val, 0, 0),
                        &diff_weights_peephole(lay - 1, dir_val, 0),
                        &diff_weights_projection(lay - 1, dir_val, 0),
                        &diff_bias(lay - 1, dir_val, 0, 0), b_gates,
                        &ws_src_layer(lay - 1, dir_val, ws_iter, 0, 0),
                        &ws_src_iter(lay, dir_val, ws_prev_iter, 0, 0),
                        &ws_src_iter_c(lay, dir_val, ws_prev_iter, 0, 0),
                        &weights_layer(lay - 1, dir_val, 0, 0),
                        &weights_iter(lay - 1, dir_val, 0, 0),
                        &weights_peephole(lay - 1, dir_val, 0),
                        &weights_projection(lay - 1, dir_val, 0),
                        &bias(lay - 1, dir_val, 0, 0),
                        &ws_src_layer(lay, dir_val, ws_iter, 0, 0),
                        &ws_src_iter_c(lay, dir_val, ws_iter, 0, 0),
                        &ws_gates(lay - 1, dir_val, ws_iter - 1, 0, 0, 0),
                        &ws_ht(lay - 1, dir_val, ws_iter - 1, 0, 0),
                        &ws_diff_src_layer(prev_lay, dir_val, iter, 0, 0),
                        &ws_diff_src_iter(lay, dir_val, prev_iter, 0, 0),
                        &ws_diff_src_iter_c(lay, dir_val, prev_iter, 0, 0),
                        cell_scratchpad_);
            }
        }

        // Finally we copy the results to the result buffers
        copy_res_bwd(prb, diff_src_layer_, diff_src_iter_, diff_src_iter_c_,
                ws_diff_src_layer, ws_diff_src_iter, ws_diff_src_iter_c,
                iter_dir, lay_dir, dir_val, action);
    };

    switch (prb.direction) {
        case dnnl_unidirectional_left2right:
            process_direction(right2left, top2bottom, 0, action_copy);
            break;
        case dnnl_unidirectional_right2left:
            process_direction(left2right, top2bottom, 0, action_copy);
            break;
        case dnnl_bidirectional_sum:
            process_direction(right2left, top2bottom, 0, action_copy);
            process_direction(left2right, top2bottom, 1, action_sum);
            break;
        case dnnl_bidirectional_concat:
            process_direction(right2left, top2bottom, 0, action_copy);
            process_direction(left2right, top2bottom, 1, action_sum);
            break;
        default: assert(!"unknown direction"); break;
    }

    delete[] b_gates;
    delete[] cell_scratchpad_;
}

void compute_ref_bwd(const prb_t &prb, dnn_mem_t &src_layer_m,
        dnn_mem_t &src_iter_m, dnn_mem_t &src_iter_c_m,
        dnn_mem_t &diff_dst_layer_m, dnn_mem_t &diff_dst_iter_m,
        dnn_mem_t &diff_dst_iter_c_m, dnn_mem_t &weights_layer_m,
        dnn_mem_t &weights_iter_m, dnn_mem_t &weights_peephole_m,
        dnn_mem_t &weights_projection_m, dnn_mem_t &bias_m,
        dnn_mem_t &dst_layer_m, dnn_mem_t &dst_iter_m, dnn_mem_t &dst_iter_c_m,
        dnn_mem_t &diff_src_layer_m, dnn_mem_t &diff_src_iter_m,
        dnn_mem_t &diff_src_iter_c_m, dnn_mem_t &diff_weights_layer_m,
        dnn_mem_t &diff_weights_iter_m, dnn_mem_t &diff_weights_peephole_m,
        dnn_mem_t &diff_weights_projection_m, dnn_mem_t &diff_bias_m) {
    std::vector<float> ws_fwd_buffer;
    AOC<float> ws_src_layer, ws_src_iter, ws_src_iter_c, ws_gates, ws_ht;
    prepare_ws_fwd(prb, ws_fwd_buffer, ws_src_layer, ws_src_iter, ws_src_iter_c,
            ws_gates, ws_ht);

    rnn_linear_fwd(prb, (float *)src_layer_m, (float *)src_iter_m,
            (float *)src_iter_c_m, (float *)weights_layer_m,
            (float *)weights_iter_m, (float *)weights_peephole_m,
            (float *)weights_projection_m, (float *)bias_m,
            (float *)dst_layer_m, (float *)dst_iter_m, (float *)dst_iter_c_m,
            ws_src_layer, ws_src_iter, ws_src_iter_c, ws_gates, ws_ht);

    rnn_linear_bwd(prb, (float *)diff_dst_layer_m, (float *)diff_dst_iter_m,
            (float *)diff_dst_iter_c_m, (float *)weights_layer_m,
            (float *)weights_iter_m, (float *)weights_peephole_m,
            (float *)weights_projection_m, (float *)bias_m,
            (float *)diff_src_layer_m, (float *)diff_src_iter_m,
            (float *)diff_src_iter_c_m, (float *)diff_weights_layer_m,
            (float *)diff_weights_iter_m, (float *)diff_weights_peephole_m,
            (float *)diff_weights_projection_m, (float *)diff_bias_m,
            ws_src_layer, ws_src_iter, ws_src_iter_c, ws_gates, ws_ht);
}

} // namespace rnn
