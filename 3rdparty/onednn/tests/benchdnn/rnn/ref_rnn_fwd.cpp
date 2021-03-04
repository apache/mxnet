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

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"

#include "rnn/cells.hpp"

namespace rnn {

void prepare_ws_fwd(const prb_t &prb, std::vector<float> &ws_fwd_buffer,
        AOC<float> &ws_src_layer, AOC<float> &ws_src_iter,
        AOC<float> &ws_src_iter_c, AOC<float> &ws_gates, AOC<float> &ws_ht) {
    bool is_lstm = prb.alg == VANILLA_LSTM;
    bool is_lstmp = prb.is_lstm_projection();

    ws_src_layer = AOC<float>(nullptr, prb.n_layer + 2, prb.n_dir(),
            prb.n_iter + 2, prb.mb, prb.wc);
    ws_src_iter = AOC<float>(nullptr, prb.n_layer + 2, prb.n_dir(),
            prb.n_iter + 2, prb.mb, prb.wc);
    ws_src_iter_c = AOC<float>(nullptr, prb.n_layer + 2, prb.n_dir(),
            prb.n_iter + 2, prb.mb, prb.wc);
    ws_gates = AOC<float>(nullptr, prb.n_layer, prb.n_dir(), prb.n_iter, prb.mb,
            prb.n_gates(), prb.dhc);
    ws_ht = AOC<float>(
            nullptr, prb.n_layer, prb.n_dir(), prb.n_iter, prb.mb, prb.wc);

    int64_t size = ws_src_layer.nelems() + is_lstm * ws_src_iter_c.nelems()
            + ws_gates.nelems() + is_lstmp * ws_ht.nelems();
    ws_fwd_buffer.resize(size);

    float *ptr = ws_fwd_buffer.data();
    ws_src_layer.set_base_ptr(ptr);
    ws_src_iter.set_base_ptr(ptr);

    ptr += ws_src_iter.nelems();
    ws_src_iter_c.set_base_ptr(ptr);

    ptr += is_lstm * ws_src_iter_c.nelems();
    ws_gates.set_base_ptr(ptr);

    ptr += is_lstmp * ws_gates.nelems();
    ws_ht.set_base_ptr(ptr);
}

/******************************************************************************/
/******************************* Copy Routines ********************************/
/******************************************************************************/
void prepare_bias(const prb_t &prb, float *bias_with_compensation_,
        const float *bias_, const float *weights_layer_,
        const float *weights_iter_) {
    AOC<const float> weights_layer(weights_layer_, prb.n_layer, prb.n_dir(),
            prb.slc, prb.n_gates(), prb.dhc);
    AOC<const float> weights_iter(weights_iter_, prb.n_layer, prb.n_dir(),
            prb.sic, prb.n_gates(), prb.dhc);

    AOC<const float> bias(
            bias_, prb.n_layer, prb.n_dir(), prb.n_gates(), prb.dhc);
    AOC<float> bias_with_compensation(bias_with_compensation_, prb.n_layer,
            prb.n_dir(), prb.n_gates(), prb.dhc);

    for (int layer = 0; layer < prb.n_layer; ++layer)
        for (int dir = 0; dir < prb.n_dir(); ++dir)
            for (int gate = 0; gate < prb.n_gates(); ++gate)
                for (int dhc = 0; dhc < prb.dhc; ++dhc) {
                    float weights_compensation = 0;
                    for (int sic = 0; sic < prb.sic; ++sic)
                        weights_compensation
                                += weights_iter(layer, dir, sic, gate, dhc);
                    for (int slc = 0; slc < prb.slc; ++slc)
                        weights_compensation
                                += weights_layer(layer, dir, slc, gate, dhc);

                    float scale = prb.data_scale
                            * prb.get_wei_scale(gate * prb.dhc + dhc);
                    bias_with_compensation(layer, dir, gate, dhc)
                            = bias(layer, dir, gate, dhc)
                            - weights_compensation * prb.data_shift / scale;
                }
}

void copy_init_fwd(const prb_t &prb, const AOC<float> &ws_src_layer,
        const AOC<float> &ws_src_iter, const AOC<float> &ws_src_iter_c,
        const float *src_layer_, const float *src_iter_,
        const float *src_iter_c_, rnn_iter_direction_t iter_dir,
        rnn_layer_direction_t lay_dir, int64_t dir_val) {
    AOC<const float> src_layer(src_layer_, prb.n_iter, prb.mb * prb.slc);
    AOC<const float> src_iter(
            src_iter_, prb.n_layer, prb.n_dir(), prb.mb * prb.sic);
    AOC<const float> src_iter_c(
            src_iter_c_, prb.n_layer, prb.n_dir(), prb.mb * prb.dhc);

    int64_t lay_dest = (lay_dir == bottom2top) ? 0 : prb.n_layer + 1;
    int64_t it_dest = (iter_dir == left2right) ? 0 : prb.n_iter + 1;

    // Copy src_layer
    for (int64_t it = 0; it < prb.n_iter; it++) {
        copy(prb.mb, prb.slc, prb.slc, prb.wc, &src_layer(it, 0),
                &ws_src_layer(lay_dest, dir_val, it + 1, 0, 0));
        if (prb.is_int8())
            data_q10n(prb.mb, prb.slc, prb.wc,
                    &ws_src_layer(lay_dest, dir_val, it + 1, 0, 0),
                    prb.data_scale, prb.data_shift);
    }

    // Copy src_iter (and src_iter_c)
    for (int64_t lay = 0; lay < prb.n_layer; lay++) {
        copy(prb.mb, prb.sic, prb.sic, prb.wc, &src_iter(lay, dir_val, 0),
                &ws_src_iter(lay + 1, dir_val, it_dest, 0, 0));
        if (prb.is_int8())
            data_q10n(prb.mb, prb.sic, prb.wc,
                    &ws_src_iter(lay + 1, dir_val, it_dest, 0, 0),
                    prb.data_scale, prb.data_shift);

        if (prb.alg == VANILLA_LSTM)
            copy(prb.mb, prb.dhc, prb.dhc, prb.wc, &src_iter_c(lay, dir_val, 0),
                    &ws_src_iter_c(lay + 1, dir_val, it_dest, 0, 0));
    }
}

void copy_res_fwd(const prb_t &prb, float *dst_layer_, float *dst_iter_,
        float *dst_iter_c_, const AOC<const float> &ws_src_layer,
        const AOC<const float> &ws_src_iter,
        const AOC<const float> &ws_src_iter_c, rnn_iter_direction_t iter_dir,
        rnn_layer_direction_t lay_dir, int64_t dir_val, rnn_action_t action) {
    AOC<float> dst_iter(dst_iter_, prb.n_layer, prb.n_dir(), prb.mb, prb.dic);
    AOC<float> dst_iter_c(
            dst_iter_c_, prb.n_layer, prb.n_dir(), prb.mb, prb.dhc);
    AOC<float> dst_layer(dst_layer_, prb.n_iter, prb.mb, prb.dlc(PRIMITIVE));

    // Copy dst_layer
    for (int64_t it = 0; it < prb.n_iter; it++) {
        for (int64_t nb = 0; nb < prb.mb; nb++) {
            auto from = &ws_src_layer(prb.n_layer, dir_val, it + 1, nb, 0);
            auto to = &dst_layer(
                    it, nb, action == action_concat ? prb.dlc(CELL) : 0);
            copy(1, prb.dlc(CELL), prb.wc, prb.dlc(PRIMITIVE), from, to, action,
                    prb.is_int8());

            if (prb.is_int8() && prb.cfg[DST_LAYER].dt != dnnl_u8) {
                float data_shift = prb.data_shift;
                bool do_deq10n = true;

                if (prb.direction == dnnl_bidirectional_sum) {
                    // In `bidir_sum` case, we need to dequantize data only
                    // after the final summation. Also, since we sum two shifted
                    // tensors, we need to enlarge the shift by 2x.
                    do_deq10n = action == action_sum;
                    data_shift *= 2;
                }

                if (do_deq10n)
                    data_deq10n(1, prb.dlc(CELL), prb.dlc(PRIMITIVE), to,
                            prb.data_scale, data_shift);
            }
        }
    }

    int64_t it_source = (iter_dir == left2right) ? prb.n_iter : 1;

    // Copy dst_iter (and dst_iter_c)
    for (int64_t lay = 0; lay < prb.n_layer; lay++) {
        if (prb.alg == VANILLA_LSTM) {
            copy(prb.mb, prb.dhc, prb.wc, prb.dhc,
                    &ws_src_iter_c(lay + 1, dir_val, it_source, 0, 0),
                    &dst_iter_c(lay, dir_val, 0, 0));
        }

        copy(prb.mb, prb.dic, prb.wc, prb.dic,
                &ws_src_iter(lay + 1, dir_val, it_source, 0, 0),
                &dst_iter(lay, dir_val, 0, 0));
        if (prb.is_int8() && prb.cfg[DST_ITER].dt != dnnl_u8)
            data_deq10n(prb.mb, prb.dic, prb.dic, &dst_iter(lay, dir_val, 0, 0),
                    prb.data_scale, prb.data_shift);
    }
}

/******************************************************************************/
/*************************** Computation Routines *****************************/
/******************************************************************************/

void rnn_cell_fwd(const prb_t &prb, float *dst_layer, float *dst_iter,
        float *dst_iter_c, float *gates, float *ht, const float *weights_layer,
        const float *weights_iter, const float *weights_peephole,
        const float *weights_projection, const float *bias,
        const float *src_layer, const float *src_iter, const float *src_iter_c,
        float *cell_scratchpad_) {
    if (prb.alg != VANILLA_LSTM) assert(dst_layer == dst_iter);

    switch (prb.alg) {
        case VANILLA_GRU:
            gru_fwd(prb, dst_layer, gates, weights_layer, weights_iter, bias,
                    src_layer, src_iter);
            break;
        case LBR_GRU:
            lbr_gru_fwd(prb, dst_layer, gates, weights_layer, weights_iter,
                    bias, src_layer, src_iter, cell_scratchpad_);
            break;
        case VANILLA_LSTM:
            lstm_fwd(prb, dst_layer, dst_iter, dst_iter_c, gates, ht,
                    weights_layer, weights_iter, weights_peephole,
                    weights_projection, bias, src_layer, src_iter, src_iter_c);
            break;
        case VANILLA_RNN:
            rnn_fwd(prb, dst_layer, gates, weights_layer, weights_iter, bias,
                    src_layer, src_iter);
            break;
        default: break;
    }
}

void rnn_linear_fwd(const prb_t &prb, const float *src_layer_,
        const float *src_iter_, const float *src_iter_c_,
        const float *weights_layer_, const float *weights_iter_,
        const float *weights_peephole_, const float *weights_projection_,
        const float *bias_, float *dst_layer_, float *dst_iter_,
        float *dst_iter_c_, const AOC<float> &ws_src_layer,
        const AOC<float> &ws_src_iter, const AOC<float> &ws_src_iter_c,
        const AOC<float> &ws_gates, const AOC<float> &ws_ht) {
    bool is_lbr = prb.alg == LBR_GRU;

    float *bias_with_compensation = nullptr;
    if (prb.is_int8()) {
        bias_with_compensation = new float[prb.n_layer * prb.n_dir()
                * (prb.n_gates() + is_lbr) * prb.dhc];
        prepare_bias(prb, bias_with_compensation, bias_, weights_layer_,
                weights_iter_);
        bias_ = bias_with_compensation;
    }

    AOC<const float> weights_peephole(
            weights_peephole_, prb.n_layer, prb.n_dir(), 3 * prb.dhc);
    AOC<const float> weights_projection(
            weights_projection_, prb.n_layer, prb.n_dir(), prb.dhc * prb.dic);
    AOC<const float> bias(bias_, prb.n_layer, prb.n_dir(),
            (prb.n_gates() + is_lbr) * prb.dhc);
    AOC<const float> weights_layer(weights_layer_, prb.n_layer, prb.n_dir(),
            prb.n_gates() * prb.dhc, prb.slc);
    AOC<const float> weights_iter(weights_iter_, prb.n_layer, prb.n_dir(),
            prb.n_gates() * prb.dhc, prb.sic);

    int64_t cell_scratchpad_size = is_lbr * prb.mb * prb.n_gates() * prb.dhc;
    float *cell_scratchpad_ = new float[cell_scratchpad_size];
    for (int i = 0; i < cell_scratchpad_size; i++) {
        cell_scratchpad_[i] = NAN;
    }

    auto process_direction = [&](rnn_iter_direction_t iter_dir,
                                     rnn_layer_direction_t lay_dir,
                                     int64_t dir_val, rnn_action_t action) {
        // we first need to copy the initial src_layer and src_iter{,_c} into
        // ws to simplify the logic of the code
        BENCHDNN_PRINT(80,
                "rnn_linear_fwd: call copy_init dir_val = " IFMT "\n", dir_val);
        copy_init_fwd(prb, ws_src_layer, ws_src_iter, ws_src_iter_c, src_layer_,
                src_iter_, src_iter_c_, iter_dir, lay_dir, dir_val);

        // We run the grid of computation
        for (int64_t il = 0; il < prb.n_layer; il++) {
            for (int64_t it = 0; it < prb.n_iter; it++) {
                BENCHDNN_PRINT(80,
                        "==== layer = " IFMT " iter = " IFMT " ===\n", il, it);
                int64_t iter
                        = (iter_dir == left2right) ? it + 1 : prb.n_iter - it;
                int64_t prev_iter
                        = (iter_dir == left2right) ? iter - 1 : iter + 1;
                int64_t lay = il + 1;
                rnn_cell_fwd(prb, &ws_src_layer(lay, dir_val, iter, 0, 0),
                        &ws_src_iter(lay, dir_val, iter, 0, 0),
                        &ws_src_iter_c(lay, dir_val, iter, 0, 0),
                        &ws_gates(lay - 1, dir_val, iter - 1, 0, 0, 0),
                        &ws_ht(lay - 1, dir_val, iter - 1, 0, 0),
                        &weights_layer(lay - 1, dir_val, 0, 0),
                        &weights_iter(lay - 1, dir_val, 0, 0),
                        &weights_peephole(lay - 1, dir_val, 0),
                        &weights_projection(lay - 1, dir_val, 0),
                        &bias(lay - 1, dir_val, 0),
                        &ws_src_layer(lay - 1, dir_val, iter, 0, 0),
                        &ws_src_iter(lay, dir_val, prev_iter, 0, 0),
                        &ws_src_iter_c(lay, dir_val, prev_iter, 0, 0),
                        cell_scratchpad_);
            }
        }

        // Finally we copy the results to the result buffers
        copy_res_fwd(prb, dst_layer_, dst_iter_, dst_iter_c_, ws_src_layer,
                ws_src_iter, ws_src_iter_c, iter_dir, lay_dir, dir_val, action);
    };

    switch (prb.direction) {
        case dnnl_unidirectional_left2right:
            process_direction(left2right, bottom2top, 0, action_copy);
            break;
        case dnnl_unidirectional_right2left:
            process_direction(right2left, bottom2top, 0, action_copy);
            break;
        case dnnl_bidirectional_sum:
            process_direction(left2right, bottom2top, 0, action_copy);
            process_direction(right2left, bottom2top, 1, action_sum);
            break;
        case dnnl_bidirectional_concat:
            process_direction(left2right, bottom2top, 0, action_copy);
            process_direction(right2left, bottom2top, 1, action_concat);
            break;
        default: assert(!"unknown direction"); break;
    }

    delete[] cell_scratchpad_;
    delete[] bias_with_compensation;
}

void compute_ref_fwd(const prb_t &prb, dnn_mem_t &src_layer_m,
        dnn_mem_t &src_iter_m, dnn_mem_t &src_iter_c_m,
        dnn_mem_t &weights_src_layer_m, dnn_mem_t &weights_src_iter_m,
        dnn_mem_t &weights_peephole_m, dnn_mem_t &weights_projection_m,
        dnn_mem_t &bias_m, dnn_mem_t &dst_layer_m, dnn_mem_t &dst_iter_m,
        dnn_mem_t &dst_iter_c_m) {
    std::vector<float> ws_fwd_buffer;
    AOC<float> ws_src_layer, ws_src_iter, ws_src_iter_c, ws_gates, ws_ht;
    prepare_ws_fwd(prb, ws_fwd_buffer, ws_src_layer, ws_src_iter, ws_src_iter_c,
            ws_gates, ws_ht);

    rnn_linear_fwd(prb, (float *)src_layer_m, (float *)src_iter_m,
            (float *)src_iter_c_m, (float *)weights_src_layer_m,
            (float *)weights_src_iter_m, (float *)weights_peephole_m,
            (float *)weights_projection_m, (float *)bias_m,
            (float *)dst_layer_m, (float *)dst_iter_m, (float *)dst_iter_c_m,
            ws_src_layer, ws_src_iter, ws_src_iter_c, ws_gates, ws_ht);
}

} // namespace rnn
