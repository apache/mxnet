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

#ifndef BENCHDNN_RNN_CELLS_HPP
#define BENCHDNN_RNN_CELLS_HPP

#include "rnn/rnn.hpp"

namespace rnn {

void rnn_fwd(const prb_t &prb, float *dst_layer_, float *gates_,
        const float *weights_layer_, const float *weights_iter_,
        const float *bias_, const float *src_layer_, const float *src_iter_);

void rnn_bwd(const prb_t &prb, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_, const float *bias_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_);

void lstm_fwd(const prb_t &prb, float *dst_layer_, float *dst_iter_,
        float *dst_iter_c_, float *gates_, float *ht_,
        const float *weights_layer_, const float *weights_iter_,
        const float *weights_peephole_, const float *weights_projection_,
        const float *bias_, const float *src_layer_, const float *src_iter_,
        const float *src_iter_c_);

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
        float *cell_scratchpad_);

void gru_fwd(const prb_t &prb, float *dst_layer_, float *gates_,
        const float *weights_layer_, const float *weights_iter_,
        const float *bias_, const float *src_layer_, const float *src_iter_);

void gru_bwd(const prb_t &prb, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_, const float *bias_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_,
        float *cell_scratchpad_);

void lbr_gru_fwd(const prb_t &prb, float *dst_layer_, float *gates_,
        const float *weights_layer_, const float *weights_iter_,
        const float *bias_, const float *src_layer_, const float *src_iter_,
        float *cell_scratchpad_);

void lbr_gru_bwd(const prb_t &prb, float *diff_src_layer_,
        float *diff_src_iter_, float *diff_weights_layer_,
        float *diff_weights_iter_, float *diff_bias_, float *b_gates_,
        const float *src_layer_, const float *src_iter_,
        const float *weights_layer_, const float *weights_iter_,
        const float *bias_, const float *gates_, const float *diff_dst_layer_,
        const float *diff_dst_iter_, float *cell_scratchpad_);

} // namespace rnn

#endif
