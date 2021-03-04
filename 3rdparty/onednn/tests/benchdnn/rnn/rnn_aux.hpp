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

#ifndef BENCHDNN_RNN_AUX_HPP
#define BENCHDNN_RNN_AUX_HPP

#include <assert.h>
#include <stdlib.h>
#include "rnn/rnn.hpp"

namespace rnn {

typedef enum {
    rnn_forward = 0,
    rnn_backward,
} rnn_propagation_t;

typedef enum {
    left2right = 0,
    right2left,
} rnn_iter_direction_t;

typedef enum {
    bottom2top = 0,
    top2bottom,
} rnn_layer_direction_t;

typedef enum { action_copy = 0, action_sum, action_concat } rnn_action_t;

dnnl_status_t init_rnn_fwd_desc(dnnl_rnn_desc_t *rd, const prb_t &prb,
        dnnl_prop_kind_t prop_kind, const dnnl_memory_desc_t *src_layer_d,
        const dnnl_memory_desc_t *src_iter_d,
        const dnnl_memory_desc_t *src_iter_c_d,
        const dnnl_memory_desc_t *weights_layer_d,
        const dnnl_memory_desc_t *weights_iter_d,
        const dnnl_memory_desc_t *weights_peephole_d,
        const dnnl_memory_desc_t *weights_projection_d,
        const dnnl_memory_desc_t *bias_d, const dnnl_memory_desc_t *dst_layer_d,
        const dnnl_memory_desc_t *dst_iter_d,
        const dnnl_memory_desc_t *dst_iter_c_d);

dnnl_status_t init_rnn_bwd_desc(dnnl_rnn_desc_t *rd, const prb_t &prb,
        dnnl_prop_kind_t prop_kind, const dnnl_memory_desc_t *src_layer_d,
        const dnnl_memory_desc_t *src_iter_d,
        const dnnl_memory_desc_t *src_iter_c_d,
        const dnnl_memory_desc_t *weights_layer_d,
        const dnnl_memory_desc_t *weights_iter_d,
        const dnnl_memory_desc_t *weights_peephole_d,
        const dnnl_memory_desc_t *weights_projection_d,
        const dnnl_memory_desc_t *bias_d, const dnnl_memory_desc_t *dst_layer_d,
        const dnnl_memory_desc_t *dst_iter_d,
        const dnnl_memory_desc_t *dst_iter_c_d,
        const dnnl_memory_desc_t *diff_src_layer_d,
        const dnnl_memory_desc_t *diff_src_iter_d,
        const dnnl_memory_desc_t *diff_src_iter_c_d,
        const dnnl_memory_desc_t *diff_weights_layer_d,
        const dnnl_memory_desc_t *diff_weights_iter_d,
        const dnnl_memory_desc_t *diff_weights_peephole_d,
        const dnnl_memory_desc_t *diff_weights_projection_d,
        const dnnl_memory_desc_t *diff_bias_d,
        const dnnl_memory_desc_t *diff_dst_layer_d,
        const dnnl_memory_desc_t *diff_dst_iter_d,
        const dnnl_memory_desc_t *diff_dst_iter_c_d);

void init_buffer(float *buf, int64_t size, float value);

float maybe_q(const prb_t &prb, float h);
float maybe_deq(const prb_t &prb, const float in);
float maybe_deq(const prb_t &prb, const float in, int64_t oc);

float logistic(float x);
float dlogistic(float x);
float relu(float x, float alpha);
float drelu(float x, float alpha);
float dtanhf(float x);
float one_m_square(float x);
float x_m_square(float x);

void copy(int64_t dimc, int64_t dimr, int64_t ld_src, int64_t ld_dst,
        const float *src_, float *dst_, rnn_action_t action = action_copy,
        bool saturate_to_u8 = false);
void data_q10n(int64_t dimc, int64_t dimr, int64_t ld_src, float *src_,
        float data_scale, float data_shift);
void data_deq10n(int64_t dimc, int64_t dimr, int64_t ld_src, float *src_,
        float data_scale, float data_shift);
void gates_reduction(
        const prb_t &prb, const float *b_gates_, float *diff_bias_);

int compare_dat(const prb_t &prb, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res, bool final_compare);

}; // namespace rnn

#endif
