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
 * Cell execution GRU with linear before reset
 */
#pragma warning(disable : 4503) /* name is too long */

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"

#include "cpu/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;
#define AOC array_offset_calculator

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::cell_execution_gru_lbr)) {
    auto src_layer_ld = rnn.src_layer_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);

    if (rnn.need_gemm_layer(cell_position)) {
        CHECK((this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dhc, rnn.mb,
                rnn.slc, 1.0, w_layer_[0], rnn.weights_layer_ld, src_layer_,
                src_layer_ld, 0.0, scratch_gates_, rnn.scratch_gates_ld));
    }
    CHECK((this->*gemm_iter_func)('N', 'N', rnn.n_gates * rnn.dhc, rnn.mb,
            rnn.sic, 1.0, w_iter_[0], rnn.weights_iter_ld, src_iter_,
            src_iter_ld, 0.0, scratch_cell_, rnn.ws_gates_ld));

    rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_layer_, dst_iter_c_, src_iter_, src_iter_c_, diff_src_layer_,
            diff_src_iter_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            nullptr, nullptr, bias_[0], ws_grid_, scratch_cell_, dst_iter_);

    return dnnl_success;
}

template rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru_lbr);
template rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_gru_lbr);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru_lbr) {
    assert(!"GRU LBR int8 is not supported");
    return dnnl_unimplemented;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename weights_data_t, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
dnnl_status_t common_bwd_cell_exec_template(T1 gemm_layer_f, T2 gemm_iter_f,
        T3 gemm_weights_layer_f, T4 gemm_weights_iter_f, T5 rnn_postgemm,
        const rnn_utils::rnn_conf_t &rnn, cell_position_t cell_position,
        src_data_t *dst_layer_, acc_data_t *diff_src_layer_,
        acc_data_t *diff_src_iter_, weights_data_t **w_layer_,
        weights_data_t **w_iter_, float **bias_, const src_data_t *src_layer_,
        const src_data_t *src_iter_, acc_data_t *diff_dst_layer_,
        acc_data_t *diff_dst_iter_, acc_data_t *diff_w_layer_,
        acc_data_t *diff_w_iter_, acc_data_t *diff_bias_, src_data_t *ws_gates_,
        src_data_t *ws_grid_, scratch_data_t *scratch_gates_,
        scratch_data_t *scratch_cell_, src_data_t *dst_iter_) {
    auto src_layer_ld = rnn.src_layer_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);

    ws_gates_aoc<scratch_data_t> scratch_gates_r(rnn, scratch_cell_);

    rnn_postgemm->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_layer_, nullptr, src_iter_, nullptr, nullptr, diff_src_iter_,
            nullptr, diff_dst_layer_, diff_dst_iter_, nullptr, nullptr,
            bias_[0], ws_grid_, scratch_cell_, dst_iter_);

    // dWx +=  dG^t * x
    if (rnn.need_gemm_layer(cell_position))
        CHECK(gemm_weights_layer_f(
                scratch_gates_, src_layer_, src_layer_ld, diff_w_layer_));

    //  dx = dG * Wx^t
    if (!rnn.merge_gemm_layer)
        CHECK(gemm_layer_f(w_layer_[0], scratch_gates_, diff_src_layer_));

    // dh +=  dGr * Wh^t
    CHECK(gemm_iter_f(w_iter_[0], scratch_cell_, diff_src_iter_));

    // dWh += dGr^t * h
    CHECK(gemm_weights_iter_f(
            scratch_cell_, src_iter_, src_iter_ld, diff_w_iter_));

    // db1-3 += e * dG
    // db4 += e * (r * dG2)
    gates_reduction(rnn, scratch_gates_, diff_bias_);

    parallel_nd(rnn.dhc, [&](int j) {
        for (int i = 0; i < rnn.mb; i++) {
            diff_bias_[3 * rnn.dhc + j] += scratch_gates_r(i, 2, j);
        }
    });

    return dnnl_success;
}

#undef AOC

template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru_lbr) {
    auto gemm_layer = [&](const float *A, const float *B, float *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0f, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0f, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_iter = [&](const float *A, const float *B, float *C) {
        return (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0f, A, rnn.weights_iter_ld, B,
                rnn.ws_gates_ld, 1.0f, C, rnn.ws_diff_states_iter_ld);
    };
    auto gemm_weights_layer
            = [&](const float *A, const float *B, int ldb, float *C) {
                  return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb,
                          1.0f, A, rnn.scratch_gates_ld, B, ldb, 1.0f, C,
                          rnn.diff_weights_layer_ld);
              };
    auto gemm_weights_iter = [&](const float *A, const float *B, int ldb,
                                     float *C) {
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.sic, rnn.mb, 1.0f, A,
                rnn.ws_gates_ld, B, ldb, 1.0f, C, rnn.diff_weights_iter_ld);
    };

    common_bwd_cell_exec_template(gemm_layer, gemm_iter, gemm_weights_layer,
            gemm_weights_iter, rnn_postgemm_, rnn, cell_position, dst_layer_,
            diff_src_layer_, diff_src_iter_, w_layer_, w_iter_, bias_,
            src_layer_, src_iter_, diff_dst_layer_, diff_dst_iter_,
            diff_w_layer_, diff_w_iter_, diff_bias_, ws_gates_, ws_grid_,
            scratch_gates_, scratch_cell_, dst_iter_);

    return dnnl_success;
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_gru_lbr) {
    auto gemm_layer = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0f, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0f, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_iter = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
        return (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0f, A, rnn.weights_iter_ld, B,
                rnn.ws_gates_ld, 1.0f, C, rnn.ws_diff_states_iter_ld);
    };
    auto gemm_weights_layer
            = [&](const bfloat16_t *A, const bfloat16_t *B, int ldb, float *C) {
                  return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb,
                          1.0f, A, rnn.scratch_gates_ld, B, ldb, 1.0f, C,
                          rnn.diff_weights_layer_ld);
              };
    auto gemm_weights_iter = [&](const bfloat16_t *A, const bfloat16_t *B,
                                     int ldb, float *C) {
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.sic, rnn.mb, 1.0f, A,
                rnn.ws_gates_ld, B, ldb, 1.0f, C, rnn.diff_weights_iter_ld);
    };

    common_bwd_cell_exec_template(gemm_layer, gemm_iter, gemm_weights_layer,
            gemm_weights_iter, rnn_postgemm_, rnn, cell_position, dst_layer_,
            diff_src_layer_, diff_src_iter_, w_layer_, w_iter_, bias_,
            src_layer_, src_iter_, diff_dst_layer_, diff_dst_iter_,
            diff_w_layer_, diff_w_iter_, diff_bias_, ws_gates_, ws_grid_,
            scratch_gates_, scratch_cell_, dst_iter_);
    return dnnl_success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
