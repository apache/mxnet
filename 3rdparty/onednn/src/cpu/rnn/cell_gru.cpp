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
 * Cell execution GRU
 */

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
        acc_type>::cell_execution_gru)) {
    ws_gates_aoc<gates_t> ws_gates(rnn, ws_gates_);
    scratch_gates_aoc<scratch_t> scratch_gates(rnn, scratch_gates_);
    bias_aoc_t bias(rnn, bias_[0]);

    auto src_layer_ld = rnn.src_layer_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);
    auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
    auto dst_iter_ld = rnn.dst_iter_ld(cell_position);

    // 1. gemm Wx[0-2],x
    if (rnn.need_gemm_layer(cell_position)) {
        CHECK((this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dhc, rnn.mb,
                rnn.slc, 1.0, w_layer_[0], rnn.weights_layer_ld, src_layer_,
                src_layer_ld, 0.0f, scratch_gates_, rnn.scratch_gates_ld));
    }

    // 2. gemm Wh[0-1],h
    CHECK((this->*gemm_iter_func)('N', 'N', (rnn.n_gates - 1) * rnn.dhc, rnn.mb,
            rnn.sic, 1.0, w_iter_[0], rnn.weights_iter_ld, src_iter_,
            src_iter_ld, 1.0f, scratch_gates_, rnn.scratch_gates_ld));

    // 3. activation zt and rt + elemwise multiplication rt,ht-1
    rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_layer_, nullptr, src_iter_, nullptr, diff_src_layer_,
            diff_src_iter_, nullptr, diff_dst_layer_, diff_dst_iter_, nullptr,
            nullptr, bias_[0], nullptr, nullptr, dst_iter_);

    // 4. gemm Wh[2],h~t
    CHECK((this->*gemm_iter_func)('N', 'N', rnn.dhc, rnn.mb, rnn.sic, 1.0,
            w_iter_[1], rnn.weights_iter_ld, dst_layer_,
            (cell_position & last_layer) ? dst_layer_ld : dst_iter_ld, 1.0,
            &(scratch_gates(0, 2, 0)), rnn.scratch_gates_ld));

    // 5. activation h~t + calculate ht
    rnn_postgemm_->execute_part2(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_layer_, dst_iter_c_, src_iter_, src_iter_c_, diff_src_layer_,
            diff_src_iter_, nullptr, diff_dst_layer_, diff_dst_iter_, nullptr,
            nullptr, bias_[0], nullptr, nullptr, dst_iter_);

    return dnnl_success;
}

template rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru);
template rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_gru);
template rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru);

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename weights_data_t, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
dnnl_status_t gru_bwd_cell_exec_template(T1 gemm_layer_f, T2 gemm_iter_f,
        T3 gemm_weights_layer_f, T4 gemm_weights_iter_f, T5 rnn_postgemm_,
        const rnn_utils::rnn_conf_t &rnn, cell_position_t cell_position,
        src_data_t *ws_gates_, scratch_data_t *scratch_gates_,
        src_data_t *dst_layer_, const src_data_t *src_iter_,
        const src_data_t *src_layer_, weights_data_t **w_layer_,
        weights_data_t **w_iter_, acc_data_t *diff_w_layer_,
        acc_data_t *diff_w_iter_, acc_data_t *diff_src_layer_,
        acc_data_t *diff_src_iter_, acc_data_t *diff_dst_iter_,
        acc_data_t *diff_dst_layer_, acc_data_t *diff_bias_,
        scratch_data_t *scratch_cell_, src_data_t *dst_iter_) {
    ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    scratch_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);

    auto src_layer_ld = rnn.src_layer_ld(cell_position);
    auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
    auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);
    ws_states_layer_aoc<src_data_t> dst_layer(rnn, dst_layer_,
            (cell_position & last_layer) ? dst_layer_ld : dst_iter_ld);
    ws_states_iter_aoc<const src_data_t> src_iter(rnn, src_iter_, src_iter_ld);
    ws_diff_w_iter_aoc_t diff_w_iter(rnn, diff_w_iter_);

    // use state memory for intermediate computations
    // TODO: use cell ws for that
    float *dhG1_ = diff_src_layer_;
    AOC<acc_data_t, 2> dhG1(
            dhG1_, rnn.ws_states_layer_nld, rnn.ws_states_layer_ld);
    // hg1 needs to be bf16 as it is used as gemm output
    // hence it cannot alias to dhG1, and should use scratch_cell
    AOC<scratch_data_t, 2> hG1(
            scratch_cell_, rnn.ws_states_layer_nld, rnn.ws_states_layer_ld);

    // 1. calculate dG2, dG1, and part of dht-1
    rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_layer_, nullptr, src_iter_, nullptr, diff_src_layer_,
            diff_src_iter_, nullptr, diff_dst_layer_, diff_dst_iter_, nullptr,
            nullptr, nullptr, nullptr, scratch_cell_, dst_iter_);

    // 2. calculate intermediate d(hG1)
    // d(hG1) = dG2 * W2h^t
    CHECK(gemm_iter_f(rnn.sic, rnn.mb, rnn.dhc, w_iter_[1],
            &(scratch_gates(0, 2, 0)), 0.0f, dhG1_));

    // 3. calculate dG1^ and part of dht-1
    rnn_postgemm_->execute_part2(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_layer_, nullptr, src_iter_, nullptr, diff_src_layer_,
            diff_src_iter_, nullptr, diff_dst_layer_, diff_dst_iter_, nullptr,
            nullptr, nullptr, nullptr, scratch_cell_, dst_iter_);

    // 4. calculate diff weights
    // dWh1 += dG1 * h, dWh2 += dG2 * h, dWh3 += dG3 * (G1(*)h)
    CHECK(gemm_weights_iter_f((rnn.n_gates - 1) * rnn.dhc, rnn.sic, rnn.mb,
            scratch_gates_, src_iter_, src_iter_ld, 1.0f, diff_w_iter_));
    CHECK(gemm_weights_iter_f(rnn.dhc, rnn.sic, rnn.mb,
            &(scratch_gates(0, 2, 0)), scratch_cell_, rnn.ws_states_layer_ld,
            1.0f, &(diff_w_iter(0, 2, 0))));

    // 5. calculate diff states
    // dht-1 += dG1 * W1h + dG0 * W0h
    CHECK(gemm_iter_f(rnn.sic, rnn.mb, (rnn.n_gates - 1) * rnn.dhc, w_iter_[0],
            scratch_gates_, 1.0f, diff_src_iter_));

    // dWx += [dG0 dG1 dG2] * [x]
    if (rnn.need_gemm_layer(cell_position))
        CHECK(gemm_weights_layer_f(
                scratch_gates_, src_layer_, src_layer_ld, diff_w_layer_));

    // dx = dG2 * W2x + dG1 * W1x + dG0 * W0x
    if (!rnn.merge_gemm_layer)
        CHECK(gemm_layer_f(w_layer_[0], scratch_gates_, diff_src_layer_));

    // 6. calculate diff bias
    gates_reduction(rnn, scratch_gates_, diff_bias_);

    return dnnl_success;
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru) {
    auto gemm_iter_f
            = [&](int m, int n, int k, const weights_t *A, const gemm_data_t *B,
                      float beta, gemm_acc_t *C) {
                  return (this->*gemm_iter_func)('N', 'N', m, n, k, 1.0f, A,
                          rnn.weights_iter_ld, B, rnn.scratch_gates_ld, beta, C,
                          rnn.ws_diff_states_iter_ld);
              };
    auto gemm_layer_f = [&](const weights_t *A, const gemm_data_t *B,
                                gemm_acc_t *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_weights_layer_f = [&](const gemm_data_t *A, const weights_t *B,
                                        int ldb, gemm_acc_t *C) {
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb, 1.0, A,
                rnn.scratch_gates_ld, B, ldb, 1.0, C,
                rnn.diff_weights_layer_ld);
    };
    auto gemm_weights_iter_f
            = [&](int m, int n, int k, const weights_t *A, const gemm_data_t *B,
                      int ldb, float beta, gemm_acc_t *C) {
                  return gemm('N', 'T', m, n, k, 1.0f, A, rnn.ws_gates_ld, B,
                          ldb, 1.0f, C, rnn.diff_weights_iter_ld);
              };

    return gru_bwd_cell_exec_template(gemm_layer_f, gemm_iter_f,
            gemm_weights_layer_f, gemm_weights_iter_f, this->rnn_postgemm_, rnn,
            cell_position, ws_gates_, scratch_gates_, dst_layer_, src_iter_,
            src_layer_, w_layer_, w_iter_, diff_w_layer_, diff_w_iter_,
            diff_src_layer_, diff_src_iter_, diff_dst_iter_, diff_dst_layer_,
            diff_bias_, scratch_cell_, dst_iter_);
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_gru) {
    auto gemm_iter_f
            = [&](int m, int n, int k, const weights_t *A, const gemm_data_t *B,
                      float beta, gemm_acc_t *C) {
                  return (this->*gemm_iter_func)('N', 'N', m, n, k, 1.0f, A,
                          rnn.weights_iter_ld, B, rnn.scratch_gates_ld, beta, C,
                          rnn.ws_diff_states_iter_ld);
              };
    auto gemm_layer_f = [&](const weights_t *A, const gemm_data_t *B,
                                gemm_acc_t *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_weights_layer_f = [&](const gemm_data_t *A, const weights_t *B,
                                        int ldb, gemm_acc_t *C) {
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb, 1.0, A,
                rnn.scratch_gates_ld, B, ldb, 1.0, C,
                rnn.diff_weights_layer_ld);
    };
    auto gemm_weights_iter_f
            = [&](int m, int n, int k, const weights_t *A, const gemm_data_t *B,
                      int ldb, float beta, gemm_acc_t *C) {
                  return gemm('N', 'T', m, n, k, 1.0f, A, rnn.ws_gates_ld, B,
                          ldb, 1.0f, C, rnn.diff_weights_iter_ld);
              };

    return gru_bwd_cell_exec_template(gemm_layer_f, gemm_iter_f,
            gemm_weights_layer_f, gemm_weights_iter_f, this->rnn_postgemm_, rnn,
            cell_position, ws_gates_, scratch_gates_, dst_layer_, src_iter_,
            src_layer_, w_layer_, w_iter_, diff_w_layer_, diff_w_iter_,
            diff_src_layer_, diff_src_iter_, diff_dst_iter_, diff_dst_layer_,
            diff_bias_, scratch_cell_, dst_iter_);
}

#undef AOC
} // namespace cpu
} // namespace impl
} // namespace dnnl
