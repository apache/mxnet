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
 * Common for RNN and LSTM cell execution
 */

#include "common/bfloat16.hpp"

#include "cpu/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
using namespace rnn_utils;

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::cell_execution)) {
    auto src_layer_ld = rnn.src_layer_ld(cell_position);
    auto src_iter_ld = rnn.src_iter_ld(cell_position);

    if (rnn.need_gemm_layer(cell_position)) {
        CHECK((this->*gemm_layer_func)('N', 'N', rnn.n_gates * rnn.dhc, rnn.mb,
                rnn.slc, 1.0f, w_layer_[0], rnn.weights_layer_ld, src_layer_,
                src_layer_ld, 0.0f, scratch_gates_, rnn.scratch_gates_ld));
    }
    CHECK((this->*gemm_iter_func)('N', 'N', rnn.n_gates * rnn.dhc, rnn.mb,
            rnn.sic, 1.0f, w_iter_[0], rnn.weights_iter_ld, src_iter_,
            src_iter_ld, 1.0f, scratch_gates_, rnn.scratch_gates_ld));

    // Note: here proj_ht is scratchpad if inference or workspace if training
    auto dst_postgemm = rnn.is_lstm_projection ? proj_ht_ : dst_layer_;
    rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_postgemm, dst_iter_c_, src_iter_, src_iter_c_, diff_src_layer_,
            diff_src_iter_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            diff_dst_iter_c_, weights_peephole_, bias_[0], ws_grid_,
            scratch_cell_, dst_iter_);

    if (rnn.is_lstm_projection) {
        // Here, because the accumulation type is likely different
        // than dst_iter, we have to use scratch to hold temporary
        // accumulators
        // TODO: for projection, directly use a type(dst_iter) for output
        auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
        auto dst_iter_ld = rnn.dst_iter_ld(cell_position);

        if (rnn.dt_conf == all_f32) {
            CHECK((this->*gemm_projection_func)('N', 'N', rnn.dic, rnn.mb,
                    rnn.dhc, 1.0f, w_projection_[0], rnn.weights_projection_ld,
                    dst_postgemm, rnn.proj_ht_ld, 0.0f,
                    (gemm_acc_t *)dst_layer_,
                    rnn.dst_layer_ld(cell_position, true)));
        } else {
            CHECK((this->*gemm_projection_func)('N', 'N', rnn.dic, rnn.mb,
                    rnn.dhc, 1.0f, w_projection_[0], rnn.weights_projection_ld,
                    dst_postgemm, rnn.proj_ht_ld, 0.0f, scratch_gates_,
                    rnn.scratch_gates_ld));

            for (int i = 0; i < rnn.mb; i++)
                cvt_float_to_bfloat16(
                        (bfloat16_t *)dst_layer_ + i * dst_layer_ld,
                        (float *)scratch_gates_ + i * rnn.scratch_gates_ld,
                        rnn.dlc);
        }
        // If dst_iter is not nullptr, we need to copy the state to dst_iter
        if (dst_iter_ != nullptr)
            for (int i = 0; i < rnn.mb; i++)
                std::memcpy(dst_iter_ + i * dst_iter_ld,
                        dst_layer_ + i * dst_layer_ld,
                        rnn.dlc * sizeof(dst_layer_t));
    }

    return dnnl_success;
}

template rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution);
template rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution);
template rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution);

template <typename scratch_data_t, typename acc_data_t>
void lstm_bwd_weights_peephole_and_bias(const rnn_utils::rnn_conf_t &rnn,
        cell_position_t cell_position, const float *src_iter_c_,
        const float *dst_iter_c_, const scratch_data_t *scratch_gates_,
        float *diff_weights_peephole_, acc_data_t *diff_bias_) {
    auto dst_iter_c_ld = rnn.dst_iter_c_ld(cell_position);
    auto src_iter_c_ld = rnn.src_iter_c_ld(cell_position);

    ws_states_iter_c_aoc<const float> dst_iter_c(
            rnn, dst_iter_c_, dst_iter_c_ld);
    ws_states_iter_c_aoc<const float> src_iter_c(
            rnn, src_iter_c_, src_iter_c_ld);
    ws_gates_aoc<const scratch_data_t> scratch_gates(rnn, scratch_gates_);
    weights_peephole_aoc_t<float> diff_weights_peephole(
            rnn, diff_weights_peephole_);

    parallel(0, [&](int ithr, int nthr) {
        int g_dhc_start {}, g_dhc_stop {};
        const int gates_to_process = 5; // 3 -- weights peephole +
                // 2 -- bias (process a pair at once)
        balance211(gates_to_process * rnn.dhc, nthr, ithr, g_dhc_start,
                g_dhc_stop);
        int g = g_dhc_start / rnn.dhc;
        int dhc = g_dhc_start % rnn.dhc;
        while (g_dhc_start++ < g_dhc_stop) {
            if (g < 3) {
                // weights peephole
                auto &c_states = g < 2 ? src_iter_c : dst_iter_c;
                const int scratch_g = g < 2 ? g : 3;
                for (int mb = 0; mb < rnn.mb; ++mb) {
                    diff_weights_peephole(g, dhc) += c_states(mb, dhc)
                            * scratch_gates(mb, scratch_g, dhc);
                }
            } else {
                // bias
                const int bias_g_start = 2 * (g - 3);
                const int bias_g_end = bias_g_start + 2;
                for_(int bias_g = bias_g_start; bias_g < bias_g_end; ++bias_g)
                for (int mb = 0; mb < rnn.mb; ++mb)
                    diff_bias_[bias_g * rnn.dhc + dhc]
                            += scratch_gates(mb, bias_g, dhc);
            }
            if (++dhc == rnn.dhc) {
                dhc = 0;
                g++;
            }
        }
    });
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
        typename T6, typename T7, typename weights_data_t, typename src_data_t,
        typename acc_data_t, typename scratch_data_t>
dnnl_status_t common_bwd_cell_exec_template(T1 gemm_layer_f, T2 gemm_iter_f,
        T3 gemm_proj_f, T4 gemm_weights_layer_f, T5 gemm_weights_iter_f,
        T6 gemm_weights_proj_f, T7 rnn_postgemm,
        const rnn_utils::rnn_conf_t &rnn, const cell_position_t cell_position,
        src_data_t *dst_layer_, float *dst_iter_c_, acc_data_t *diff_src_layer_,
        acc_data_t *diff_src_iter_, acc_data_t *diff_src_iter_c_,
        weights_data_t **w_layer_, weights_data_t **w_iter_,
        weights_data_t **w_proj_, const float *weights_peephole_, float **bias_,
        const src_data_t *src_layer_, const src_data_t *src_iter_,
        const float *src_iter_c_, acc_data_t *diff_dst_layer_,
        acc_data_t *diff_dst_iter_, acc_data_t *diff_dst_iter_c_,
        acc_data_t *diff_w_layer_, acc_data_t *diff_w_iter_,
        float *diff_weights_projection_, float *diff_weights_peephole_,
        acc_data_t *diff_bias_, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *ws_ht_,
        acc_data_t *scratch_diff_ht_, src_data_t *ws_grid_,
        scratch_data_t *scratch_cell_, src_data_t *dst_iter_) {

    if (rnn.is_lstm_projection) {
        parallel_nd(rnn.mb, [&](int i) {
            PRAGMA_OMP_SIMD()
            for (int j = 0; j < rnn.dlc; j++)
                scratch_diff_ht_[i * rnn.scratch_diff_ht_ld + j]
                        = diff_dst_layer_[i * rnn.ws_diff_states_layer_ld + j]
                        + diff_dst_iter_[i * rnn.ws_diff_states_iter_ld + j];
        });

        CHECK(gemm_weights_proj_f(
                scratch_diff_ht_, ws_ht_, diff_weights_projection_));
        CHECK(gemm_proj_f(w_proj_[0], scratch_diff_ht_, diff_dst_layer_));
    }

    rnn_postgemm->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            dst_layer_, dst_iter_c_, src_iter_, src_iter_c_, diff_src_layer_,
            diff_src_iter_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            diff_dst_iter_c_, weights_peephole_, bias_[0], ws_grid_,
            scratch_cell_, dst_iter_);

    /// bwd by data on the cell
    CHECK(gemm_iter_f(w_iter_[0], scratch_gates_, diff_src_iter_));

    /// bwd by weights on the cell
    if (rnn.need_gemm_layer(cell_position))
        CHECK(gemm_weights_layer_f(scratch_gates_, src_layer_, diff_w_layer_));

    if (!rnn.merge_gemm_layer)
        CHECK(gemm_layer_f(w_layer_[0], scratch_gates_, diff_src_layer_));

    if (!rnn.merge_gemm_iter)
        CHECK(gemm_weights_iter_f(scratch_gates_, src_iter_, diff_w_iter_));

    if (rnn.is_lstm_peephole) {
        /// bwd by weights peephole and bias
        lstm_bwd_weights_peephole_and_bias(rnn, cell_position, src_iter_c_,
                dst_iter_c_, scratch_gates_, diff_weights_peephole_,
                diff_bias_);
    } else {
        /// bwd by bias we just accumulate diffs from the gates
        gates_reduction(rnn, scratch_gates_, diff_bias_);
    }
    return dnnl_success;
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution) {
    auto gemm_layer = [&](const float *A, const float *B, float *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_iter = [&](const float *A, const float *B, float *C) {
        return (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_iter_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_iter_ld);
    };
    auto gemm_proj = [&](const float *A, const float *B, float *C) {
        return (this->*gemm_projection_func)('N', 'N', rnn.dhc, rnn.mb, rnn.dic,
                1.0, A, rnn.weights_projection_ld, B, rnn.scratch_diff_ht_ld,
                0.0f, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_weights_layer = [&](const float *A, const float *B, float *C) {
        auto src_layer_ld = rnn.src_layer_ld(cell_position);
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb, 1.0, A,
                rnn.scratch_gates_ld, B, src_layer_ld, 1.0, C,
                rnn.diff_weights_layer_ld);
    };
    auto gemm_weights_iter = [&](const float *A, const float *B, float *C) {
        auto src_iter_ld = rnn.src_iter_ld(cell_position);
        return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.sic, rnn.mb, 1.0, A,
                rnn.scratch_gates_ld, B, src_iter_ld, 1.0, C,
                rnn.diff_weights_iter_ld);
    };
    auto gemm_weights_proj = [&](const float *A, const float *B, float *C) {
        return gemm('N', 'T', rnn.dlc, rnn.dhc, rnn.mb, 1.0f, A,
                rnn.scratch_diff_ht_ld, B, rnn.ws_ht_ld, 1.0f, C,
                rnn.diff_weights_projection_ld);
    };
    return common_bwd_cell_exec_template(gemm_layer, gemm_iter, gemm_proj,
            gemm_weights_layer, gemm_weights_iter, gemm_weights_proj,
            rnn_postgemm_, rnn, cell_position, dst_layer_, dst_iter_c_,
            diff_src_layer_, diff_src_iter_, diff_src_iter_c_, w_layer_,
            w_iter_, w_projection_, weights_peephole_, bias_, src_layer_,
            src_iter_, src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            diff_dst_iter_c_, diff_w_layer_, diff_w_iter_,
            diff_weights_projection_, diff_weights_peephole_, diff_bias_,
            ws_gates_, scratch_gates_, proj_ht_, scratch_diff_ht_, ws_grid_,
            scratch_cell_, dst_iter_);
}

template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution) {
    auto gemm_layer = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
        return (this->*gemm_layer_func)('N', 'N', rnn.slc, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_layer_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_layer_ld);
    };
    auto gemm_iter = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
        return (this->*gemm_iter_func)('N', 'N', rnn.sic, rnn.mb,
                rnn.n_gates * rnn.dhc, 1.0, A, rnn.weights_iter_ld, B,
                rnn.scratch_gates_ld, 0.0, C, rnn.ws_diff_states_iter_ld);
    };
    auto gemm_proj = [&](const bfloat16_t *, const float *, float *) {
        assert(!"unimplemented");
        return dnnl_unimplemented;
    };
    auto gemm_weights_layer
            = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
                  auto src_layer_ld = rnn.src_layer_ld(cell_position);
                  return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.slc, rnn.mb,
                          1.0, A, rnn.scratch_gates_ld, B, src_layer_ld, 1.0, C,
                          rnn.diff_weights_layer_ld);
              };
    auto gemm_weights_iter
            = [&](const bfloat16_t *A, const bfloat16_t *B, float *C) {
                  auto src_iter_ld = rnn.src_iter_ld(cell_position);
                  return gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.sic, rnn.mb,
                          1.0, A, rnn.scratch_gates_ld, B, src_iter_ld, 1.0, C,
                          rnn.diff_weights_iter_ld);
              };
    auto gemm_weights_proj = [&](const float *, const bfloat16_t *, float *) {
        assert(!"unimplemented");
        return dnnl_unimplemented;
    };
    return common_bwd_cell_exec_template(gemm_layer, gemm_iter, gemm_proj,
            gemm_weights_layer, gemm_weights_iter, gemm_weights_proj,
            rnn_postgemm_, rnn, cell_position, dst_layer_, dst_iter_c_,
            diff_src_layer_, diff_src_iter_, diff_src_iter_c_, w_layer_,
            w_iter_, w_projection_, weights_peephole_, bias_, src_layer_,
            src_iter_, src_iter_c_, diff_dst_layer_, diff_dst_iter_,
            diff_dst_iter_c_, diff_w_layer_, diff_w_iter_,
            diff_weights_projection_, diff_weights_peephole_, diff_bias_,
            ws_gates_, scratch_gates_, proj_ht_, scratch_diff_ht_, ws_grid_,
            scratch_cell_, dst_iter_);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
