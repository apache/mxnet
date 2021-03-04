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

#ifndef CPU_RNN_POSTGEMM_DISPATCHER_HPP
#define CPU_RNN_POSTGEMM_DISPATCHER_HPP

#include <memory>

#include "common/z_magic.hpp"

#include "cpu/platform.hpp"

#include "cpu/rnn/cpu_rnn_pd.hpp"
#include "cpu/rnn/rnn_utils.hpp"

#if DNNL_X64
#include "cpu/x64/rnn/jit_uni_gru_cell_postgemm_1_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_cell_postgemm_1_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_cell_postgemm_2_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_cell_postgemm_2_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_lbr_cell_postgemm_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_gru_lbr_cell_postgemm_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_lstm_cell_postgemm_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_lstm_cell_postgemm_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_cell_postgemm_bwd.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_cell_postgemm_fwd.hpp"
#include "cpu/x64/rnn/jit_uni_rnn_common_postgemm.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

template <alg_kind_t alg_kind, prop_kind_t prop_kind>
float activation(float s, float alpha, float cliping);

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t scratch_type, impl::data_type_t acc_type>
struct rnn_postgemm_dispatcher {

    typedef typename prec_traits<src_type>::type src_layer_t;
    typedef typename prec_traits<src_type>::type src_iter_t;
    typedef typename prec_traits<src_type>::type dst_layer_t;
    typedef typename prec_traits<src_type>::type dst_iter_t;
    typedef typename prec_traits<acc_type>::type gemm_acc_t;
    typedef typename prec_traits<scratch_type>::type scratch_t;
    typedef typename prec_traits<src_type>::type ht_t;
    typedef typename prec_traits<src_type>::type gates_t;

    using class_name
            = rnn_postgemm_dispatcher<aprop, src_type, scratch_type, acc_type>;
    typedef rnn_postgemm_sig((class_name::*postgemm_f));

    rnn_postgemm_dispatcher(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : pd_(pd) {
        // add check if in testing mode
        if (pd->attr()->rnn_tparams_.test_mode_) {
            auto ngates = utils::map(pd->cell_kind(), 0, alg_kind::vanilla_rnn,
                    1, alg_kind::vanilla_lstm, 4, alg_kind::vanilla_gru, 3,
                    alg_kind::lbr_gru, 3);
            assert(pd->attr()->rnn_tparams_.ngates_ == ngates);
            MAYBE_UNUSED(ngates);
        }

        switch (pd->cell_kind()) {
            case alg_kind::vanilla_lstm:
                postgemm_func = &class_name::lstm_postgemm;
                break;
            case alg_kind::vanilla_rnn:
                postgemm_func = &class_name::rnn_postgemm;
                switch (pd->activation_kind()) {
                    case alg_kind::eltwise_relu:
                        activation_func
                                = &activation<alg_kind::eltwise_relu, aprop>;
                        break;
                    case alg_kind::eltwise_tanh:
                        activation_func
                                = &activation<alg_kind::eltwise_tanh, aprop>;
                        break;
                    case alg_kind::eltwise_logistic:
                        activation_func
                                = &activation<alg_kind::eltwise_logistic,
                                        aprop>;
                        break;
                    default: assert(!"Unsupported activation function"); break;
                }
                break;
            case alg_kind::vanilla_gru:
                postgemm_func = &class_name::gru_part1_postgemm;
                postgemm_part2_func = &class_name::gru_part2_postgemm;
                break;
            case alg_kind::lbr_gru:
                postgemm_func = &class_name::gru_lbr_postgemm;
                break;
            default: assert(!"Unsupported algorithm kind"); break;
        }

        DNNL_X64_ONLY(initialize_jit(rnn));
    }

    ~rnn_postgemm_dispatcher() = default;

    rnn_postgemm_sig(unpoison) {
        // XXX (rsdubtso): This is a big hammer that unpoisons everything
        // that a postgemm may touch to avoid writing per-cell-kind
        // versions of unpoisoning code. This must be removed alongside with
        // the big unpoison_outputs() hammer in common/primitive.cpp.

        size_t states_nelems = rnn.ws_states_layer_nld * rnn.ws_states_layer_ld;
        size_t gates_nelems = rnn.scratch_gates_nld * rnn.scratch_gates_ld;

        if (pd_->is_fwd()) {
            msan_unpoison(dst_layer_, sizeof(*dst_layer_) * states_nelems);
            msan_unpoison(dst_iter_, sizeof(*dst_iter_) * states_nelems);
            if (rnn.is_training)
                msan_unpoison(ws_gates_, sizeof(*ws_gates_) * gates_nelems);
        } else {
            msan_unpoison(diff_src_layer_,
                    sizeof(*diff_src_layer_) * (rnn.n_iter + 1)
                            * rnn.ws_diff_states_layer_nld
                            * rnn.ws_diff_states_layer_ld);
            msan_unpoison(diff_src_iter_,
                    sizeof(*diff_src_iter_) * (rnn.n_iter + 1)
                            * rnn.ws_diff_states_iter_nld
                            * rnn.ws_diff_states_iter_ld);
            msan_unpoison(diff_src_iter_c_,
                    sizeof(*diff_src_iter_c_) * (rnn.n_iter + 1)
                            * rnn.ws_diff_states_iter_c_nld
                            * rnn.ws_diff_states_iter_c_ld);
            msan_unpoison(
                    scratch_gates_, sizeof(*scratch_gates_) * gates_nelems);
            msan_unpoison(
                    scratch_cell_, sizeof(*scratch_cell_) * states_nelems);
        }
    }

    // template <typename src_data_t, typename acc_data_t>
    rnn_postgemm_sig(execute) {
#if DNNL_X64
        if (rnn_postgemm_) {
            rnn_postgemm_->execute(rnn, cell_position, ws_gates_,
                    scratch_gates_, dst_layer_, dst_iter_c_, src_iter_,
                    src_iter_c_, diff_src_layer_, diff_src_iter_,
                    diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
                    diff_dst_iter_c_, weights_peephole_, bias_, ws_grid_,
                    scratch_cell_, dst_iter_);
            unpoison(rnn, cell_position, ws_gates_, scratch_gates_, dst_layer_,
                    dst_iter_c_, src_iter_, src_iter_c_, diff_src_layer_,
                    diff_src_iter_, diff_src_iter_c_, diff_dst_layer_,
                    diff_dst_iter_, diff_dst_iter_c_, weights_peephole_, bias_,
                    ws_grid_, scratch_cell_, dst_iter_);
            return;
        }
#endif
        (this->*postgemm_func)(rnn, cell_position, ws_gates_, scratch_gates_,
                dst_layer_, dst_iter_c_, src_iter_, src_iter_c_,
                diff_src_layer_, diff_src_iter_, diff_src_iter_c_,
                diff_dst_layer_, diff_dst_iter_, diff_dst_iter_c_,
                weights_peephole_, bias_, ws_grid_, scratch_cell_, dst_iter_);
    }

    // template <typename src_data_t, typename acc_data_t>
    rnn_postgemm_sig(execute_part2) {
#if DNNL_X64
        if (rnn_postgemm_part2_) {
            rnn_postgemm_part2_->execute(rnn, cell_position, ws_gates_,
                    scratch_gates_, dst_layer_, dst_iter_c_, src_iter_,
                    src_iter_c_, diff_src_layer_, diff_src_iter_,
                    diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
                    diff_dst_iter_c_, weights_peephole_, bias_, ws_grid_,
                    scratch_cell_, dst_iter_);
            unpoison(rnn, cell_position, ws_gates_, scratch_gates_, dst_layer_,
                    dst_iter_c_, src_iter_, src_iter_c_, diff_src_layer_,
                    diff_src_iter_, diff_src_iter_c_, diff_dst_layer_,
                    diff_dst_iter_, diff_dst_iter_c_, weights_peephole_, bias_,
                    ws_grid_, scratch_cell_, dst_iter_);
            return;
        }
#endif
        (this->*postgemm_part2_func)(rnn, cell_position, ws_gates_,
                scratch_gates_, dst_layer_, dst_iter_c_, src_iter_, src_iter_c_,
                diff_src_layer_, diff_src_iter_, diff_src_iter_c_,
                diff_dst_layer_, diff_dst_iter_, diff_dst_iter_c_,
                weights_peephole_, bias_, ws_grid_, scratch_cell_, dst_iter_);
    }

private:
    float (*activation_func)(float s, float alpha, float cliping);
    rnn_postgemm_sig(rnn_postgemm);
    rnn_postgemm_sig(lstm_postgemm);
    rnn_postgemm_sig(gru_part1_postgemm);
    rnn_postgemm_sig(gru_part2_postgemm);
    rnn_postgemm_sig(gru_lbr_postgemm);

    const rnn_pd_t *pd_;

    postgemm_f postgemm_func;
    postgemm_f postgemm_part2_func;

    DNNL_DISALLOW_COPY_AND_ASSIGN(rnn_postgemm_dispatcher);

#if DNNL_X64
    std::unique_ptr<x64::jit_uni_rnn_postgemm> rnn_postgemm_;
    std::unique_ptr<x64::jit_uni_rnn_postgemm> rnn_postgemm_part2_;

    void initialize_jit(const rnn_utils::rnn_conf_t &rnn) {
        using namespace dnnl::impl::cpu::x64;

        if (pd_->attr()->rnn_tparams_.test_mode_) return;

        const bool jit_fwd = pd_->is_fwd()
                && utils::one_of(src_type, data_type::f32, data_type::u8,
                        data_type::bf16);
        const bool jit_bwd = !pd_->is_fwd()
                && utils::one_of(src_type, data_type::f32, data_type::bf16);

#define CREATE_WITH_DIR(k, ker_t) \
    do { \
        if (mayiuse(avx512_core)) \
            k.reset(new ker_t<avx512_core, src_type, scratch_type>(rnn, pd_)); \
        else if (mayiuse(avx2)) \
            k.reset(new ker_t<avx2, src_type, scratch_type>(rnn, pd_)); \
        else \
            k.reset(new ker_t<sse41, src_type, scratch_type>(rnn, pd_)); \
    } while (0)
#define CREATE(k, ker_t) \
    do { \
        if (jit_fwd) CREATE_WITH_DIR(k, CONCAT2(ker_t, _fwd)); \
        if (jit_bwd) CREATE_WITH_DIR(k, CONCAT2(ker_t, _bwd)); \
    } while (0)

        if (pd_->cell_kind() == alg_kind::vanilla_lstm) {
            CREATE(rnn_postgemm_, jit_uni_lstm_cell_postgemm);
        } else if (pd_->cell_kind() == alg_kind::vanilla_rnn) {
            CREATE(rnn_postgemm_, jit_uni_rnn_cell_postgemm);
        } else if (pd_->cell_kind() == alg_kind::vanilla_gru) {
            CREATE(rnn_postgemm_, jit_uni_gru_cell_postgemm_part1);
            CREATE(rnn_postgemm_part2_, jit_uni_gru_cell_postgemm_part2);
        } else if (pd_->cell_kind() == alg_kind::lbr_gru) {
            CREATE(rnn_postgemm_, jit_uni_gru_lbr_cell_postgemm);
        } else
            assert(!"Unsupported algorithm kind");

#undef CREATE
#undef CREATE_WITH_DIR

        if (rnn_postgemm_) rnn_postgemm_->init(src_type);
        if (rnn_postgemm_part2_) rnn_postgemm_part2_->init(src_type);
    }
#endif
};

using rnn_postgemm_fwd_f32_t = rnn_postgemm_dispatcher<prop_kind::forward,
        data_type::f32, data_type::f32, data_type::f32>;
using rnn_postgemm_bwd_f32_t = rnn_postgemm_dispatcher<prop_kind::backward,
        data_type::f32, data_type::f32, data_type::f32>;

using rnn_postgemm_fwd_bf16_t = rnn_postgemm_dispatcher<prop_kind::forward,
        data_type::bf16, data_type::f32, data_type::f32>;
using rnn_postgemm_bwd_bf16_t = rnn_postgemm_dispatcher<prop_kind::backward,
        data_type::bf16, data_type::bf16, data_type::f32>;

using rnn_postgemm_fwd_u8_t = rnn_postgemm_dispatcher<prop_kind::forward,
        data_type::u8, data_type::s32, data_type::s32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
