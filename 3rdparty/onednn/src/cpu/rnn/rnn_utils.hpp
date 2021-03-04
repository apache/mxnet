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

#ifndef CPU_RNN_RNN_UTILS_HPP
#define CPU_RNN_RNN_UTILS_HPP

#include <type_traits>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/gemm_pack.hpp"

#if DNNL_X64
#include "cpu/x64/cpu_isa_traits.hpp"
#endif

#define rnn_postgemm_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, gates_t *ws_gates_, \
            scratch_t *scratch_gates_, dst_layer_t *dst_layer_, \
            float *dst_iter_c_, const src_iter_t *src_iter_, \
            const float *src_iter_c_, gemm_acc_t *diff_src_layer_, \
            gemm_acc_t *diff_src_iter_, gemm_acc_t *diff_src_iter_c_, \
            gemm_acc_t *diff_dst_layer_, gemm_acc_t *diff_dst_iter_, \
            gemm_acc_t *diff_dst_iter_c_, const float *weights_peephole_, \
            float *bias_, gates_t *ws_grid_, scratch_t *scratch_cell_, \
            dst_iter_t *dst_iter_) const

#define rnn_cell_execution_sig(f) \
    dnnl_status_t f(const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, dst_layer_t *dst_layer_, \
            float *dst_iter_c_, gemm_acc_t *diff_src_layer_, \
            gemm_acc_t *diff_src_iter_, gemm_acc_t *diff_src_iter_c_, \
            weights_t **w_layer_, weights_t **w_iter_, \
            weights_t **w_projection_, const float *weights_peephole_, \
            float **bias_, const src_layer_t *src_layer_, \
            const src_iter_t *src_iter_, const float *src_iter_c_, \
            gemm_acc_t *diff_dst_layer_, gemm_acc_t *diff_dst_iter_, \
            gemm_acc_t *diff_dst_iter_c_, gemm_acc_t *diff_w_layer_, \
            gemm_acc_t *diff_w_iter_, float *diff_weights_projection_, \
            float *diff_weights_peephole_, float *diff_bias_, \
            gates_t *ws_gates_, scratch_t *scratch_gates_, ht_t *proj_ht_, \
            gemm_acc_t *scratch_diff_ht_, gates_t *ws_grid_, \
            scratch_t *scratch_cell_, dst_iter_t *dst_iter_) const

#define rnn_grid_execution_sig(f) \
    dnnl_status_t f(const rnn_utils::rnn_conf_t &rnn, \
            weights_t **weights_layer_, weights_t **weights_iter_, \
            weights_t **weights_projection_, const float *weights_peephole_, \
            float **bias_, const src_layer_t *src_layer_, \
            const src_iter_t *src_iter_, const float *src_iter_c_, \
            dst_layer_t *dst_layer_, dst_iter_t *dst_iter_, \
            float *dst_iter_c_, src_layer_t *ws_states_layer_, \
            src_iter_t *ws_states_iter_, float *ws_states_iter_c_, \
            gemm_acc_t *ws_diff_states_layer_, \
            gemm_acc_t *ws_diff_states_iter_, \
            gemm_acc_t *ws_diff_states_iter_c_, gates_t *ws_gates_, \
            ht_t *ws_ht_, gates_t *ws_grid_, scratch_t *scratch_gates_, \
            ht_t *scratch_ht_, gemm_acc_t *scratch_diff_ht_, \
            scratch_t *scratch_cell_, gemm_acc_t *diff_weights_layer_, \
            gemm_acc_t *diff_weights_iter_, float *diff_weights_projection_, \
            float *diff_weights_peephole_, float *diff_bias_) const

#define rnn_gemm_sig(f) \
    dnnl_status_t f(const char transA, const char transB, dim_t m, dim_t n, \
            dim_t k, const float alpha, const weights_t *a_, const dim_t ldA, \
            const gemm_data_t *b_, const dim_t ldB, const float beta, \
            gemm_acc_t *c_, const dim_t ldC) const

#define rnn_bias_prepare_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, float **bias_, const float *b_, \
            float *scratch_bias_) const

#define rnn_bias_finalize_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, float *scratch_bias_, \
            const float *w_iter_comp, const float *w_layer_comp) const

#define rnn_weights_assign_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, const memory_desc_t *md, \
            int n_parts, const int *gates_per_part, weights_t **weights_, \
            const weights_t *w_) const

namespace dnnl {
namespace impl {
namespace cpu {

namespace rnn_utils {

enum execution_direction_t {
    l2r,
    r2l,
    bi_concat,
    bi_sum,
};

enum cell_position_t {
    middle_cell = 0x0,
    first_layer = 0x1,
    first_iter = 0x2,
    last_layer = 0x4,
    last_iter = 0x8,
    c_state_first_iter = 0x10,
    c_state_last_iter = 0x20
};

enum class weights_type_t {
    layer,
    iter,
    projection,
    peephole,
};

inline cell_position_t &operator|=(cell_position_t &lhs, cell_position_t rhs) {
    lhs = static_cast<cell_position_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
    return lhs;
}

inline cell_position_t operator|(cell_position_t lhs, cell_position_t rhs) {
    return static_cast<cell_position_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

enum data_type_conf_t {
    all_f32,
    all_bf16,
    u8u8u8f32,
    f32u8f32f32,
    u8u8u8u8,
    f32u8f32u8
};

struct rnn_conf_t {
    execution_direction_t exec_dir;
    data_type_conf_t dt_conf;
    int n_layer, n_iter, n_dir, n_gates, n_states;
    int mb;
    int slc, sic, dhc, dic, dlc;
    //int gates_ld, gates_nld, gates_ws_ld;

    int n_parts_weights_layer;
    int parts_weights_layer[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_layer_pack_size[DNNL_RNN_MAX_N_PARTS];

    int n_parts_weights_iter;
    int parts_weights_iter[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_iter_pack_size[DNNL_RNN_MAX_N_PARTS];

    int n_parts_weights_projection;
    int parts_weights_projection[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_projection_pack_size[DNNL_RNN_MAX_N_PARTS];

    int n_bias, n_parts_bias, parts_bias[DNNL_RNN_MAX_N_PARTS];

    /* Size of packed data in bytes */
    size_t weights_layer_comp_offset, weights_layer_pack_size;
    size_t weights_iter_comp_offset, weights_iter_pack_size;
    size_t weights_projection_comp_offset, weights_projection_pack_size;

    bool copy_bias;
    int weights_layer_ld, weights_layer_nld;
    int diff_weights_layer_ld, diff_weights_layer_nld;
    int weights_iter_ld, weights_iter_nld;
    int diff_weights_iter_ld, diff_weights_iter_nld;
    int weights_projection_ld, weights_projection_nld;
    int diff_weights_projection_ld, diff_weights_projection_nld;

    int proj_ht_ld, proj_ht_nld;

    int ws_gates_ld, ws_gates_nld;
    int ws_ht_ld, ws_ht_nld;
    int ws_states_layer_ld, ws_states_layer_nld;
    int ws_states_iter_ld, ws_states_iter_nld;
    int ws_states_iter_c_ld, ws_states_iter_c_nld;
    int ws_diff_states_layer_ld, ws_diff_states_layer_nld;
    int ws_diff_states_iter_ld, ws_diff_states_iter_nld;
    int ws_diff_states_iter_c_ld, ws_diff_states_iter_c_nld;

    int scratch_gates_ld, scratch_gates_nld;
    int scratch_ht_ld, scratch_ht_nld;
    int scratch_diff_ht_ld, scratch_diff_ht_nld;

    int src_layer_ld_, src_layer_nld_;
    int src_iter_ld_, src_iter_nld_;
    int src_iter_c_ld_, src_iter_c_nld_;
    int dst_layer_ld_, dst_layer_nld_;
    int dst_iter_ld_, dst_iter_nld_;
    int dst_iter_c_ld_, dst_iter_c_nld_;

    int weights_iter_compensation_size, weights_layer_compensation_size;
    bool is_fwd, is_training, is_lbr, is_lstm_peephole, is_lstm_projection;
    bool use_workspace;

    // Size of workspace for each tensor in bytes
    // Notes:
    // 1. For non-LSTMP ws_states_iter_size == ws_states_layer_size. The corresponding
    //    pointers should point to the same places.
    size_t ws_gates_size;
    size_t ws_ht_size;
    size_t ws_states_layer_size;
    size_t ws_states_iter_size;
    size_t ws_states_iter_c_size;
    size_t ws_diff_states_layer_size;
    size_t ws_diff_states_iter_size;
    size_t ws_diff_states_iter_c_size;
    size_t scratch_gates_size;
    size_t scratch_ht_size;
    size_t scratch_diff_ht_size;
    size_t scratch_cell_size;
    size_t ws_grid_comp_size;
    size_t ws_per_cell;
    size_t ws_bias_size;

    bool merge_gemm_iter, merge_gemm_layer, force_nocopy, use_layer_packed_gemm,
            use_iter_packed_gemm, use_projection_packed_gemm;
    int n_iter_scratch_gates;

    inline bool is_int8() const {
        return utils::one_of(
                dt_conf, u8u8u8f32, f32u8f32f32, u8u8u8u8, f32u8f32u8);
    }

    inline bool skip_src_layer_copy() const {
        // Note: this currently always returns true
        return (exec_dir == l2r)
                && utils::one_of(dt_conf, u8u8u8u8, u8u8u8f32, f32u8f32u8,
                        f32u8f32f32, all_f32, all_bf16);
    }
    inline bool skip_src_iter_copy() const {
        return (exec_dir == l2r) && (src_iter_ld_ > 0)
                && utils::one_of(
                        dt_conf, u8u8u8u8, u8u8u8f32, all_f32, all_bf16);
    }
    inline bool skip_dst_layer_copy() const {
        // TODO: enable skip copy with lstm_projection
        return (exec_dir == l2r) && !is_lstm_projection
                && utils::one_of(
                        dt_conf, u8u8u8u8, f32u8f32u8, all_f32, all_bf16);
    }
    inline bool skip_dst_iter_copy() const {
        // TODO: enable skip copy with lstm_projection
        return (exec_dir == l2r) && (dst_iter_ld_ > 0) && !is_lstm_projection
                && utils::one_of(
                        dt_conf, u8u8u8u8, u8u8u8f32, all_f32, all_bf16);
    }

    inline dim_t src_layer_ld(cell_position_t cell_position) const {
        return (cell_position & first_layer) && skip_src_layer_copy()
                ? src_layer_ld_
                : (cell_position & last_iter) && skip_dst_iter_copy()
                        ? dst_iter_ld_
                        : ws_states_layer_ld;
    }

    inline dim_t src_iter_ld(cell_position_t cell_position) const {
        return (cell_position & first_iter) && skip_src_iter_copy()
                ? src_iter_ld_
                : ((cell_position & last_layer) && skip_dst_layer_copy()
                                        && !(cell_position & first_iter)
                                ? dst_layer_ld_
                                : ws_states_iter_ld);
    }

    inline dim_t src_iter_c_ld(cell_position_t cell_position) const {
        return (cell_position & c_state_first_iter) ? src_iter_c_ld_
                                                    : ws_states_iter_c_ld;
    }

    inline dim_t dst_layer_ld(
            cell_position_t cell_position, bool after_proj = false) const {
        // We use scratch_ht and not dst_layer for lstmp
        if (is_lstm_projection && !after_proj) return scratch_ht_ld;

        return (cell_position & last_layer) && skip_dst_layer_copy()
                ? dst_layer_ld_
                : (cell_position & last_iter) && skip_dst_iter_copy()
                        ? dst_iter_ld_
                        : ws_states_layer_ld;
    }

    inline dim_t dst_iter_ld(cell_position_t cell_position) const {
        return (cell_position & last_iter) && skip_dst_iter_copy()
                ? dst_iter_ld_
                : ws_states_iter_ld;
    }

    inline dim_t dst_iter_c_ld(cell_position_t cell_position) const {
        return (cell_position & c_state_last_iter) ? dst_iter_c_ld_
                                                   : ws_states_iter_c_ld;
    }

    // // when skipping copy, the output ld can be states_ws_ld,
    // // dst_iter_ld or dst_layer_ld depending on the cell position
    // inline dim_t dst_ld(cell_position_t cell_position) const {
    //     return (cell_position & last_layer) ? dst_layer_ld(cell_position)
    //                                         : dst_iter_ld(cell_position);
    // }
    inline dim_t dst_copy_ld(cell_position_t cell_position) const {
        return dst_iter_ld(cell_position);
    }

    inline bool need_gemm_layer(cell_position_t cell_position) const {
        // In case of merge_gemm_layer we might still need a layer gemm if we store
        // the states of the last iteration in the destination memory. The
        // exception of this rule is the first layer though, in which case all
        // states are kept in user's src_layer, hence making full merged gemm
        // possible.
        return IMPLICATION(merge_gemm_layer,
                skip_dst_iter_copy() && (cell_position & last_iter)
                        && !(cell_position & first_layer));
    }
};

bool is_ldigo(const memory_desc_wrapper &md);
bool is_ldgoi(const memory_desc_wrapper &md);
bool is_ldio(const memory_desc_wrapper &md);
bool is_ldoi(const memory_desc_wrapper &md);

int get_good_ld(int dim, int sizeof_dt);

template <typename T>
bool init_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &src_iter_c_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &weights_projection_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &dst_iter_c_d) {
    rnn.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    rnn.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    rnn.is_lbr = rd.cell_kind == dnnl_lbr_gru;
    rnn.is_lstm_peephole = rd.cell_kind == dnnl_vanilla_lstm
            && !memory_desc_wrapper(rd.weights_peephole_desc).is_zero();
    rnn.is_lstm_projection = rd.cell_kind == dnnl_vanilla_lstm
            && !memory_desc_wrapper(rd.weights_projection_desc).is_zero();

    switch (rd.direction) {
        case dnnl_unidirectional_left2right: rnn.exec_dir = l2r; break;
        case dnnl_unidirectional_right2left: rnn.exec_dir = r2l; break;
        case dnnl_bidirectional_concat: rnn.exec_dir = bi_concat; break;
        case dnnl_bidirectional_sum: rnn.exec_dir = bi_sum; break;
        default: break;
    }

    if (utils::everyone_is(data_type::f32, src_layer_d.data_type(),
                dst_layer_d.data_type(), weights_layer_d.data_type()))
        rnn.dt_conf = all_f32;
    else if (utils::everyone_is(data_type::bf16, src_layer_d.data_type(),
                     dst_layer_d.data_type(), weights_layer_d.data_type())) {
        if (!platform::has_data_type_support(data_type::bf16)) return false;
        rnn.dt_conf = all_bf16;
    } else if (dst_layer_d.data_type() == data_type::u8) {
        if (IMPLICATION(
                    src_iter_d.md_, src_iter_d.data_type() == data_type::u8))
            rnn.dt_conf = u8u8u8u8;
        else
            rnn.dt_conf = f32u8f32u8;
    } else {
        if (IMPLICATION(
                    src_iter_d.md_, src_iter_d.data_type() == data_type::u8))
            rnn.dt_conf = u8u8u8f32;
        else
            rnn.dt_conf = f32u8f32f32;
    }

    // Set problem members defining problem sizes
    rnn.n_layer = weights_layer_d.dims()[0];
    rnn.n_iter = src_layer_d.dims()[0];
    rnn.n_dir = weights_layer_d.dims()[1];
    rnn.n_gates = weights_layer_d.dims()[3];
    rnn.n_states = rd.cell_kind == dnnl_vanilla_lstm ? 2 : 1;
    rnn.n_bias = rnn.n_gates + rnn.is_lbr;
    rnn.mb = src_layer_d.dims()[1];
    rnn.sic = weights_iter_d.dims()[2];
    rnn.slc = weights_layer_d.dims()[2];
    rnn.dhc = weights_layer_d.dims()[4];
    rnn.dlc = rnn.is_lstm_projection ? weights_projection_d.dims()[3] : rnn.dhc;
    // All supported cells have dic == dlc
    rnn.dic = rnn.dlc;

    // set members with user memories leading dimensions
    // Assumption: weights datatype size is the same as state datatype size
    assert(types::data_type_size(weights_layer_d.data_type())
            == types::data_type_size(src_layer_d.data_type()));

    // set workspace leading dimensions (and non leading-dimensions)

    // the ws and scratch proj_ht need to match as we use them interchangeably
    assert(IMPLICATION(rnn.is_lstm_projection,
            sizeof(typename T::ht_t) == sizeof(typename T::dst_iter_t)));
    rnn.proj_ht_nld = rnn.mb;
    rnn.proj_ht_ld = get_good_ld(rnn.dhc, sizeof(typename T::ht_t));

    rnn.ws_gates_nld = rnn.mb;
    rnn.ws_gates_ld
            = get_good_ld(rnn.dhc * rnn.n_gates, sizeof(typename T::gates_t));
    rnn.ws_ht_nld = rnn.proj_ht_nld;
    rnn.ws_ht_ld = rnn.proj_ht_ld;

    rnn.ws_states_layer_nld = rnn.mb;
    static_assert(std::is_same<typename T::src_layer_t,
                          typename T::src_iter_t>::value,
            "src_layer_t and src_iter_t must be the same");
    rnn.ws_states_layer_ld
            = get_good_ld(nstl::max(rnn.sic, nstl::max(rnn.slc, rnn.dlc)),
                    sizeof(typename T::src_layer_t));
    // there is no need for al separate ws_states_iter for now as all
    // supported cell have dst_iter == dst_layer
    rnn.ws_states_iter_nld = rnn.ws_states_layer_nld;
    rnn.ws_states_iter_ld = rnn.ws_states_layer_ld;

    // we do not need a good ld for iter_c as it is not involved in GEMM
    rnn.ws_states_iter_c_nld = rnn.mb;
    rnn.ws_states_iter_c_ld = rnn.dhc;

    // TODO: be more restrictive on the leading dimensions
    rnn.ws_diff_states_layer_nld = rnn.mb;
    rnn.ws_diff_states_layer_ld = get_good_ld(
            nstl::max(nstl::max(rnn.slc, rnn.dic), nstl::max(rnn.sic, rnn.dhc)),
            sizeof(typename T::gemm_acc_t));

    rnn.ws_diff_states_iter_nld = rnn.mb;
    rnn.ws_diff_states_iter_ld = get_good_ld(
            nstl::max(nstl::max(rnn.slc, rnn.dic), nstl::max(rnn.sic, rnn.dhc)),
            sizeof(typename T::gemm_acc_t));

    rnn.ws_diff_states_iter_c_nld = rnn.mb;
    rnn.ws_diff_states_iter_c_ld = rnn.dhc;

    // set scratch (not)leading dimensions
    rnn.scratch_gates_nld = rnn.mb;
    rnn.scratch_gates_ld
            = get_good_ld(rnn.n_gates * rnn.dhc, sizeof(typename T::scratch_t));
    rnn.scratch_ht_nld = rnn.proj_ht_nld;
    rnn.scratch_ht_ld = rnn.proj_ht_ld;

    rnn.scratch_diff_ht_nld = rnn.mb;
    rnn.scratch_diff_ht_ld
            = get_good_ld(rnn.dlc, sizeof(typename T::gemm_acc_t));

    // Assumption: {src,dst}_layer has tnc layout, {src,dst}_iter has ldnc,
    rnn.src_layer_ld_ = src_layer_d.blocking_desc().strides[1];
    rnn.dst_layer_ld_ = dst_layer_d.blocking_desc().strides[1];
    rnn.src_iter_ld_ = types::is_zero_md(src_iter_d.md_)
            ? 0
            : src_iter_d.blocking_desc().strides[2];
    rnn.dst_iter_ld_ = types::is_zero_md(dst_iter_d.md_)
            ? 0
            : dst_iter_d.blocking_desc().strides[2];
    rnn.src_iter_c_ld_ = types::is_zero_md(src_iter_c_d.md_)
            ? 0
            : src_iter_c_d.blocking_desc().strides[2];
    rnn.dst_iter_c_ld_ = types::is_zero_md(dst_iter_c_d.md_)
            ? 0
            : dst_iter_c_d.blocking_desc().strides[2];

    /* Set the correct number of weights parts */
    bool is_orig_gru = rd.cell_kind == alg_kind::vanilla_gru;
    rnn.n_parts_weights_layer = 1;
    rnn.parts_weights_layer[0] = rnn.n_gates;
    rnn.parts_weights_layer[1] = 0;

    rnn.n_parts_weights_iter = is_orig_gru ? 2 : 1;
    rnn.parts_weights_iter[0] = is_orig_gru ? 2 : rnn.n_gates;
    rnn.parts_weights_iter[1] = is_orig_gru ? 1 : 0;

    rnn.n_parts_weights_projection = 1;
    rnn.parts_weights_projection[0] = 1;

    rnn.n_parts_bias = 1;
    rnn.parts_bias[0] = rnn.n_bias;
    rnn.parts_bias[1] = 0;

    /* Decide wich gemm implementation to use: packed/nonpacked jit/cblas
     * and if to mergre gemm across iterations */
    bool is_f32 = rnn.dt_conf == all_f32, is_bf16 = rnn.dt_conf == all_bf16;
    bool is_gru = utils::one_of(
            rd.cell_kind, alg_kind::vanilla_gru, alg_kind::lbr_gru);
    bool is_inference = !rnn.is_training;

    // To be able to merge the GEMM on the layer input when not
    // copying, we need to have a trivial stride for the T dimension
    auto src_layer_is_trivial_stride = src_layer_d.blocking_desc().strides[0]
            == (rnn.src_layer_ld_ * rnn.mb);
    auto dst_layer_is_trivial_stride = dst_layer_d.blocking_desc().strides[0]
            == (rnn.dst_layer_ld_ * rnn.mb);

    rnn.merge_gemm_layer = ((rnn.is_fwd && src_layer_is_trivial_stride)
                                   || ((rd.prop_kind == prop_kind::backward)
                                           && dst_layer_is_trivial_stride))
            && (((rnn.is_fwd && rnn.mb < 128) || !rnn.is_fwd) || rnn.is_int8());
    rnn.merge_gemm_iter
            = dst_layer_is_trivial_stride && !(rnn.is_fwd || is_gru);
    rnn.force_nocopy = false;
#if DNNL_X64
    rnn.force_nocopy = !x64::mayiuse(x64::avx512_mic) && x64::mayiuse(x64::avx)
            && ((is_inference && (rnn.n_layer > 1 || rnn.mb < 100))
                    || (rnn.is_training && rnn.dhc < 500));
#endif

    /* Decide to copy bias */
    rnn.copy_bias = rnn.is_int8();

    rnn.use_layer_packed_gemm
            = utils::one_of(weights_layer_d.format_kind(), format_kind::any,
                      format_kind::rnn_packed)
            && is_inference
            && ((is_f32 && pack_sgemm_supported() && rnn.n_iter == 1)
                    || rnn.is_int8() || is_bf16);
    rnn.use_iter_packed_gemm
            = utils::one_of(weights_iter_d.format_kind(), format_kind::any,
                      format_kind::rnn_packed)
            && is_inference
            && ((is_f32 && pack_sgemm_supported() && rnn.mb >= 16)
                    || rnn.is_int8() || is_bf16);
    rnn.use_projection_packed_gemm = false;

    /* Set packed gemm sizes */
    /* TODO: investigate the benefit of mixing packed and non-packed weights parts */
    auto set_pack_sizes
            = [&](bool merge, bool &do_pack, size_t &weights_pack_size,
                      int &n_parts, int *parts, size_t *parts_pack_size,
                      size_t &comp_offset, int feature_size, dim_t weights_oc,
                      dim_t data_ld) -> bool {
        bool pack = true;
        weights_pack_size = 0;
        for (int p = 0; p < n_parts; p++) {
            dim_t m_p = rnn.is_fwd ? (parts[p] * rnn.dhc) : feature_size;
            dim_t k_p = rnn.is_fwd ? feature_size : (parts[p] * rnn.dhc);
            dim_t n_p = merge ? rnn.mb * rnn.n_iter : rnn.mb;
            bool pack_part = true;

            dnnl_status_t st = dnnl_success;
            switch (rnn.dt_conf) {
                case all_f32:
                    st = sgemm_pack_get_size("A", "N", "N", &m_p, &n_p, &k_p,
                            &m_p, &data_ld, &parts_pack_size[p], &pack_part);
                    break;
                case u8u8u8f32:
                case f32u8f32f32:
                case u8u8u8u8:
                case f32u8f32u8:
                    st = gemm_s8u8s32_pack_get_size("A", "N", "N", &m_p, &n_p,
                            &k_p, &m_p, &data_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                case all_bf16:
                    st = gemm_bf16bf16f32_pack_get_size("A", "N", "N", &m_p,
                            &n_p, &k_p, &m_p, &data_ld, &parts_pack_size[p],
                            &pack_part);
                    break;
                default: assert(!"Unsupported configuration");
            }
            if (st != dnnl_success) return false;

            pack = pack && pack_part;
            weights_pack_size += rnn.n_layer * rnn.n_dir * parts_pack_size[p];
        }

        // NOTE: pack is updated only for f32. We force pack for int8
        do_pack = (rnn.dt_conf == all_f32) ? pack : true;
        comp_offset = weights_pack_size;
        const bool need_compensation = rnn.is_int8();
        weights_pack_size += (need_compensation ? rnn.n_layer * rnn.n_dir : 0)
                * weights_oc * sizeof(float);

        return true;
    };
    // TODO: the activation leading dimension can vary for first layer/iteration
    if (rnn.use_layer_packed_gemm) {
        bool ok = set_pack_sizes(rnn.merge_gemm_layer,
                rnn.use_layer_packed_gemm, rnn.weights_layer_pack_size,
                rnn.n_parts_weights_layer, rnn.parts_weights_layer,
                rnn.part_weights_layer_pack_size, rnn.weights_layer_comp_offset,
                rnn.slc, rnn.n_gates * rnn.dhc, rnn.ws_states_layer_ld);
        if (!ok) return false;
    }

    if (rnn.use_iter_packed_gemm) {
        bool ok = set_pack_sizes(rnn.merge_gemm_iter, rnn.use_iter_packed_gemm,
                rnn.weights_iter_pack_size, rnn.n_parts_weights_iter,
                rnn.parts_weights_iter, rnn.part_weights_iter_pack_size,
                rnn.weights_iter_comp_offset, rnn.sic, rnn.n_gates * rnn.dhc,
                rnn.ws_states_iter_ld);
        if (!ok) return false;
    }

    return true;
}

template <typename T>
void set_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &weights_projection_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d,
        const memory_desc_wrapper &diff_weights_projection_d) {

    // Set leading dimensions for input weights arrays depending on input format
    auto set_dims = [&](const memory_desc_wrapper &md, int &ld, int &nld) {
        ld = 0;
        nld = 0;
        if (md.is_blocking_desc()) {
            if (is_ldigo(md)) {
                ld = (int)md.blocking_desc().strides[2];
                nld = md.dims()[2];
            } else if (is_ldgoi(md)) {
                ld = (int)md.blocking_desc().strides[4];
                nld = md.dims()[3] * md.dims()[4];
            } else if (is_ldoi(md)) {
                ld = (int)md.blocking_desc().strides[3];
                nld = md.dims()[3];
            } else if (is_ldio(md)) {
                ld = (int)md.blocking_desc().strides[2];
                nld = md.dims()[2];
            } else
                assert(!"unsupported weights format");
        }
    };
    set_dims(weights_layer_d, rnn.weights_layer_ld, rnn.weights_layer_nld);
    set_dims(weights_iter_d, rnn.weights_iter_ld, rnn.weights_iter_nld);
    set_dims(weights_projection_d, rnn.weights_projection_ld,
            rnn.weights_projection_nld);
    if (!rnn.is_fwd) {
        set_dims(diff_weights_layer_d, rnn.diff_weights_layer_ld,
                rnn.diff_weights_layer_nld);
        set_dims(diff_weights_iter_d, rnn.diff_weights_iter_ld,
                rnn.diff_weights_iter_nld);
        set_dims(diff_weights_projection_d, rnn.diff_weights_projection_ld,
                rnn.diff_weights_projection_nld);
    }

    assert(weights_layer_d.data_type() == weights_iter_d.data_type());
    assert(IMPLICATION(diff_weights_layer_d.ndims() != 0,
            (diff_weights_layer_d.data_type()
                    == diff_weights_iter_d.data_type())));

    /* Set workspace sizes to store:
     * states to compute a pass
     * diff states to compute bwd pass (training onl)y
     * intermediate results from the gates
     */

    assert(sizeof(typename T::src_layer_t) == sizeof(typename T::dst_layer_t));
    assert(sizeof(typename T::src_iter_t) == sizeof(typename T::dst_iter_t));

    rnn.use_workspace = rnn.is_training;
    // TODO: for inference, we can make ws_states_* smaller, but
    // dependant of the grid execution though
    rnn.ws_states_layer_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.mb * rnn.ws_states_layer_ld
            * sizeof(typename T::src_layer_t);
    rnn.ws_states_iter_size = (size_t)(rnn.n_layer + 1) * rnn.n_dir
            * (rnn.n_iter + 1) * rnn.mb * rnn.ws_states_iter_ld
            * sizeof(typename T::src_iter_t);
    bool is_lstm = rd.cell_kind == dnnl_vanilla_lstm;
    rnn.ws_states_iter_c_size = is_lstm
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_states_iter_c_ld * sizeof(float)
            : 0;

    rnn.ws_diff_states_layer_size = rnn.is_training
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_diff_states_layer_ld
                    * sizeof(typename T::gemm_acc_t)
            : (size_t)0;
    rnn.ws_diff_states_iter_size = rnn.is_training
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_diff_states_iter_ld
                    * sizeof(typename T::gemm_acc_t)
            : (size_t)0;
    rnn.ws_diff_states_iter_c_size = rnn.is_training && is_lstm
            ? (size_t)(rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1) * rnn.mb
                    * rnn.ws_diff_states_iter_c_ld
                    * sizeof(typename T::gemm_acc_t)
            : (size_t)0;

    rnn.ws_gates_size = rnn.is_training
            ? (size_t)rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.ws_gates_nld
                    * rnn.ws_gates_ld * sizeof(typename T::gates_t)
            : (size_t)0;
    rnn.ws_ht_size = rnn.is_training
            ? (size_t)rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.ws_ht_nld
                    * rnn.ws_ht_ld * sizeof(typename T::dst_iter_t)
            : (size_t)0;
    rnn.n_iter_scratch_gates
            = (rnn.merge_gemm_layer || rnn.merge_gemm_iter) ? rnn.n_iter : 1;
    rnn.scratch_gates_size = rnn.n_iter_scratch_gates * rnn.scratch_gates_nld
            * rnn.scratch_gates_ld * sizeof(typename T::scratch_t);
    rnn.scratch_ht_size
            = rnn.scratch_ht_nld * rnn.scratch_ht_ld * sizeof(typename T::ht_t);
    rnn.scratch_diff_ht_size = rnn.is_training ? rnn.scratch_diff_ht_nld
                    * rnn.scratch_diff_ht_ld * sizeof(typename T::gemm_acc_t)
                                               : (size_t)0;

    /* set other sizes */
    /// scratchpad buffer for each cell to hold intermediate data in gru/lbr_gru
    rnn.scratch_cell_size = rnn.is_lbr
            ? (size_t)rnn.scratch_gates_nld * rnn.scratch_gates_ld
                    * sizeof(typename T::gemm_acc_t)
            : (rd.cell_kind == alg_kind::vanilla_gru
                            ? (size_t)rnn.ws_states_layer_nld
                                    * rnn.ws_states_layer_ld
                                    * sizeof(typename T::gemm_acc_t)
                            : 0);
    /// workspace needed for lbr GRU
    rnn.ws_per_cell = (size_t)rnn.is_lbr * rnn.mb * rnn.dhc
            * sizeof(typename T::gemm_acc_t);
    rnn.ws_grid_comp_size = (size_t)rnn.is_lbr * rnn.is_training * rnn.n_layer
            * rnn.n_dir * rnn.n_iter * rnn.ws_per_cell * sizeof(float);
    /// bias ws needed to add compensation in int8
    rnn.ws_bias_size = (size_t)rnn.n_layer * rnn.n_dir * rnn.n_bias * rnn.dhc
            * sizeof(float);
}

void set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_ht_offset, size_t &ws_state_layer_offset,
        size_t &ws_states_iter_offset, size_t &ws_states_iter_c_offset,
        size_t &ws_diff_states_layer_offset, size_t &ws_diff_states_iter_offset,
        size_t &ws_diff_states_iter_c_offset, size_t &ws_grid_comp_offset,
        size_t &ws_bias_offset, size_t &scratch_gates_offset,
        size_t &scratch_ht_offset, size_t &scratch_diff_ht_offset,
        size_t &scratch_cell_offset, size_t &scratchpad_size,
        size_t &workspace_size);

void get_scratchpad_and_workspace_sizes(
        const rnn_conf_t &rnn, size_t &scratchpad_size, size_t &workspace_size);
status_t set_expected_desc(rnn_conf_t &rnn, memory_desc_t &weights_md,
        weights_type_t weights_type);
status_t set_good_strides(memory_desc_t &weights_md, format_tag_t tag);

template <typename T>
struct ws_gates_aoc {
    ws_gates_aoc(const rnn_conf_t &rnn, T *data)
        : gates_(data, rnn.ws_gates_nld, rnn.ws_gates_ld), DHC_(rnn.dhc) {}
    T &operator()(int batch, int gate, int dhc) {
        return gates_(batch, gate * DHC_ + dhc);
    }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> gates_;
    int DHC_;
};
using ws_gates_aoc_t = ws_gates_aoc<float>;
using ws_gates_aoc_s32_t = ws_gates_aoc<int32_t>;

template <typename T>
struct ws_ht_aoc {
    ws_ht_aoc(const rnn_conf_t &rnn, T *data)
        : ht_(data, rnn.ws_ht_nld, rnn.ws_ht_ld) {}
    T &operator()(int batch, int dhc) { return ht_(batch, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> ht_;
};

template <typename T>
struct scratch_gates_aoc {
    scratch_gates_aoc(const rnn_conf_t &rnn, T *data)
        : gates_(data, rnn.scratch_gates_nld, rnn.scratch_gates_ld)
        , DHC_(rnn.dhc) {}
    T &operator()(int batch, int gate, int dhc) {
        return gates_(batch, gate * DHC_ + dhc);
    }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> gates_;
    int DHC_;
};
using scratch_gates_aoc_t = scratch_gates_aoc<float>;
using scratch_gates_aoc_s32_t = scratch_gates_aoc<int32_t>;

template <typename T>
struct scratch_ht_aoc {
    scratch_ht_aoc(const rnn_conf_t &rnn, T *data)
        : ht_(data, rnn.scratch_ht_nld, rnn.scratch_ht_ld) {}
    T &operator()(int batch, int dhc) { return ht_(batch, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> ht_;
};
using scratch_ht_aoc_t = scratch_ht_aoc<float>;
using scratch_ht_aoc_s32_t = scratch_ht_aoc<int32_t>;

template <typename T>
struct weights_peephole_aoc_t {
    weights_peephole_aoc_t(const rnn_conf_t &rnn, T *data)
        : weights_peephole_(data, 3, rnn.dhc) {}
    T &operator()(int g, int dhc) { return weights_peephole_(g, dhc); }

private:
    utils::array_offset_calculator<T, 2> weights_peephole_;
};

struct bias_aoc_t {
    bias_aoc_t(const rnn_conf_t &rnn, const float *data)
        : bias_(data, rnn.n_bias, rnn.dhc) {}
    const float &operator()(int bias_n, int dhc) { return bias_(bias_n, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<const float, 2> bias_;
};

template <typename T>
struct ws_states_layer_aoc {
    ws_states_layer_aoc(const rnn_conf_t &rnn, T *data, int leading_dim)
        : state_(data, rnn.ws_states_layer_nld, leading_dim) {}
    ws_states_layer_aoc(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.ws_states_layer_nld, rnn.ws_states_layer_ld) {}
    T &operator()(int batch, int dhc) { return state_(batch, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> state_;
};

template <typename T>
struct ws_states_iter_aoc {
    ws_states_iter_aoc(const rnn_conf_t &rnn, T *data, int leading_dim)
        : state_(data, rnn.ws_states_iter_nld, leading_dim) {}
    ws_states_iter_aoc(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.ws_states_iter_nld, rnn.ws_states_iter_ld) {}
    T &operator()(int batch, int dhc) { return state_(batch, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> state_;
};

template <typename T>
struct ws_states_iter_c_aoc {
    ws_states_iter_c_aoc(const rnn_conf_t &rnn, T *data, int leading_dim)
        : state_(data, rnn.ws_states_iter_c_nld, leading_dim) {}
    ws_states_iter_c_aoc(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.ws_states_iter_c_nld, rnn.ws_states_iter_c_ld) {}
    T &operator()(int batch, int dhc) { return state_(batch, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> state_;
};

template <typename T>
struct ws_diff_states_layer_aoc {
    ws_diff_states_layer_aoc(const rnn_conf_t &rnn, T *data)
        : diff_states_layer_(data, rnn.ws_diff_states_layer_nld,
                rnn.ws_diff_states_layer_ld) {}
    T &operator()(int batch, int dhc) { return diff_states_layer_(batch, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> diff_states_layer_;
};

template <typename T>
struct ws_diff_states_iter_aoc {
    ws_diff_states_iter_aoc(const rnn_conf_t &rnn, T *data)
        : diff_states_iter_(data, rnn.ws_diff_states_iter_nld,
                rnn.ws_diff_states_iter_ld) {}
    T &operator()(int batch, int dhc) { return diff_states_iter_(batch, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> diff_states_iter_;
};

template <typename T>
struct ws_diff_states_iter_c_aoc {
    ws_diff_states_iter_c_aoc(const rnn_conf_t &rnn, T *data)
        : diff_states_iter_c_(data, rnn.ws_diff_states_iter_c_nld,
                rnn.ws_diff_states_iter_c_ld) {}
    T &operator()(int batch, int dhc) {
        return diff_states_iter_c_(batch, dhc);
    }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> diff_states_iter_c_;
};

struct ws_diff_w_iter_aoc_t {
    ws_diff_w_iter_aoc_t(const rnn_conf_t &rnn, float *data)
        : diff_weights_iter_(
                data, rnn.diff_weights_iter_nld, rnn.diff_weights_iter_ld)
        , DHC_(rnn.dhc) {}
    float &operator()(int sic, int gate, int dhc) {
        return diff_weights_iter_(sic, gate * DHC_ + dhc);
    }

private:
    dnnl::impl::utils::array_offset_calculator<float, 2> diff_weights_iter_;
    int DHC_;
};

} // namespace rnn_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
