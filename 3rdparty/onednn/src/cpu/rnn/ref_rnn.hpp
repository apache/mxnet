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

#ifndef CPU_RNN_REF_RNN_HPP
#define CPU_RNN_REF_RNN_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm/os_blas.hpp"

#include "cpu/rnn/cpu_rnn_pd.hpp"
#include "cpu/rnn/postgemm_dispatcher.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
template <typename gates_t, typename acc_t>
// The loop body needs to be put in a function as some versions of icc have
// an issue with lambdas & macros inside omp simd loops
inline void body_loop(int i, int k, const gates_t *ws_gates, acc_t *diff_bias,
        const rnn_utils::rnn_conf_t &rnn) {
    for (int j = 0; j < rnn.mb; j++)
        diff_bias[i * rnn.dhc + k]
                += ws_gates[j * rnn.scratch_gates_ld + i * rnn.dhc + k];
}
} // namespace

template <typename gates_t, typename acc_t>
void gates_reduction(const rnn_utils::rnn_conf_t &rnn, const gates_t *ws_gates_,
        acc_t *diff_bias_) {

    // @todo block k on simd-width to enable vectorization in
    // parallel_nd path
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP && _OPENMP >= 201307 \
        && __INTEL_COMPILER != 1910
#pragma omp parallel for simd collapse(2)
    for (int i = 0; i < rnn.n_gates; i++)
        for (int k = 0; k < rnn.dhc; k++)
            body_loop(i, k, ws_gates_, diff_bias_, rnn);
#else
    parallel_nd(rnn.n_gates, rnn.dhc,
            [&](int i, int k) { body_loop(i, k, ws_gates_, diff_bias_, rnn); });
#endif
}

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
struct _ref_rnn_common_t : public primitive_t {
    static constexpr impl::data_type_t scratch_type
            = aprop == prop_kind::forward ? acc_type : src_type;

    /* These types are defined for each element in the cell execution */
    typedef typename prec_traits<src_type>::type src_layer_t;
    typedef typename prec_traits<src_type>::type src_iter_t;
    typedef typename prec_traits<src_type>::type dst_layer_t;
    typedef typename prec_traits<src_type>::type dst_iter_t;
    typedef typename prec_traits<weights_type>::type weights_t;
    typedef typename prec_traits<src_type>::type gemm_data_t;
    typedef typename prec_traits<acc_type>::type gemm_acc_t;
    typedef typename prec_traits<scratch_type>::type scratch_t;
    typedef typename prec_traits<src_type>::type ht_t;
    typedef typename prec_traits<src_type>::type gates_t;

    using class_name
            = _ref_rnn_common_t<aprop, src_type, weights_type, acc_type>;

    typedef rnn_cell_execution_sig((class_name::*cell_execution_f));
    typedef rnn_grid_execution_sig((class_name::*grid_execution_f));

    typedef rnn_gemm_sig((class_name::*gemm_t));
    typedef rnn_bias_prepare_sig((class_name::*bias_prepare_t));
    typedef rnn_bias_finalize_sig((class_name::*bias_finalize_t));
    typedef rnn_weights_assign_sig((class_name::*weights_assign_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    cpu_rnn_fwd_pd_t, cpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {
        using base_pd_t::base_pd_t;

        DECLARE_COMMON_PD_T("ref:any", class_name, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            using namespace format_tag;
            using namespace rnn_utils;
            const alg_kind_t cell_kind = this->desc()->cell_kind;

            data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
            data_type_t weights_iter_dt
                    = this->desc()->weights_iter_desc.data_type;
            data_type_t weights_layer_dt
                    = this->desc()->weights_layer_desc.data_type;

            bool ok = true
                    && one_of(cell_kind, alg_kind::vanilla_rnn,
                            alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                            alg_kind::lbr_gru)
                    && IMPLICATION(aprop == prop_kind::forward,
                            one_of(this->desc()->prop_kind, forward_training,
                                    forward_inference))
                    && IMPLICATION(aprop == backward,
                            one_of(this->desc()->prop_kind, backward))
                    && src_layer_dt == src_type
                    && everyone_is(
                            weights_type, weights_iter_dt, weights_layer_dt)
                    && this->set_default_params() == status::success
                    && this->with_bias();
            if (!ok) return status::unimplemented;

            ok = init_conf<class_name>(rnn_, *this->desc(), this->src_md(0),
                    this->src_md(1), this->src_md(2), this->weights_md(0),
                    this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION), this->dst_md(0),
                    this->dst_md(1), this->dst_md(2));
            if (!ok) return status::unimplemented;

            /* check that only supported attr have been passed */
            primitive_attr_t::skip_mask_t attr_mask
                    = primitive_attr_t::skip_mask_t::rnn_tparams;
            if (weights_layer_dt == data_type::s8)
                attr_mask = attr_mask
                        | primitive_attr_t::skip_mask_t::rnn_data_qparams
                        | primitive_attr_t::skip_mask_t::rnn_weights_qparams;
            ok = ok && this->attr()->has_default_values(attr_mask);
            if (!ok) return status::unimplemented;

            // Set weights descriptors to desired format
            memory_desc_t new_weights_layer_md = *this->weights_md(0);
            CHECK(set_expected_desc(rnn_, new_weights_layer_md,
                    rnn_utils::weights_type_t::layer));
            if (this->weights_layer_md_.format_kind == format_kind::any) {
                this->weights_layer_md_ = new_weights_layer_md;
            } else if (this->weights_layer_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_layer_md_ != new_weights_layer_md)
                    return status::unimplemented;
            }

            memory_desc_t new_weights_iter_md = *this->weights_md(1);
            CHECK(set_expected_desc(rnn_, new_weights_iter_md,
                    rnn_utils::weights_type_t::iter));
            if (this->weights_iter_md_.format_kind == format_kind::any) {
                this->weights_iter_md_ = new_weights_iter_md;
            } else if (this->weights_iter_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_iter_md_ != new_weights_iter_md)
                    return status::unimplemented;
            }

            if (rnn_.is_lstm_projection) {
                memory_desc_t new_weights_projection_md
                        = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
                CHECK(set_expected_desc(rnn_, new_weights_projection_md,
                        rnn_utils::weights_type_t::projection));
                if (this->weights_projection_md_.format_kind
                        == format_kind::any) {
                    this->weights_projection_md_ = new_weights_projection_md;
                } else if (this->weights_projection_md_.format_kind
                        == format_kind::rnn_packed) {
                    if (this->weights_projection_md_
                            != new_weights_projection_md)
                        return status::unimplemented;
                }
            }

            CHECK(this->check_layout_consistency());

            set_conf<class_name>(rnn_, *this->desc(), this->weights_md(0),
                    this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                    this->diff_weights_md(0), this->diff_weights_md(1),
                    this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));

            size_t scratchpad_sz {0}, ws_sz {0};
            get_scratchpad_and_workspace_sizes(rnn_, scratchpad_sz, ws_sz);

            // initialize the workspace if needed
            if (rnn_.is_training) {
                dims_t ws_dims = {(dim_t)ws_sz};
                dnnl_memory_desc_init_by_tag(&this->ws_md_, 1, ws_dims,
                        data_type::u8, format_tag::x);
            }

            init_scratchpad(scratchpad_sz);

            return status::success;
        }

        rnn_utils::rnn_conf_t rnn_;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.template book<float>(key_rnn_space, scratchpad_sz, 4096);

            int max_nparts = this->cell_kind() == alg_kind::vanilla_gru ? 2 : 1;
            int ptr_wei_sz = rnn_.n_layer * rnn_.n_dir * max_nparts;
            scratchpad.template book<float *>(
                    key_rnn_ptrs_wei_layer, ptr_wei_sz);
            scratchpad.template book<float *>(
                    key_rnn_ptrs_wei_iter, ptr_wei_sz);
            scratchpad.template book<float *>(
                    key_rnn_ptrs_wei_projection, ptr_wei_sz);
            scratchpad.template book<float *>(key_rnn_ptrs_bia, ptr_wei_sz);
            scratchpad.template book<scratch_t>(
                    key_rnn_gates, rnn_.scratch_gates_size);
            scratchpad.template book<ht_t>(key_rnn_ht, rnn_.scratch_ht_size);
            scratchpad.template book<gemm_acc_t>(
                    key_rnn_diff_ht, rnn_.scratch_diff_ht_size);
            scratchpad.template book<scratch_t>(
                    key_rnn_cell, rnn_.scratch_cell_size);
        }
    };

    _ref_rnn_common_t(const pd_t *apd)
        : primitive_t(apd), rnn_postgemm_(nullptr) {}

    status_t init(engine_t *engine) override {
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dhc and sic != dhc
        /// respectively

        bias_preparation_func = &class_name::bias_prepare;
        bias_finalization_func = &class_name::bias_finalize;

        auto set_gemm_funcs
                = [](bool packed_gemm, gemm_t &g, weights_assign_t &a) {
                      if (packed_gemm) {
                          g = &class_name::packed_gemm;
                          a = &class_name::assign_packed_weights;
                      } else {
                          g = &class_name::gemm;
                          a = &class_name::assign_weights;
                      }
                  };
        set_gemm_funcs(pd()->rnn_.use_iter_packed_gemm, gemm_iter_func,
                weights_iter_assign_func);

        set_gemm_funcs(pd()->rnn_.use_layer_packed_gemm, gemm_layer_func,
                weights_layer_assign_func);

        if (pd()->rnn_.is_lstm_projection) {
            set_gemm_funcs(pd()->rnn_.use_projection_packed_gemm,
                    gemm_projection_func, weights_projection_assign_func);
        }

        rnn_postgemm_ = new rnn_postgemm_dispatcher<aprop, src_type,
                scratch_type, acc_type>(pd()->rnn_, pd());
        assert(rnn_postgemm_ != nullptr);
        switch (pd()->cell_kind()) {
            case alg_kind::vanilla_rnn:
            case alg_kind::vanilla_lstm:
                cell_func = &class_name::cell_execution;
                break;
            case alg_kind::vanilla_gru:
                cell_func = &class_name::cell_execution_gru;
                break;
            case alg_kind::lbr_gru:
                cell_func = &class_name::cell_execution_gru_lbr;
                break;
            default: break;
        }

        grid_computation = &class_name::linear_execution;

        size_t scratchpad_size, workspace_size;
        rnn_utils::set_offsets(pd()->rnn_, ws_gates_offset_, ws_ht_offset_,
                ws_states_layer_offset_, ws_states_iter_offset_,
                ws_states_iter_c_offset_, ws_diff_states_layer_offset_,
                ws_diff_states_iter_offset_, ws_diff_states_iter_c_offset_,
                ws_grid_comp_offset_, ws_bias_offset_, scratch_gates_offset_,
                scratch_ht_offset_, scratch_diff_ht_offset_,
                scratch_cell_offset_, scratchpad_size, workspace_size);

        return status::success;
    }

    ~_ref_rnn_common_t() { delete rnn_postgemm_; }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_(ctx);
        return status::success;
    }

private:
    void execute_(const exec_ctx_t &ctx) const;
    rnn_grid_execution_sig(linear_execution);
    rnn_cell_execution_sig(cell_execution);
    rnn_cell_execution_sig(cell_execution_gru);
    rnn_cell_execution_sig(cell_execution_gru_lbr);
    rnn_gemm_sig(gemm);
    rnn_gemm_sig(packed_gemm);
    rnn_bias_prepare_sig(bias_prepare);
    rnn_bias_finalize_sig(bias_finalize);
    rnn_weights_assign_sig(assign_weights);
    rnn_weights_assign_sig(assign_packed_weights);

    float (*activation_func)(float s, float alpha, float cliping);

    void copy_init_layer(const rnn_utils::rnn_conf_t &rnn,
            src_layer_t *ws_states_layer_, gemm_acc_t *ws_diff_states_layer_,
            const src_layer_t *xt_, const gemm_acc_t *diff_dst_layer) const;

    template <typename input_t>
    void copy_init_iter(const rnn_utils::rnn_conf_t &rnn,
            src_iter_t *ws_states_iter_, float *ws_states_iter_c_,
            gemm_acc_t *ws_diff_states_iter_,
            gemm_acc_t *ws_diff_states_iter_c_, const input_t *src_iter_,
            const float *src_iter_c_, const gemm_acc_t *diff_dst_iter_,
            const float *diff_dst_iter_c_) const;

    template <typename dst_layer_dt, typename dst_iter_dt>
    void copy_res_layer(const rnn_utils::rnn_conf_t &rnn,
            dst_layer_dt *dst_layer_, gemm_acc_t *diff_src_layer_,
            const dst_iter_dt *dst_iter_, const src_layer_t *ws_states_layer_,
            const gemm_acc_t *ws_diff_states_layer_) const;

    template <typename prim_dst_iter_t, typename prim_dst_layer_t>
    void copy_res_iter(const rnn_utils::rnn_conf_t &rnn,
            prim_dst_iter_t *dst_iter_, float *dst_iter_c_,
            gemm_acc_t *diff_src_iter_, float *diff_src_iter_c_,
            const prim_dst_layer_t *dst_layer_,
            const src_iter_t *ws_states_iter_, const float *ws_states_iter_c,
            const gemm_acc_t *ws_diff_states_iter_,
            const gemm_acc_t *ws_diff_states_iter_c_) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    size_t ws_gates_offset_;
    size_t ws_ht_offset_;
    size_t ws_states_layer_offset_;
    size_t ws_states_iter_offset_;
    size_t ws_states_iter_c_offset_;
    size_t ws_bias_offset_;
    size_t ws_diff_states_layer_offset_;
    size_t ws_diff_states_iter_offset_;
    size_t ws_diff_states_iter_c_offset_;
    size_t ws_grid_comp_offset_;
    size_t scratch_gates_offset_;
    size_t scratch_ht_offset_;
    size_t scratch_diff_ht_offset_;
    size_t scratch_cell_offset_;
    rnn_postgemm_dispatcher<aprop, src_type, scratch_type, acc_type>
            *rnn_postgemm_;

    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    bias_prepare_t bias_preparation_func;
    bias_finalize_t bias_finalization_func;
    weights_assign_t weights_layer_assign_func;
    weights_assign_t weights_iter_assign_func;
    weights_assign_t weights_projection_assign_func;

    gemm_t gemm_layer_func;
    gemm_t gemm_iter_func;
    gemm_t gemm_projection_func;
};

using ref_rnn_fwd_f32_t = _ref_rnn_common_t<prop_kind::forward, data_type::f32,
        data_type::f32, data_type::f32>;
using ref_rnn_bwd_f32_t = _ref_rnn_common_t<prop_kind::backward, data_type::f32,
        data_type::f32, data_type::f32>;
using ref_rnn_fwd_bf16_t = _ref_rnn_common_t<prop_kind::forward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_bwd_bf16_t = _ref_rnn_common_t<prop_kind::backward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_fwd_u8s8_t = _ref_rnn_common_t<prop_kind::forward, data_type::u8,
        data_type::s8, data_type::s32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
