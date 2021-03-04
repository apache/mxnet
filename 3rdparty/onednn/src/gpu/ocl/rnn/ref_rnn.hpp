/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_OCL_RNN_REF_RNN_HPP
#define GPU_OCL_RNN_REF_RNN_HPP

#include <assert.h>
#include <stdio.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_rnn_pd.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/rnn/rnn_utils.hpp"
#include "gpu/primitive_conf.hpp"

// TODO just to debug
#define WS_NAN_FILLING 0

#define DEBUGPRINT 0

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

enum gemm_kind_t {
    gemm_iter_fwd,
    gemm_iter_fwd_2,
    gemm_layer_fwd,
    gemm_iter_bwd,
    gemm_iter_bwd_2,
    gemm_layer_bwd,
    gemm_diff_wei_iter,
    gemm_diff_wei_iter_2,
    gemm_diff_wei_layer
};

template <prop_kind_t aprop>
struct _ref_rnn_common_t : public gpu_primitive_t {

    using class_name = _ref_rnn_common_t<aprop>;

    typedef elemwise_sig((class_name::*elemwise_f));
    typedef cell_execution_sig((class_name::*cell_execution_f));
    typedef grid_execution_sig((class_name::*grid_execution_f));
    typedef gemm_sig((class_name::*gemm_t));
    typedef weights_assign_sig((class_name::*weights_assign_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    gpu_rnn_fwd_pd_t, gpu_rnn_bwd_pd_t>::type;
    enum {
        key_gemm_iter_fwd = memory_tracking::names::key_nested_multiple,
        key_gemm_iter_fwd_2,
        key_gemm_layer_fwd,
        key_gemm_iter_bwd,
        key_gemm_iter_bwd_2,
        key_gemm_layer_bwd,
        key_gemm_diff_wei_layer,
        key_gemm_diff_wei_iter,
        key_gemm_diff_wei_iter_2,
    };

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        pd_t(const pd_t &other) : base_pd_t(other) { copy_from(other); }

        DECLARE_COMMON_PD_T("ref:any", class_name);

        status_t init(engine_t *engine);

        status_t set_default_params();

        rnn_conf_t conf;
        rnn_offsets_t off;
        rnn_utils::conf_t rnn_conf;
        data_type_t acc_data_t;
        data_type_t src_type;
        data_type_t weights_type;

        std::unique_ptr<primitive_desc_t> gemm_iter_fwd_pd_;
        std::unique_ptr<primitive_desc_t> gemm_iter_fwd_2_pd_;
        std::unique_ptr<primitive_desc_t> gemm_layer_fwd_pd_;
        std::unique_ptr<primitive_desc_t> gemm_iter_bwd_pd_;
        std::unique_ptr<primitive_desc_t> gemm_iter_bwd_2_pd_;
        std::unique_ptr<primitive_desc_t> gemm_layer_bwd_pd_;
        std::unique_ptr<primitive_desc_t> gemm_diff_wei_layer_pd_;
        std::unique_ptr<primitive_desc_t> gemm_diff_wei_iter_pd_;
        std::unique_ptr<primitive_desc_t> gemm_diff_wei_iter_2_pd_;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, scratchpad_sz, 1,
                    OCL_BUFFER_ALIGNMENT, 4096);
            scratchpad.book(key_rnn_gates, rnn_conf.scratch_gates_size, 1,
                    OCL_BUFFER_ALIGNMENT, 4096);
            scratchpad.book(key_rnn_cell, rnn_conf.scratch_cell_size, 1,
                    OCL_BUFFER_ALIGNMENT, 4096);
            // book scratchpad for nested primitives
            switch (aprop) {
                case prop_kind::forward:
                    scratchpad.book(key_gemm_iter_fwd,
                            gemm_iter_fwd_pd_->scratchpad_registry());
                    scratchpad.book(key_gemm_layer_fwd,
                            gemm_layer_fwd_pd_->scratchpad_registry());
                    if (conf.is_vanilla_gru)
                        scratchpad.book(key_gemm_iter_fwd_2,
                                gemm_iter_fwd_2_pd_->scratchpad_registry());
                    break;
                case prop_kind::backward:
                    scratchpad.book(key_gemm_iter_bwd,
                            gemm_iter_bwd_pd_->scratchpad_registry());
                    scratchpad.book(key_gemm_layer_bwd,
                            gemm_layer_bwd_pd_->scratchpad_registry());
                    scratchpad.book(key_gemm_diff_wei_layer,
                            gemm_diff_wei_layer_pd_->scratchpad_registry());
                    scratchpad.book(key_gemm_diff_wei_iter,
                            gemm_diff_wei_iter_pd_->scratchpad_registry());
                    if (conf.is_vanilla_gru) {
                        scratchpad.book(key_gemm_iter_bwd_2,
                                gemm_iter_bwd_2_pd_->scratchpad_registry());
                        scratchpad.book(key_gemm_diff_wei_iter_2,
                                gemm_diff_wei_iter_2_pd_
                                        ->scratchpad_registry());
                    }
                    break;
                default: assert(!"unknown prop_kind");
            }
        }

        void copy_from(const pd_t &other) {
            conf = other.conf;
            off = other.off;
            rnn_conf = other.rnn_conf;
            acc_data_t = other.acc_data_t;
            src_type = other.src_type;
            weights_type = other.weights_type;
            gemm_layer_fwd_pd_.reset(other.gemm_layer_fwd_pd_
                            ? other.gemm_layer_fwd_pd_->clone()
                            : nullptr);
            gemm_iter_fwd_pd_.reset(other.gemm_iter_fwd_pd_
                            ? other.gemm_iter_fwd_pd_->clone()
                            : nullptr);
            gemm_iter_fwd_2_pd_.reset(other.gemm_iter_fwd_2_pd_
                            ? other.gemm_iter_fwd_2_pd_->clone()
                            : nullptr);
            gemm_layer_bwd_pd_.reset(other.gemm_layer_bwd_pd_
                            ? other.gemm_layer_bwd_pd_->clone()
                            : nullptr);
            gemm_iter_bwd_pd_.reset(other.gemm_iter_bwd_pd_
                            ? other.gemm_iter_bwd_pd_->clone()
                            : nullptr);
            gemm_iter_bwd_2_pd_.reset(other.gemm_iter_bwd_2_pd_
                            ? other.gemm_iter_bwd_2_pd_->clone()
                            : nullptr);
            gemm_diff_wei_layer_pd_.reset(other.gemm_diff_wei_layer_pd_
                            ? other.gemm_diff_wei_layer_pd_->clone()
                            : nullptr);
            gemm_diff_wei_iter_pd_.reset(other.gemm_diff_wei_iter_pd_
                            ? other.gemm_diff_wei_iter_pd_->clone()
                            : nullptr);
            gemm_diff_wei_iter_2_pd_.reset(other.gemm_diff_wei_iter_2_pd_
                            ? other.gemm_diff_wei_iter_2_pd_->clone()
                            : nullptr);
        }
    }; // struct pd_t : public base_pd_t

    _ref_rnn_common_t(const pd_t *apd) : gpu_primitive_t(apd) {
        using namespace rnn_utils;
        auto assign_funcs = [](gemm_t &g, weights_assign_t &p) {
            g = &class_name::gemm_primitive;
            p = &class_name::assign_weights;
        };

        assign_funcs(gemm_iter_func, weights_iter_assign_func);
        assign_funcs(gemm_layer_func, weights_layer_assign_func);

        switch (pd()->cell_kind()) {
            case dnnl_vanilla_lstm:
                cell_func = &class_name::cell_execution;
                elemwise_func = pd()->src_type == data_type::u8
                                && pd()->weights_type == data_type::s8
                        ? &class_name::lstm_elemwise_u8s8
                        : &class_name::lstm_elemwise;
                break;
            case dnnl_vanilla_rnn:
                cell_func = &class_name::cell_execution;
                elemwise_func = &class_name::rnn_elemwise;
                break;
            case dnnl_vanilla_gru:
                cell_func = &class_name::cell_execution_gru;
                elemwise_func = &class_name::gru_elemwise;
                break;
            case dnnl_lbr_gru:
                cell_func = &class_name::cell_execution_gru_lbr;
                elemwise_func = &class_name::gru_lbr_elemwise;
                break;
            default: break;
        }

        grid_computation = &class_name::linear_execution;

        size_t scratchpad_size, workspace_size;
        rnn_utils::set_offsets(pd()->rnn_conf, ws_gates_offset_,
                ws_states_offset_, ws_c_states_offset_, ws_diff_states_offset_,
                ws_grid_comp_offset_, scratch_cell_offset_, ws_dhG1_offset_,
                ws_bias_offset_, scratch_gates_offset_, scratchpad_size,
                workspace_size);
        int max_nparts = (pd()->cell_kind() == alg_kind::vanilla_gru) ? 2 : 1;
        int wei_offsets_iter_sz = pd()->L() * pd()->D() * max_nparts;
        int wei_offsets_layer_sz = pd()->L() * pd()->D();

        wei_layer_offset_ptr
                = (size_t *)malloc(sizeof(size_t) * wei_offsets_layer_sz, 64);
        wei_iter_offset_ptr
                = (size_t *)malloc(sizeof(size_t) * wei_offsets_iter_sz, 64);
    }

    status_t init(engine_t *engine) override;

    ~_ref_rnn_common_t() {
        free(wei_layer_offset_ptr);
        free(wei_iter_offset_ptr);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_(ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {gemm_layer_fwd_.get(), gemm_iter_fwd_.get(),
                gemm_iter_fwd_2_.get(), gemm_layer_bwd_.get(),
                gemm_iter_bwd_.get(), gemm_iter_bwd_2_.get(),
                gemm_diff_wei_layer_.get(), gemm_diff_wei_iter_.get(),
                gemm_diff_wei_iter_2_.get()};
    }

    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override;

private:
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    // set the class names
    grid_execution_sig(linear_execution);

    cell_execution_sig(cell_execution);
    cell_execution_sig(cell_execution_gru);
    cell_execution_sig(cell_execution_gru_lbr);

    elemwise_sig(rnn_elemwise);
    elemwise_sig(lstm_elemwise);
    elemwise_sig(lstm_elemwise_u8s8);
    elemwise_sig(gru_lbr_elemwise);
    elemwise_sig(gru_elemwise);

    gemm_sig(gemm_primitive);

    weights_assign_sig(assign_weights);

    float (*activation_func)(float dd, float s, float alpha, float cliping);
    void bias_prepare(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, int n_layer, int n_dir,
            int n_bias, int n_gates, int dhc, const memory_storage_t &ws,
            const memory_storage_t &scales, const memory_storage_t &wei_layer,
            const memory_storage_t &wei_iter,
            const memory_storage_t &bias) const;
    void copy_init_layer(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, bool lr, bool rl,
            int n_iter, int batch, int slc, const memory_storage_t &ws,
            const memory_storage_t &input,
            const memory_storage_t &diff_dst_layer) const;
    void copy_init_iter(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, int n_layer, int n_dir,
            int batch, int sic, int dhc, const memory_storage_t &ws,
            const memory_storage_t &firstit_states,
            const memory_storage_t &firstit_c_states,
            const memory_storage_t &diff_dst_iter,
            const memory_storage_t &diff_dst_iter_c, const float shift,
            const float scale, const bool quantize) const;
    void copy_res_layer(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, bool lr, bool rl,
            int n_iter, int batch, int slc, int dlc,
            const memory_storage_t &dst_last_layer,
            const memory_storage_t &diff_src_layer, const memory_storage_t &ws,
            const float shift, const float scale, const bool dequantize) const;
    void copy_res_iter(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream, int n_layer, int n_dir,
            int batch, int sic, int dhc, const memory_storage_t &dst_last_iter,
            const memory_storage_t &dst_last_iter_c,
            const memory_storage_t &diff_src_iter,
            const memory_storage_t &diff_src_iter_c, const memory_storage_t &ws,
            const float shift, const float scale, const bool dequantize) const;
    void gates_reduction(const exec_ctx_t &ctx, int dir, int lay, int iter,
            int n_gates, int dhc, int batch, const memory_storage_t &gates,
            const memory_storage_t &cell,
            const memory_storage_t &diff_bias) const;
    void ws_set(const exec_ctx_t &ctx,
            compute::compute_stream_t *compute_stream,
            const memory_storage_t &workspace, const cl_ulong ws_offset,
            const int ws_part, const float val, const size_t size) const;
#if DEBUGPRINT
    void ws_print(const exec_ctx_t &ctx, compute::compute_stream_t *s,
            const memory_storage_t &workspace) const;
    compute::kernel_t ws_print_kernel_;
#endif

    compute::kernel_t bias_prepare_kernel_;
    compute::kernel_t copy_init_layer_kernel_;
    compute::kernel_t copy_init_iter_kernel_;
    compute::kernel_t copy_res_layer_kernel_;
    compute::kernel_t copy_res_iter_kernel_;

    compute::kernel_t ws_set_kernel_;
    compute::kernel_t elemwise_fwd_kernel_;
    compute::kernel_t elemwise_bwd_kernel_;
    compute::kernel_t gates_reduction_kernel_;

    // ptrs to GEMM primitives
    std::shared_ptr<primitive_t> gemm_layer_fwd_;
    std::shared_ptr<primitive_t> gemm_iter_fwd_;
    std::shared_ptr<primitive_t> gemm_iter_fwd_2_;
    std::shared_ptr<primitive_t> gemm_layer_bwd_;
    std::shared_ptr<primitive_t> gemm_iter_bwd_;
    std::shared_ptr<primitive_t> gemm_iter_bwd_2_;
    std::shared_ptr<primitive_t> gemm_diff_wei_layer_;
    std::shared_ptr<primitive_t> gemm_diff_wei_iter_;
    std::shared_ptr<primitive_t> gemm_diff_wei_iter_2_;

    // offset variables set in workspace and used in offset calculations for
    // grid & cell execution and fwd & bwd kernel macros
    cl_ulong ws_gates_offset_;
    cl_ulong ws_states_offset_;
    cl_ulong ws_c_states_offset_;
    cl_ulong ws_diff_states_offset_;
    cl_ulong ws_grid_comp_offset_;
    cl_ulong scratch_cell_offset_;
    cl_ulong ws_bias_offset_;
    cl_ulong scratch_gates_offset_;
    cl_ulong ws_dhG1_offset_;

    // ptrs for storing weight offsets which are pre-calculated in
    // in grid execution as weights_*_assing_func
    size_t *wei_layer_offset_ptr;
    size_t *wei_iter_offset_ptr;

    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    weights_assign_t weights_layer_assign_func;
    weights_assign_t weights_iter_assign_func;

    gemm_t gemm_iter_func;
    gemm_t gemm_layer_func;
    elemwise_f elemwise_func;

    enum { SCALES_ = 0, TM_SCALES_ = 1 };
};
using ref_rnn_fwd_t = _ref_rnn_common_t<prop_kind::forward>;
using ref_rnn_bwd_t = _ref_rnn_common_t<prop_kind::backward>;
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
