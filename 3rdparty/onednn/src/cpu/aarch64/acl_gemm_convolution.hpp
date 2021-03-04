/*******************************************************************************
* Copyright 2020 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_GEMM_CONVOLUTION_HPP
#define CPU_AARCH64_ACL_GEMM_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/aarch64/acl_gemm_convolution_utils.hpp"
#include "cpu/gemm/gemm.hpp"

#include "cpu/cpu_convolution_pd.hpp"

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Scheduler.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_obj_t {
    arm_compute::NEGEMMConvolutionLayer gemm_conv;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_resource_t : public resource_t {
    acl_resource_t() : acl_obj_(utils::make_unique<acl_obj_t>()) {}

    status_t configure(const acl_conv_gemm_conf_t &acp) {
        if (!acl_obj_) return status::out_of_memory;

        // clang-format off
        // Validate convolution manually to check for return status
        arm_compute::NEGEMMConvolutionLayer acl_gemm_conv;
        auto acl_st = acl_gemm_conv.validate(
            &acp.src_info,
            &acp.wei_info,
            acp.with_bias ? &acp.bia_info : nullptr,
            &acp.dst_info,
            acp.padstride_info,
            acp.weights_info,
            acp.dilation_info,
            acp.act_info);
        // clang-format on
        if (acl_st.error_code() != arm_compute::ErrorCode::OK) {
            return status::unimplemented;
        }

        // Init Compute Library tensors based on info from descriptor
        acl_obj_->src_tensor.allocator()->init(acp.src_info);
        acl_obj_->wei_tensor.allocator()->init(acp.wei_info);
        acl_obj_->dst_tensor.allocator()->init(acp.dst_info);
        acl_obj_->bia_tensor.allocator()->init(acp.bia_info);

        // clang-format off
        acl_obj_->gemm_conv.configure(
            &acl_obj_->src_tensor,
            &acl_obj_->wei_tensor,
            acp.with_bias ? &acl_obj_->bia_tensor : nullptr,
            &acl_obj_->dst_tensor,
            acp.padstride_info,
            acp.weights_info,
            acp.dilation_info,
            acp.act_info);
        // clang-format on

        return status::success;
    }

    acl_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_resource_t);

private:
    std::unique_ptr<acl_obj_t> acl_obj_;

}; // acl_resource_t

struct acl_gemm_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), acp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, acl_gemm_convolution_fwd_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops)
                    && post_ops_ok();
            if (!ok) return status::unimplemented;

            auto conf_status = acl_gemm_convolution_utils::init_conf(acp_,
                    src_md_, weights_md_, dst_md_, bias_md_, *desc(), *attr());
            if (conf_status != status::success) return status::unimplemented;

            // Number of threads in Compute Library is set by OMP_NUM_THREADS
            // dnnl_get_max_threads() == OMP_NUM_THREADS
            arm_compute::Scheduler::get().set_num_threads(
                    dnnl_get_max_threads());

            // TODO: remove dependence on scratchpad memory
            // Using user provided memory for the biases currently segfaults
            if (acp_.with_bias) {
                auto scratchpad = scratchpad_registry().registrar();
                const size_t bia_mem_sz_
                        = acp_.bia_info.tensor_shape()[0] * sizeof(data_t);
                scratchpad.template book<data_t>(
                        memory_tracking::names::key_none, bia_mem_sz_);
            }

            return status::success;
        }

        acl_conv_gemm_conf_t acp_;

    protected:
        bool post_ops_ok() const {
            auto const &po = attr()->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(); };

            bool eltwise_ok = false;
            // Compute Library supports only one eltwise post-op
            if (po.len() == 1 && is_eltwise(0)) {
                const auto act_type = po.entry_[0].eltwise.alg;
                eltwise_ok = acl_gemm_convolution_utils::acl_act_ok(act_type);
            }

            return eltwise_ok || (po.len() == 0);
        }
    };

    acl_gemm_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        auto st = r->configure(pd()->acp_);
        if (st == status::success) { mapper.add(this, std::move(r)); }

        return st;
    }

    ~acl_gemm_convolution_fwd_t() {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

}; // acl_gemm_convolution_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
