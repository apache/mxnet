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

#ifndef GPU_OCL_GEN9_WINO_CONVOLUTION_HPP
#define GPU_OCL_GEN9_WINO_CONVOLUTION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_wino_convolution_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : gpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen9:wino", gen9_wino_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto src_data_t = this->desc()->src_desc.data_type;
            auto dst_data_t = this->desc()->dst_desc.data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool ok = set_default_alg_kind(alg_kind::convolution_winograd)
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                            forward_inference)
                    && this->desc()->alg_kind == alg_kind::convolution_winograd
                    && utils::one_of(
                            true, expect_data_types(f32, f32, f32, f32, f32))
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(src_data_t == f16,
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short))
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(attr_skip_mask, dst_data_t)
                    && post_ops_with_binary_ok(attr(), dst_data_t);
            if (!ok) return status::unimplemented;

            status_t status = init_conf();
            if (status != status::success) return status;

            init_scratchpad();

            ok = set_default_formats_common(
                    conf.src_tag, conf.wei_tag, conf.dst_tag);

            return ok ? status::success : status::unimplemented;
        }

        status_t init_conf();
        void init_scratchpad();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conv_conf_t conf;
    };

    gen9_wino_convolution_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        std::vector<const char *> kernel_names
                = {"gen9_wino_conv_fwd", "gen9_wino_wei_transform",
                        "gen9_wino_src_transform", "gen9_wino_dst_transform"};

        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        std::vector<compute::kernel_t> kernels;
        create_kernels(engine, &kernels, kernel_names, kernel_ctx);
        kernel_ = kernels[0];
        wei_trans_kernel_ = kernels[1];
        src_trans_kernel_ = kernels[2];
        dst_trans_kernel_ = kernels[3];
        if (!kernel_ || !wei_trans_kernel_ || !src_trans_kernel_
                || !dst_trans_kernel_)
            return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
    compute::kernel_t wei_trans_kernel_;
    compute::kernel_t src_trans_kernel_;
    compute::kernel_t dst_trans_kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
