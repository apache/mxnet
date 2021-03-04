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

#ifndef GPU_OCL_REF_INNER_PRODUCT_HPP
#define GPU_OCL_REF_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_inner_product_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_inner_product_fwd_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::oscale
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool ok = true
                    && utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference)
                    && set_default_params() == status::success
                    && utils::one_of(true,
                            expect_data_types(
                                    u8, s8, data_type::undef, s8, s32),
                            expect_data_types(
                                    u8, s8, data_type::undef, u8, s32),
                            expect_data_types(
                                    u8, s8, data_type::undef, s32, s32),
                            expect_data_types(
                                    s8, s8, data_type::undef, s8, s32),
                            expect_data_types(
                                    s8, s8, data_type::undef, u8, s32),
                            expect_data_types(
                                    s8, s8, data_type::undef, s32, s32),
                            expect_data_types(
                                    bf16, bf16, data_type::undef, bf16, f32),
                            expect_data_types(
                                    bf16, bf16, data_type::undef, f32, f32),
                            expect_data_types(f32, f32, f32, f32, f32),
                            expect_data_types(f16, f16, f16, f16, f16))
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type, u8, s8,
                                    bf16, f16, f32))
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_with_binary_ok(
                            attr(), desc()->dst_desc.data_type)
                    && IMPLICATION(!attr()->output_scales_.has_default_values(),
                            utils::one_of(src_md_.data_type, s8, u8)
                                    && attr()->output_scales_.mask_ == 0)
                    && IMPLICATION(desc()->src_desc.data_type == f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16));
            if (!ok) return status::unimplemented;

            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        inner_product_conf_t conf;
        offsets_t off;
    };

    ref_inner_product_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        create_kernel(engine, &kernel_, "ref_inner_product_fwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_inner_product_bwd_data_t : public gpu_primitive_t {
    struct pd_t : public gpu_inner_product_bwd_data_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            assert(engine->kind() == engine_kind::gpu);

            bool ok = true
                    && utils::one_of(
                            this->desc()->prop_kind, backward, backward_data)
                    && this->set_default_params() == status::success
                    && utils::one_of(true,
                            expect_data_types(
                                    bf16, bf16, data_type::undef, bf16, f32),
                            expect_data_types(
                                    f32, bf16, data_type::undef, bf16, f32),
                            expect_data_types(
                                    f32, f32, data_type::undef, f32, f32))
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        inner_product_conf_t conf;
        offsets_t off;
    };

    ref_inner_product_bwd_data_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        create_kernel(
                engine, &kernel_, "ref_inner_product_bwd_data", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_inner_product_bwd_weights_t : public gpu_primitive_t {
    struct pd_t : public gpu_inner_product_bwd_weights_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            assert(engine->kind() == engine_kind::gpu);
            bool ok = true
                    && utils::one_of(
                            this->desc()->prop_kind, backward, backward_weights)
                    && this->set_default_params() == status::success
                    && utils::one_of(true,
                            expect_data_types(bf16, bf16, bf16, bf16, f32),
                            expect_data_types(bf16, f32, f32, bf16, f32),
                            expect_data_types(f32, f32, f32, f32, f32))
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        inner_product_conf_t conf;
        offsets_t off;
    };

    ref_inner_product_bwd_weights_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        create_kernel(
                engine, &kernel_, "ref_inner_product_bwd_weights", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
