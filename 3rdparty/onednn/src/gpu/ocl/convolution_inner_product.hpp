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

#ifndef GPU_OCL_CONVOLUTION_INNER_PRODUCT_HPP
#define GPU_OCL_CONVOLUTION_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct convolution_inner_product_fwd_t : public primitive_t {
    struct pd_t : public gpu_inner_product_fwd_pd_t {
        using gpu_inner_product_fwd_pd_t::gpu_inner_product_fwd_pd_t;

        pd_t(const pd_t &rhs)
            : gpu_inner_product_fwd_pd_t(rhs)
            , conf(rhs.conf)
            , cpd_(rhs.cpd_->clone()) {
            if (rhs.rpd_dst_) rpd_dst_.reset(rhs.rpd_dst_->clone());
            if (rhs.rpd_postop_) rpd_postop_.reset(rhs.rpd_postop_->clone());
        }

        DECLARE_COMMON_PD_T("ocl:conv", convolution_inner_product_fwd_t);

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
                    && set_default_params(true) == status::success
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
                                    bf16, bf16, data_type::undef, f32, f32),
                            expect_data_types(f32, f32, f32, f32, f32),
                            expect_data_types(f16, f16, f16, f16, f16))
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type, u8, s8,
                                    bf16, f16, f32))
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_with_binary_ok(
                            attr(), desc()->dst_desc.data_type)
                    && IMPLICATION(desc()->src_desc.data_type == f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && (invariant_src_md()->format_desc.blocking.inner_nblks > 0
                            || invariant_wei_md()
                                            ->format_desc.blocking.inner_nblks
                                    > 0
                            || (src_md_.format_kind == format_kind::any
                                    && weights_md_.format_kind
                                            == format_kind::any));

            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));
            CHECK(init_scratchpad());
            return status::success;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        inner_product_conf_t conf;

        std::unique_ptr<primitive_desc_t> cpd_;
        std::unique_ptr<primitive_desc_t> rpd_postop_;
        std::unique_ptr<primitive_desc_t> rpd_dst_;

    private:
        status_t init_scratchpad();
    };

    convolution_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        pd()->cpd_->create_primitive(conv_, engine);
        if (pd()->conf.reorder_dst) {
            if (pd()->rpd_postop_)
                pd()->rpd_postop_->create_primitive(postop_reorder_, engine);
            pd()->rpd_dst_->create_primitive(dst_reorder_, engine);
        }
        return status::success;
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        CHECK(conv_->create_resource(engine, mapper));
        if (pd()->conf.reorder_dst) {
            if (postop_reorder_)
                CHECK(postop_reorder_->create_resource(engine, mapper));
            CHECK(dst_reorder_->create_resource(engine, mapper));
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> conv_;
    std::shared_ptr<primitive_t> postop_reorder_;
    std::shared_ptr<primitive_t> dst_reorder_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
