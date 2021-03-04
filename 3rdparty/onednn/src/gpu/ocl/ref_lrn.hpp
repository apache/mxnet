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

#ifndef GPU_OCL_REF_LRN_HPP
#define GPU_OCL_REF_LRN_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_lrn_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_lrn_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_lrn_fwd_pd_t {
        pd_t(const lrn_desc_t *adesc, const primitive_attr_t *attr,
                const lrn_fwd_pd_t *hint_fwd_pd)
            : gpu_lrn_fwd_pd_t(adesc, attr, hint_fwd_pd) {}
        virtual ~pd_t() {}

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            bool ok = true
                    && utils::one_of(desc()->prop_kind,
                            prop_kind::forward_inference,
                            prop_kind::forward_training)
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::lrn_across_channels,
                            alg_kind::lrn_within_channel)
                    && utils::one_of(
                            desc()->data_desc.data_type, f32, f16, bf16)
                    && attr()->has_default_values()
                    && IMPLICATION(desc()->data_desc.data_type == f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16));
            if (!ok) return status::unimplemented;

            if (desc_.prop_kind == prop_kind::forward_training) {
                ws_md_ = *src_md();
                if (ws_md_.data_type == data_type::bf16)
                    ws_md_.data_type = data_type::f32;
            }

            dispatch = compute_engine->create_dispatch(&data_md_);
            dispatch.define_dim("MB", 0, MB());
            dispatch.define_dim("IC", 1, C());
            dispatch.define_dim("ID", nstl::max(1, data_md_.ndims - 3), D());
            dispatch.define_dim("IH", nstl::max(1, data_md_.ndims - 2), H());
            dispatch.define_dim("IW", nstl::max(1, data_md_.ndims - 1), W());
            dispatch.generate();

            return status::success;
        }

        compute::dispatch_t dispatch;
    };

    ref_lrn_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;

        status_t status = status::success;
        const auto *desc = pd()->desc();

        kernel_ctx.set_data_type(desc->data_desc.data_type);

        kernel_ctx.define_int("IS_FWD", 1);

        if (desc->prop_kind == prop_kind::forward_training)
            kernel_ctx.define_int("IS_TRAINING", 1);

        switch (desc->alg_kind) {
            case lrn_across_channels:
                kernel_ctx.define_int("ACROSS_CHANNEL", 1);
                break;
            case lrn_within_channel:
                kernel_ctx.define_int("WITHIN_CHANNEL", 1);
                break;
            default: status = status::unimplemented;
        }
        if (status != status::success) return status;

        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper dst_d(pd()->dst_md());
        const int ndims = src_d.ndims();

        kernel_ctx.define_int("NDIMS", ndims);
        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("IC", pd()->C());
        kernel_ctx.define_int("ID", pd()->D());
        kernel_ctx.define_int("IH", pd()->H());
        kernel_ctx.define_int("IW", pd()->W());

        const uint32_t round_norm_size = desc->local_size;
        uint32_t num_elements = pow(round_norm_size, nstl::max(0, ndims - 2));
        if (desc->alg_kind == lrn_across_channels) {
            num_elements = round_norm_size;
        }
        const float num_element_div = 1.f / (float)num_elements;
        const auto padding = (desc->local_size - 1) / 2;

        kernel_ctx.define_float("NUM_ELEMENTS_DIV", num_element_div);
        kernel_ctx.define_int("PADDING", padding);
        kernel_ctx.define_int(
                "LOCAL_SIZE", desc->local_size - 1 + desc->local_size % 2);
        kernel_ctx.define_float("LRN_ALPHA", desc->lrn_alpha);
        kernel_ctx.define_float("LRN_BETA", desc->lrn_beta);
        kernel_ctx.define_float("LRN_K", desc->lrn_k);

        offsets_t off;
        set_offsets(src_d, off.src_off);
        set_offsets(dst_d, off.dst_off);
        def_offsets(off.src_off, kernel_ctx, "SRC", ndims);
        def_offsets(off.dst_off, kernel_ctx, "DST", ndims);

        def_dispatch(kernel_ctx, pd()->dispatch);

        create_kernel(engine, &kernel_, "ref_lrn_fwd", kernel_ctx);
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

struct ref_lrn_bwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_lrn_bwd_pd_t {
        pd_t(const lrn_desc_t *adesc, const primitive_attr_t *attr,
                const lrn_fwd_pd_t *hint_fwd_pd)
            : gpu_lrn_bwd_pd_t(adesc, attr, hint_fwd_pd) {}
        virtual ~pd_t() {}

        DECLARE_COMMON_PD_T("ref:any", ref_lrn_bwd_t);

        status_t init(engine_t *engine) {
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            bool ok = true
                    && utils::one_of(
                            desc()->prop_kind, prop_kind::backward_data)
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::lrn_across_channels,
                            alg_kind::lrn_within_channel)
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f32, data_type::bf16)
                    && set_default_formats_common()
                    && attr()->has_default_values()
                    && IMPLICATION(
                            desc()->data_desc.data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16));
            if (!ok) return status::unimplemented;

            ws_md_ = *src_md();
            if (ws_md_.data_type == data_type::bf16)
                ws_md_.data_type = data_type::f32;
            if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;

            dispatch = compute_engine->create_dispatch(&diff_data_md_);
            dispatch.define_dim("MB", 0, MB());
            dispatch.define_dim("IC", 1, C());
            dispatch.define_dim("ID", nstl::max(1, data_md_.ndims - 3), D());
            dispatch.define_dim("IH", nstl::max(1, data_md_.ndims - 2), H());
            dispatch.define_dim("IW", nstl::max(1, data_md_.ndims - 1), W());
            dispatch.generate();

            return status::success;
        }

        compute::dispatch_t dispatch;
    };

    ref_lrn_bwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;

        status_t status = status::success;
        const auto *desc = pd()->desc();

        kernel_ctx.set_data_type(desc->data_desc.data_type);

        kernel_ctx.define_int("IS_BWD", 1);

        switch (desc->alg_kind) {
            case lrn_across_channels:
                kernel_ctx.define_int("ACROSS_CHANNEL", 1);
                break;
            case lrn_within_channel:
                kernel_ctx.define_int("WITHIN_CHANNEL", 1);
                break;
            default: status = status::unimplemented;
        }
        if (status != status::success) return status;

        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
        const int ndims = src_d.ndims();

        kernel_ctx.define_int("NDIMS", ndims);
        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("IC", pd()->C());
        kernel_ctx.define_int("ID", pd()->D());
        kernel_ctx.define_int("IH", pd()->H());
        kernel_ctx.define_int("IW", pd()->W());

        const uint32_t round_norm_size = desc->local_size;
        uint32_t num_elements = pow(round_norm_size, nstl::max(0, ndims - 2));
        if (desc->alg_kind == lrn_across_channels) {
            num_elements = round_norm_size;
        }
        const float num_element_div = 1.f / (float)num_elements;
        const auto padding = (desc->local_size - 1) / 2;

        kernel_ctx.define_float("NUM_ELEMENTS_DIV", num_element_div);
        kernel_ctx.define_int("PADDING", padding);
        kernel_ctx.define_int(
                "LOCAL_SIZE", desc->local_size - 1 + desc->local_size % 2);
        kernel_ctx.define_float("LRN_ALPHA", desc->lrn_alpha);
        kernel_ctx.define_float("LRN_BETA", desc->lrn_beta);
        kernel_ctx.define_float("LRN_K", desc->lrn_k);

        offsets_t off;
        set_offsets(src_d, off.src_off);
        set_offsets(diff_dst_d, off.dst_off);
        def_offsets(off.src_off, kernel_ctx, "SRC", ndims);
        def_offsets(off.dst_off, kernel_ctx, "DST", ndims);

        def_dispatch(kernel_ctx, pd()->dispatch);

        create_kernel(engine, &kernel_, "ref_lrn_bwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
