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

#ifndef GPU_OCL_REF_RESAMPLING_HPP
#define GPU_OCL_REF_RESAMPLING_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resampling_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_resampling_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_resampling_fwd_pd_t {
        pd_t(const resampling_desc_t *adesc, const primitive_attr_t *attr,
                const resampling_fwd_pd_t *hint_fwd_pd)
            : gpu_resampling_fwd_pd_t(adesc, attr, hint_fwd_pd) {}
        virtual ~pd_t() {}

        DECLARE_COMMON_PD_T("ref:any", ref_resampling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            bool ok = is_fwd() && src_md()->data_type == dst_md()->data_type
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            auto blocking = src_md()->format_desc.blocking;
            bool only_c_blocked
                    = blocking.inner_nblks == 1 && blocking.inner_idxs[0] == 1;
            c_block = only_c_blocked ? blocking.inner_blks[0] : 1;

            dispatch = compute_engine->create_dispatch(dst_md());
            dispatch.define_dim("MB", 0, MB());
            dispatch.define_dim("C", 1, C(), c_block);
            dispatch.define_dim("OD", nstl::max(2, dst_md()->ndims - 3), OD());
            dispatch.define_dim("OH", nstl::max(2, dst_md()->ndims - 2), OH());
            dispatch.define_dim("OW", nstl::max(2, dst_md()->ndims - 1), OW());
            dispatch.generate();

            return status::success;
        }
        int c_block;
        compute::dispatch_t dispatch;
    };

    ref_resampling_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;

        status_t status = status::success;
        const auto *desc = pd()->desc();

        kernel_ctx.set_data_type(pd()->src_md()->data_type);

        kernel_ctx.define_int("IS_FWD", 1);

        switch (desc->alg_kind) {
            case resampling_nearest: kernel_ctx.define_int("NEAREST", 1); break;
            case resampling_linear: kernel_ctx.define_int("LINEAR", 1); break;
            default: status = status::unimplemented;
        }
        if (status != status::success) return status;

        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper dst_d(pd()->dst_md());
        const int ndims = dst_d.ndims();

        kernel_ctx.define_int("NDIMS", ndims);
        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("C", pd()->C());
        kernel_ctx.define_int("C_BLOCK", pd()->c_block);
        kernel_ctx.define_int("ID", pd()->ID());
        kernel_ctx.define_int("IH", pd()->IH());
        kernel_ctx.define_int("IW", pd()->IW());
        kernel_ctx.define_int("OD", pd()->OD());
        kernel_ctx.define_int("OH", pd()->OH());
        kernel_ctx.define_int("OW", pd()->OW());
        kernel_ctx.define_float("FD", pd()->FD());
        kernel_ctx.define_float("FH", pd()->FH());
        kernel_ctx.define_float("FW", pd()->FW());

        offsets_t off;
        set_offsets(src_d, off.src_off);
        set_offsets(dst_d, off.dst_off);
        def_offsets(off.src_off, kernel_ctx, "SRC", ndims);
        def_offsets(off.dst_off, kernel_ctx, "DST", ndims);

        def_dispatch(kernel_ctx, pd()->dispatch);

        create_kernel(engine, &kernel_, "ref_resampling_fwd", kernel_ctx);
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

struct ref_resampling_bwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_resampling_bwd_pd_t {
        pd_t(const resampling_desc_t *adesc, const primitive_attr_t *attr,
                const resampling_fwd_pd_t *hint_fwd_pd)
            : gpu_resampling_bwd_pd_t(adesc, attr, hint_fwd_pd) {}
        virtual ~pd_t() {}

        DECLARE_COMMON_PD_T("ref:any", ref_resampling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            bool ok = !is_fwd()
                    && utils::one_of(diff_src_md()->data_type, f32, bf16)
                    && diff_src_md()->data_type == diff_dst_md()->data_type
                    && set_default_params() == status::success
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            dispatch = compute_engine->create_dispatch(diff_src_md());
            dispatch.define_dim("MB", 0, MB());
            dispatch.define_dim("C", 1, C());
            dispatch.define_dim(
                    "ID", nstl::max(2, diff_src_md()->ndims - 3), ID());
            dispatch.define_dim(
                    "IH", nstl::max(2, diff_src_md()->ndims - 2), IH());
            dispatch.define_dim(
                    "IW", nstl::max(2, diff_src_md()->ndims - 1), IW());
            dispatch.generate();

            return status::success;
        }
        compute::dispatch_t dispatch;
    };

    ref_resampling_bwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        using namespace alg_kind;

        compute::kernel_ctx_t kernel_ctx;

        status_t status = status::success;
        const auto *desc = pd()->desc();

        kernel_ctx.set_data_type(pd()->diff_src_md()->data_type);

        kernel_ctx.define_int("IS_BWD", 1);

        switch (desc->alg_kind) {
            case resampling_nearest: kernel_ctx.define_int("NEAREST", 1); break;
            case resampling_linear: kernel_ctx.define_int("LINEAR", 1); break;
            default: status = status::unimplemented;
        }
        if (status != status::success) return status;

        const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
        const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
        const int ndims = diff_dst_d.ndims();

        kernel_ctx.define_int("NDIMS", ndims);
        kernel_ctx.define_int("MB", pd()->MB());
        kernel_ctx.define_int("C", pd()->C());
        kernel_ctx.define_int("ID", pd()->ID());
        kernel_ctx.define_int("IH", pd()->IH());
        kernel_ctx.define_int("IW", pd()->IW());
        kernel_ctx.define_int("OD", pd()->OD());
        kernel_ctx.define_int("OH", pd()->OH());
        kernel_ctx.define_int("OW", pd()->OW());
        kernel_ctx.define_float("FD", pd()->FD());
        kernel_ctx.define_float("FH", pd()->FH());
        kernel_ctx.define_float("FW", pd()->FW());

        offsets_t off;
        set_offsets(diff_src_d, off.src_off);
        set_offsets(diff_dst_d, off.dst_off);
        def_offsets(off.src_off, kernel_ctx, "SRC", ndims);
        def_offsets(off.dst_off, kernel_ctx, "DST", ndims);

        def_dispatch(kernel_ctx, pd()->dispatch);

        create_kernel(engine, &kernel_, "ref_resampling_bwd", kernel_ctx);
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
