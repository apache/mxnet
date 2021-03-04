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

#ifndef GPU_OCL_REF_MATMUL_HPP
#define GPU_OCL_REF_MATMUL_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_matmul_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_matmul_t : public gpu_primitive_t {
    struct pd_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            src_dt_ = src_md()->data_type;
            dst_dt_ = dst_md()->data_type;
            wei_dt_ = weights_md(0)->data_type;
            bia_dt_ = with_bias() ? weights_md(1)->data_type : data_type::f32;

            bool ok = IMPLICATION(desc()->accum_data_type == s32,
                              attr()->zero_points_.common())
                    && IMPLICATION(desc()->accum_data_type != s32,
                            attr()->zero_points_.has_default_values())
                    && attr()->has_default_values(smask_t::oscale_runtime
                            | smask_t::zero_points_runtime | smask_t::post_ops)
                    && attr_oscale_ok()
                    && post_ops_with_binary_ok(attr(), dst_dt_)
                    && set_default_formats()
                    && ((utils::one_of(src_dt_, u8, s8)
                                && utils::one_of(wei_dt_, u8, s8)
                                && utils::one_of(dst_dt_, f32, s8, u8, s32)
                                && IMPLICATION(with_bias(),
                                        utils::one_of(
                                                bia_dt_, f32, u8, s8, s32)))
                            || ((utils::everyone_is(
                                         f32, src_dt_, wei_dt_, dst_dt_)
                                        || utils::everyone_is(
                                                f16, src_dt_, wei_dt_, dst_dt_)
                                        || (utils::everyone_is(
                                                    bf16, src_dt_, wei_dt_)
                                                && utils::one_of(
                                                        dst_dt_, bf16, f32)))
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt_, f32))));

            if (!ok) return status::unimplemented;

            non_default_attrs_ = !attr()->has_default_values();
            is_defined_[SCALES_] = !attr()->output_scales_.has_default_values();
            is_defined_[A0_]
                    = !attr()->zero_points_.has_default_values(DNNL_ARG_SRC);
            is_defined_[B0_] = !attr()->zero_points_.has_default_values(
                    DNNL_ARG_WEIGHTS);
            is_defined_[C0_]
                    = !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
            status_t status = init_scales_md();
            if (status != status::success) return status;

            status = init_zero_points_md(A0_, a0_md_);
            if (status != status::success) return status;

            status = init_zero_points_md(B0_, b0_md_);
            if (status != status::success) return status;

            status = init_zero_points_md(C0_, c0_md_);
            if (status != status::success) return status;

            attr_info_ = attr_info_t::create(attr());

            return status::success;
        }

        const memory_desc_t *scales_md() const { return &scales_md_; }
        const memory_desc_t *zero_points_md(int idx) const {
            switch (idx) {
                case A0_: return &a0_md_;
                case B0_: return &b0_md_;
                case C0_: return &c0_md_;
            }
            return nullptr;
        }

        bool non_default_attrs_ = false;
        bool is_defined_[4] = {};
        data_type_t bia_dt_ = data_type::undef;
        data_type_t src_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;

        attr_info_t attr_info_ = {};

    private:
        bool attr_oscale_ok() const {
            const auto &oscale = attr()->output_scales_;
            return oscale.mask_ == 0 || oscale.mask_ == (1 << (batched() + 1));
        }

        status_t init_scales_md() {
            scales_md_.data_type = data_type::f32;
            scales_md_.ndims = 1;
            scales_md_.dims[0]
                    = is_defined_[SCALES_] ? attr()->output_scales_.count_ : 1;
            return memory_desc_init_by_tag(scales_md_, format_tag::x);
        }

        status_t init_zero_points_md(int idx, memory_desc_t &md_) {
            //TODO: add count when set for the primitive
            md_.data_type = data_type::s32;
            md_.ndims = 1;
            md_.dims[0] = 1;
            return memory_desc_init_by_tag(md_, format_tag::x);
        }

        memory_desc_t a0_md_ = memory_desc_t();
        memory_desc_t b0_md_ = memory_desc_t();
        memory_desc_t c0_md_ = memory_desc_t();
        memory_desc_t scales_md_ = memory_desc_t();
    };

    ref_matmul_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.define_int("DST_NDIMS", pd()->dst_md()->ndims);
        kernel_ctx.define_int("WITH_BIAS", pd()->with_bias());
        kernel_ctx.define_int("NON_DEFAULT_ATTRS", pd()->non_default_attrs_);

        kernel_ctx.set_data_type(pd()->dst_dt_);
        def_attr_info(kernel_ctx, pd()->attr_info_);

        def_data_type(kernel_ctx, pd()->src_dt_, "SRC");
        def_data_type(kernel_ctx, pd()->wei_dt_, "WEI");
        def_data_type(kernel_ctx, pd()->dst_dt_, "DST");
        def_data_type(kernel_ctx, pd()->bia_dt_, "BIA");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");
        create_kernel(engine, &kernel_, "ref_matmul", kernel_ctx);
        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

    status_t handle_runtime_value(engine_t *engine, int idx,
            const memory_desc_t *md,
            std::unique_ptr<memory_storage_t> &mem_storage) const {
        const primitive_attr_t &attr = *pd()->attr();
        void *p;
        memory_desc_wrapper mdw(*md);
        size_t sz = (idx == SCALES_) ? sizeof(float) : sizeof(int);
        memory_storage_t *mem_s_ptr;
        status_t status
                = engine->create_memory_storage(&mem_s_ptr, mdw.nelems() * sz);
        if (status != status::success) {
            mem_storage.reset();
            return status;
        }
        mem_storage.reset(mem_s_ptr);
        assert(sizeof(float) == sizeof(int));
        status = mem_storage->map_data(
                &p, nullptr, sizeof(float) * mdw.nelems());
        if (status != status::success) return status;
        if (!pd()->is_defined_[idx]) {
            if (idx == SCALES_) {
                utils::array_set((float *)p, (float)1, mdw.nelems());
            } else {
                utils::array_set((int *)p, (int)0, mdw.nelems());
            }
        } else {
            switch (idx) {
                case SCALES_:
                    utils::array_copy((float *)p, attr.output_scales_.scales_,
                            attr.output_scales_.count_);
                    break;
                case A0_:
                    utils::array_copy((int *)p,
                            (int *)attr.zero_points_.get(DNNL_ARG_SRC),
                            mdw.nelems());
                    break;
                case B0_:
                    utils::array_copy((int *)p,
                            (int *)attr.zero_points_.get(DNNL_ARG_WEIGHTS),
                            mdw.nelems());
                    break;
                case C0_:
                    utils::array_copy((int *)p,
                            (int *)attr.zero_points_.get(DNNL_ARG_DST),
                            mdw.nelems());
                    break;
            }
        }
        status = mem_storage->unmap_data(p, nullptr);
        return status;
    }

protected:
    status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const override {
        std::unique_ptr<memory_storage_t> tmp_mem_storage;
        for (const auto &idx : {A0_, B0_, C0_}) {
            CHECK(handle_runtime_value(
                    engine, idx, pd()->zero_points_md(idx), tmp_mem_storage));
            r->add_memory_storage(idx, std::move(tmp_mem_storage));
        }

        CHECK(handle_runtime_value(
                engine, SCALES_, pd()->scales_md(), tmp_mem_storage));
        r->add_memory_storage(SCALES_, std::move(tmp_mem_storage));
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    compute::kernel_t kernel_;
    enum { SCALES_ = 0, A0_ = 1, B0_ = 2, C0_ = 3 };
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
