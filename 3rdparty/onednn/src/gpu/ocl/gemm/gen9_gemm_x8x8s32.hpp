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

#ifndef GPU_OCL_GEMM_GEN9_GEMM_X8X8S32_HPP
#define GPU_OCL_GEMM_GEN9_GEMM_X8X8S32_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/gemm/gen9_gemm_kernel_x8x8s32.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_gemm_x8x8s32_t : public gpu_gemm_t {
    enum class type { no_copy };

    struct pd_t : public gpu_gemm_pd_t {
        using gpu_gemm_pd_t::gpu_gemm_pd_t;

        DECLARE_COMMON_PD_T("ocl:gemm:any", gen9_gemm_x8x8s32_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            using namespace primitive_kind;
            using smask_t = primitive_attr_t::skip_mask_t;

            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = smask_t::oscale | smask_t::post_ops
                    | smask_t::zero_points_runtime;

            const auto d = desc();
            // LIMITATIONS:
            // - batch is not supported
            // - runtime dims are not supported
            // - bias is not supported
            // - runtime zero points are supported for dst only
            // - attribute zero points are supported for src and weights only
            bool limits_ok = d->batch == 1
                    && !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m, d->n, d->k,
                            d->lda, d->ldb, d->ldc)
                    && d->bias_type == data_type::undef;

            bool ok = limits_ok
                    && utils::one_of(d->a_type, data_type::u8, data_type::s8)
                    && utils::one_of(d->b_type, data_type::u8, data_type::s8)
                    && utils::one_of(d->c_type, data_type::s32)
                    && d->acc_type == d->c_type
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(desc()->c_type == data_type::s32,
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short))
                    && attr()->has_default_values(attr_skip_mask)
                    && zero_points_ok() && attr()->output_scales_.mask_ == 0
                    && attr()->post_ops_.len() <= 2
                    && IMPLICATION(attr()->post_ops_.len() == 1,
                            attr()->post_ops_.find(eltwise) != -1
                                    || attr()->post_ops_.find(sum) != -1)
                    && IMPLICATION(attr()->post_ops_.len() == 2,
                            attr()->post_ops_.find(sum) == 0
                                    && attr()->post_ops_.find(eltwise) == 1);
            if (!ok) return status::unimplemented;
            init_scratchpad();
            attr_info = attr_info_t::create(attr());
            return status::success;
        }

        bool zero_points_ok() const {
            return attr()->zero_points_.defined(DNNL_ARG_SRC)
                    && attr()->zero_points_.defined(DNNL_ARG_WEIGHTS)
                    && (attr()->zero_points_.has_default_values(DNNL_ARG_DST)
                            || !attr()->zero_points_.defined(DNNL_ARG_DST));
        }

        float alpha() const { return attr()->output_scales_.scales_[0]; }

        float beta() const {
            using namespace primitive_kind;
            const auto &p = attr()->post_ops_;
            return p.contain(sum, 0) ? p.entry_[0].sum.scale : 0.f;
        }

        attr_info_t attr_info = {};

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;
        size_t dyn_offset_co = 0;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_gemm_int_c_in_acc_dt,
                    desc()->m * desc()->n, sizeof(int), OCL_BUFFER_ALIGNMENT);
        }
    };

    status_t init(engine_t *engine) override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        auto *dev_info = compute_engine->device_info();

        eu_count_ = dev_info->eu_count();
        hw_threads_ = dev_info->hw_threads();

        gemm_type_ = get_gemm_type();

        switch (gemm_type_) {
            case type::no_copy: return init_nocopy(engine);
        }

        return status::invalid_arguments;
    }

    gen9_gemm_x8x8s32_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    status_t init_nocopy(engine_t *engine) {
        const char *kernel_name = nullptr;

        //compute kernel
        switch (pd()->desc()->c_type) {
            case data_type::s32:
                kernel_name = "gen9_gemm_compute_x8x8s32";
                break;
            default: return status::unimplemented;
        }

        compute::kernel_ctx_t kernel_ctx;

        int cmask = 0;
        pd()->attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &cmask, nullptr);
        bool fixed_c = (0 == cmask);
        bool column_c = (1 << 0 == cmask);
        bool row_c = (1 << 1 == cmask);

        auto status = gen9_gemm_x8x8s32_kernel_t::init_kernel_ctx(kernel_ctx,
                pd()->desc()->transa, pd()->desc()->transb, fixed_c, column_c,
                row_c, pd()->attr_info, pd()->desc()->a_type,
                pd()->desc()->b_type, pd()->desc()->c_type);
        if (status != status::success) return status;

        create_kernel(
                engine, &compute_x8x8s32_kernel_, kernel_name, kernel_ctx);
        if (!compute_x8x8s32_kernel_) return status::runtime_error;

        //scale kernel
        kernel_name = "gen9_gemm_scale_x8x8s32";

        status = gen9_gemm_scale_x8x8s32_kernel_t::init_kernel_ctx(kernel_ctx,
                pd()->attr_info, pd()->desc()->a_type, pd()->desc()->b_type,
                pd()->desc()->c_type);
        if (status != status::success) return status;

        create_kernel(engine, &scale_x8x8s32_kernel_, kernel_name, kernel_ctx);
        if (!scale_x8x8s32_kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    status_t launch_x8x8s32(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, const memory_storage_t &a,
            const memory_storage_t &b, const memory_storage_t &c,
            int64_t offset_a, int64_t offset_b, int64_t offset_c, int64_t lda,
            int64_t ldb, int64_t ldc, int64_t m, int64_t n, int64_t k,
            int64_t beta, int32_t ao, int32_t bo, const memory_storage_t &co,
            int64_t offset_co, bool apply_co, bool apply_eltwise,
            float eltwise_alpha, float eltwise_beta, float eltwise_scale) const;

    status_t launch_scale_x8x8s32(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, const memory_storage_t &c_temp,
            const memory_storage_t &c, char offsetc, int64_t offset_c,
            int64_t m, int64_t n, int64_t ldc, float alpha, float beta,
            const memory_storage_t &co, int64_t offset_co, bool alpha_is_zero,
            bool apply_eltwise, float eltwise_alpha, float eltwise_beta,
            float eltwise_scale) const;

    status_t launch_scale_x8x8s32(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, const memory_storage_t &c_temp,
            const memory_storage_t &c, char offsetc, int64_t offset_c,
            int64_t m, int64_t n, int64_t ldc, float alpha, float beta,
            const memory_storage_t &co, int64_t offset_co, bool alpha_is_zero,
            bool apply_eltwise, float eltwise_alpha, float eltwise_beta) const;

    virtual status_t execute_standard(const gemm_exec_ctx_t &ctx) const;

    compute::kernel_t compute_x8x8s32_kernel_;
    compute::kernel_t scale_x8x8s32_kernel_;

    type gemm_type_ = type::no_copy;
    int hw_threads_ = 0;
    int eu_count_ = 0;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    type get_gemm_type() const { return type::no_copy; }
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
