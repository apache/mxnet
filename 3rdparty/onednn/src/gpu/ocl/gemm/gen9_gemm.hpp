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

#ifndef GPU_OCL_GEMM_GEN9_GEMM_HPP
#define GPU_OCL_GEMM_GEN9_GEMM_HPP

#include <assert.h>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/gemm/gen9_gemm_kernel.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

union plan_element_t {
    struct {
        int i0 : 31;
        int kid0 : 1;
        int j0 : 31;
        int kid1 : 1;
    };
    struct {
        int32_t next_id;
        int32_t done_count;
    };
};

struct gen9_gemm_t : public gpu_gemm_t {

    enum class type {
        copy_based,
        no_copy,
        no_copy_if_even_off,
        no_copy_superkernel,
        no_copy_k_unroll
    };

    struct pd_t : public gpu_gemm_pd_t {
        using gpu_gemm_pd_t::gpu_gemm_pd_t;

        DECLARE_COMMON_PD_T("ocl:gemm:any", gen9_gemm_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            using namespace primitive_kind;
            using smask_t = primitive_attr_t::skip_mask_t;

            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = smask_t::oscale | smask_t::post_ops;

            const auto d = desc();

            // LIMITATIONS:
            // - runtime dims are not supported
            // - bias is not supported
            bool limits_ok = !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->batch,
                                     d->m, d->n, d->k, d->lda, d->ldb, d->ldc)
                    && d->bias_type == data_type::undef;

            bool ok = limits_ok
                    && utils::one_of(d->a_type, data_type::f32, data_type::bf16,
                            data_type::f16)
                    && utils::one_of(d->b_type, data_type::f32, data_type::bf16,
                            data_type::f16)
                    && utils::one_of(d->c_type, data_type::f32, data_type::bf16,
                            data_type::f16)
                    && utils::one_of(
                            d->acc_type, data_type::f32, data_type::f16)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(desc()->c_type == data_type::f16,
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short))
                    && attr()->has_default_values(attr_skip_mask)
                    && attr()->output_scales_.mask_ == 0
                    && attr()->post_ops_.len() <= 2
                    && IMPLICATION(attr()->post_ops_.len() == 1,
                            attr()->post_ops_.find(eltwise) != -1
                                    || attr()->post_ops_.find(sum) != -1)
                    && IMPLICATION(attr()->post_ops_.len() == 2,
                            attr()->post_ops_.find(sum) == 0
                                    && attr()->post_ops_.find(eltwise) == 1);
            if (!ok) return status::unimplemented;

            // The threshold values for m, n and k were obtained by collecting
            // performance results for the following batch and size values:
            // - batch sizes: 4, 16, 64, 128, 256;
            // - matrix sizes: 1, 16, 32, 64, 128, 512, 1024.
            // Benchdnn with matmul driver was used for obtaining
            // performance results.
            // The size of 32 elements for m, n, k turned out not to
            // introduce any performance regression for smaller sizes
            // but gave performance benefit for bigger matrices.
            bool is_batched = d->batch > 1;
            const int threshold_m = 32;
            const int threshold_n = 32;
            const int threshold_k = 32;
            bool size_ok = d->m > threshold_m && d->n > threshold_n
                    && d->k > threshold_k;
            if (is_batched && !size_ok) return status::unimplemented;

            auto *dev_info = compute_engine->device_info();

            eu_count_ = dev_info->eu_count();
            hw_threads_ = dev_info->hw_threads();

            gemm_type_ = get_gemm_type(engine);
            init_scratchpad();

            attr_info_ = attr_info_t::create(attr());

            return status::success;
        }

        void init_scratchpad() {
            switch (gemm_type_) {
                case type::copy_based: init_scratchpad_copy_based(); break;
                case type::no_copy: init_scratchpad_nocopy(); break;
                case type::no_copy_if_even_off: {
                    init_scratchpad_copy_based();
                    init_scratchpad_nocopy();
                    break;
                }
                case type::no_copy_superkernel:
                    init_scratchpad_nocopy_superkernel();
                    break;
                case type::no_copy_k_unroll: init_scratchpad_nocopy(); break;
            }
        }

        void init_scratchpad_nocopy() {
            int unroll_m, unroll_n, unroll_k;
            bool transa = (desc()->transa == dnnl_trans);
            bool transb = (desc()->transb == dnnl_trans);

            gen9_gemm_nocopy_kernel_t::get_unrolls(transa, transb, unroll_m,
                    unroll_n, unroll_k, desc()->c_type);
            auto m = desc()->m;
            auto n = desc()->n;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_gemm_flag,
                    ((m + unroll_m - 1) / unroll_m)
                            * ((n + unroll_n - 1) / unroll_n),
                    sizeof(int), OCL_BUFFER_ALIGNMENT);
        }

        void init_scratchpad_copy_based() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer,
                    128 << 20, 1, OCL_BUFFER_ALIGNMENT);
        }

        void init_scratchpad_nocopy_superkernel() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer,
                    max_plan_size(), 1, OCL_BUFFER_ALIGNMENT);
        }

        bool with_eltwise() const {
            return attr()->post_ops_.find(primitive_kind::eltwise) != -1;
        }

        float eltwise_alpha() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                    : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                    : 0.0f;
        }

        float eltwise_scale() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.scale
                    : 1.0f;
        }

        float alpha() const { return attr()->output_scales_.scales_[0]; }

        float beta() const {
            using namespace primitive_kind;
            const auto &p = attr()->post_ops_;
            return p.contain(sum, 0) ? p.entry_[0].sum.scale : 0.f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                    : dnnl_alg_kind_undef;
        }

        bool use_superkernel(engine_t *engine) const {
            if (disable_superkernel) return false;

            if (desc()->c_type != data_type::f32) return false;
            if (desc()->a_type != desc()->c_type
                    || desc()->b_type != desc()->c_type)
                return false;

            // Older OpenCL runtimes spill registers very badly with superkernels
            //  (~2% resulting efficiency). Avoid using superkernels for these
            //  versions.
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            auto *dev_info = compute_engine->device_info();
            compute::runtime_version_t min_version = {19, 11, 12599};

            if (dev_info->runtime_version() < min_version) return false;

            bool transa = (desc()->transa == dnnl_trans);
            auto k = desc()->k;

            return !transa && (hw_threads_ > 0) && (k >= 384);
        }

        bool use_nocopy_k_unroll() const {
            bool transa = (desc()->transa == dnnl_trans);
            bool transb = (desc()->transb == dnnl_trans);

            auto m = desc()->m;
            auto n = desc()->n;
            auto k = desc()->k;
            auto lda = desc()->lda;
            auto ldb = desc()->ldb;
            auto c_type = desc()->c_type;

            if (!utils::one_of(c_type, data_type::f32, data_type::f16))
                return false;

            // f16 no-copy kernels require even lda, ldb, offset_a, and offset_b.
            if (c_type == data_type::f16)
                if ((lda & 1) || (ldb & 1)) return false;

            if (c_type == data_type::f32)
                return ((n == 1) && transa && !transb && (m <= 1024)
                        && (k >= 128));

            if (c_type == data_type::f16)
                return ((n == 1) && transa && !transb && (m <= 1024)
                        && (k >= 384));

            return false;
        }

        bool use_nocopy() const {
            bool transa = (desc()->transa == dnnl_trans);
            bool transb = (desc()->transb == dnnl_trans);

            auto m = desc()->m;
            auto n = desc()->n;
            auto k = desc()->k;
            auto lda = desc()->lda;
            auto ldb = desc()->ldb;

            if (!utils::one_of(desc()->c_type, data_type::f32, data_type::f16))
                return false;
            if (desc()->a_type != desc()->c_type
                    || desc()->b_type != desc()->c_type)
                return false;
            if (desc()->acc_type != desc()->c_type) return false;

            // f16 no-copy kernels require even lda, ldb, offset_a, and offset_b.
            if (desc()->c_type == data_type::f16)
                if ((lda & 1) || (ldb & 1)) return false;

            if (transa && !transb) return (m < 1024 || n < 1024);

            if (desc()->c_type == data_type::f16) {
                if (!(lda & 0x3FF) && (n >= 256)) return false;
                if (!transa && transb && (k <= 64)) return false;
            }

            return true;
        }

        type get_gemm_type(engine_t *engine) const {
            return use_nocopy_k_unroll()
                    ? type::no_copy_k_unroll
                    : !use_nocopy() ? type::copy_based
                                    : use_superkernel(engine)
                                    ? type::no_copy_superkernel
                                    : (desc()->c_type == data_type::f16)
                                            ? type::no_copy_if_even_off
                                            : type::no_copy;
        }

        size_t max_plan_size() const {

            auto m = desc()->m;
            auto n = desc()->n;
            bool transa = (desc()->transa == dnnl_trans);
            bool transb = (desc()->transb == dnnl_trans);

            int unroll_m[2], unroll_n;
            gen9_gemm_nocopy_superkernel_t::get_unrolls(
                    transa, transb, unroll_m, unroll_n);

            auto max_threads = utils::div_up(m, unroll_m[1])
                    * utils::div_up(n, unroll_n);

            return sizeof(plan_element_t) * (max_threads + 1);
        }

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;
        int hw_threads_ = 0;
        int eu_count_ = 0;

        type gemm_type_ = type::copy_based;

        attr_info_t attr_info_ = {};

    protected:
#ifdef _WIN32
        bool disable_superkernel = true;
#else
        bool disable_superkernel = false;
#endif
    };

    gen9_gemm_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    status_t init(engine_t *engine) override {
        switch (pd()->gemm_type_) {
            case type::copy_based: return init_copy_based(engine);
            case type::no_copy: return init_nocopy(engine);
            case type::no_copy_if_even_off: {
                status_t result = init_copy_based(engine);
                if (result != status::success) return result;
                return init_nocopy(engine);
            }
            case type::no_copy_superkernel:
                return init_nocopy_superkernel(engine);
            case type::no_copy_k_unroll: return init_nocopy(engine);
        }

        return status::invalid_arguments;
    }

    status_t init_copy_based(engine_t *engine) {
        for (bool beta0 : {false, true}) {
            if (beta0 && pd()->beta() != 0) continue;

            compute::kernel_ctx_t kernel_ctx;
            auto status = gen9_gemm_compute_kernel_t::init_kernel_ctx(
                    kernel_ctx, beta0, pd()->attr_info_, pd()->desc()->acc_type,
                    pd()->desc()->c_type);
            if (status != status::success) return status;

            create_kernel(engine, &compute_kernel_[beta0], "gen9_gemm_compute",
                    kernel_ctx);
            if (!compute_kernel_[beta0]) return status::runtime_error;
        }

        for (bool outer : {false, true}) {
            compute::kernel_ctx_t kernel_ctx;
            auto trans = !outer ? !pd()->desc()->transa : pd()->desc()->transb;
            auto status = !outer
                    ? gen9_gemm_copy_kernel_t::init_kernel_ctx(kernel_ctx,
                            false, trans, pd()->desc()->a_type,
                            pd()->desc()->acc_type)
                    : gen9_gemm_copy_kernel_t::init_kernel_ctx(kernel_ctx, true,
                            trans, pd()->desc()->b_type,
                            pd()->desc()->acc_type);
            if (status != status::success) return status;

            create_kernel(engine, &copy_kernel_[outer][trans], "gen9_gemm_copy",
                    kernel_ctx);
            if (!copy_kernel_[outer][trans]) return status::runtime_error;
        }

        compute::kernel_ctx_t kernel_ctx;
        auto status = gen9_gemm_beta_kernel_t::init_kernel_ctx(
                kernel_ctx, pd()->desc()->c_type, pd()->desc()->acc_type);
        if (status != status::success) return status;

        create_kernel(engine, &beta_kernel_, "gen9_gemm_beta", kernel_ctx);
        if (!beta_kernel_) return status::runtime_error;

        return status::success;
    }

    status_t init_nocopy(engine_t *engine) {
        const char *kernel_name = nullptr;

        switch (pd()->desc()->c_type) {
            case data_type::f32: kernel_name = "gen9_gemm_nocopy_f32"; break;
            case data_type::f16: kernel_name = "gen9_gemm_nocopy_f16"; break;
            default: return status::unimplemented;
        }

        int unroll_m, unroll_n, unroll_k;
        bool transa = (pd()->desc()->transa == dnnl_trans);
        bool transb = (pd()->desc()->transb == dnnl_trans);

        gen9_gemm_nocopy_kernel_t::get_unrolls(transa, transb, unroll_m,
                unroll_n, unroll_k, pd()->desc()->c_type);

        bool with_k_unroll = pd()->use_nocopy_k_unroll();

        compute::kernel_ctx_t kernel_ctx;

        auto status = gen9_gemm_nocopy_kernel_t::init_kernel_ctx(kernel_ctx,
                pd()->desc()->transa, pd()->desc()->transb, with_k_unroll,
                unroll_k, pd()->attr_info_, pd()->desc()->c_type);
        if (status != status::success) return status;

        create_kernel(engine, &nocopy_kernel_, kernel_name, kernel_ctx);
        if (!nocopy_kernel_) return status::runtime_error;

        return status::success;
    }

    status_t init_nocopy_superkernel(engine_t *engine) {
        if (pd()->desc()->c_type != data_type::f32 || pd()->desc()->transa)
            return status::unimplemented;

        compute::kernel_ctx_t kernel_ctx;

        auto status = gen9_gemm_nocopy_superkernel_t::init_kernel_ctx(
                kernel_ctx, pd()->desc()->transa, pd()->desc()->transb,
                pd()->attr_info_, pd()->desc()->c_type);
        if (status != status::success) return status;

        create_kernel(engine, &nocopy_superkernel_,
                "gen9_gemm_nocopy_superkernel_f32", kernel_ctx);
        if (!nocopy_superkernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    status_t launch_beta(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, int64_t m, int64_t n, float alpha,
            const memory_storage_t &a, int64_t offseta, int64_t lda) const;

    status_t launch_copy(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, int64_t m, int64_t n,
            const memory_storage_t &a, int64_t offseta, int64_t lda,
            float alpha, const memory_storage_t &b, int64_t offsetb, bool outer,
            bool trans) const;

    status_t launch_compute(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, int64_t m, int64_t n, int64_t k,
            const memory_storage_t &base, int32_t offset_a, int32_t offset_b,
            const memory_storage_t &c, int64_t offset_c, int64_t ldc,
            int last_k_block, float eltwise_alpha, float eltwise_beta,
            float eltwise_scale, bool beta0) const;

    status_t launch_nocopy(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, const memory_storage_t &a,
            const memory_storage_t &b, const memory_storage_t &c,
            int64_t offset_a, int64_t offset_b, int64_t offset_c, int32_t lda,
            int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k,
            float alpha, float beta, int last_k_block, float eltwise_alpha,
            float eltwise_beta, float eltwise_scale,
            memory_storage_t &flag) const;

    status_t launch_nocopy_superkernel(const gemm_exec_ctx_t &ctx,
            compute::compute_stream_t *s, const memory_storage_t &plan,
            int32_t threads, const memory_storage_t &a,
            const memory_storage_t &b, const memory_storage_t &c,
            int64_t offset_a, int64_t offset_b, int64_t offset_c, int32_t lda,
            int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k,
            float alpha, float beta, int last_k_block, float eltwise_alpha,
            float eltwise_beta, float eltwise_scale) const;

    virtual status_t execute_standard(const gemm_exec_ctx_t &ctx) const;
    virtual status_t execute_superkernel(const gemm_exec_ctx_t &ctx) const;

    compute::kernel_t compute_kernel_[2];
    compute::kernel_t copy_kernel_[2][2];
    compute::kernel_t beta_kernel_;
    compute::kernel_t nocopy_kernel_;
    compute::kernel_t nocopy_superkernel_;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
