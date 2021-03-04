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

#ifndef GPU_OCL_GEMM_INNER_PRODUCT_HPP
#define GPU_OCL_GEMM_INNER_PRODUCT_HPP

#include <assert.h>
#include <string>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_iterator.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

namespace {
status_t create_gemm_pd(std::unique_ptr<primitive_desc_t> &gemm_pd,
        engine_t *engine, transpose_t transa, transpose_t transb, int m, int n,
        int k, int lda, int ldb, int ldc, data_type_t a_dt, data_type_t b_dt,
        data_type_t c_dt, const primitive_attr_t &attr, dim_t bias_mask = 0,
        bool is_fwd = false) {
    auto gemm_desc = gemm_desc_t();
    gemm_desc.primitive_kind = primitive_kind::gemm;
    gemm_desc.transa = transa;
    gemm_desc.transb = transb;
    gemm_desc.batch = 1;
    gemm_desc.m = m;
    gemm_desc.n = n;
    gemm_desc.k = k;
    gemm_desc.lda = lda;
    gemm_desc.ldb = ldb;
    gemm_desc.ldc = ldc;
    gemm_desc.stride_a = lda;
    gemm_desc.stride_b = ldb;
    gemm_desc.stride_c = ldc;
    gemm_desc.a_type = a_dt;
    gemm_desc.b_type = b_dt;
    gemm_desc.c_type = c_dt;
    gemm_desc.acc_type = c_dt;
    if (bias_mask != 0) {
        gemm_desc.bias_mask = bias_mask;
        gemm_desc.bias_type = c_dt;
    }
    primitive_attr_t gemm_attr(attr);
    if (!gemm_attr.is_initialized()) return status::out_of_memory;
    gemm_attr.set_scratchpad_mode(scratchpad_mode::user);

    dnnl_primitive_desc_iterator it(
            engine, (op_desc_t *)&gemm_desc, &gemm_attr, nullptr);
    if (!it.is_initialized()) return status::out_of_memory;
    ++it;
    gemm_pd.reset(it.fetch_once());
    if (!gemm_pd) return status::unimplemented;
    std::string impl_name(gemm_pd.get()->name());
    if (is_fwd && impl_name.find("ref") != std::string::npos)
        return status::unimplemented;
    return status::success;
}
} // namespace

struct gemm_inner_product_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_inner_product_fwd_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_fwd_pd_t(adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : gpu_inner_product_fwd_pd_t(rhs) {
            gemm_pd_.reset(rhs.gemm_pd_->clone());
            attr_info_ = rhs.attr_info_;
        }
        ~pd_t() = default;

        DECLARE_COMMON_PD_T("ocl:gemm", gemm_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace data_type;

            assert(engine->kind() == engine_kind::gpu);

            attr_info_ = attr_info_t::create(attr());

            bool ok = is_fwd() && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && utils::one_of(true,
                            expect_data_types(f16, f16, f16, f16, f16),
                            expect_data_types(f32, f32, f32, f32, f32))
                    && attr()->post_ops_.len() <= 2
                    && IMPLICATION(attr()->post_ops_.len() == 2,
                            attr()->post_ops_.find(primitive_kind::sum) == 0)
                    && dense_consitency_check(src_md(), weights_md(), dst_md())
                    && dense_gemm_consitency_check(
                            src_md(), weights_md(), dst_md());
            if (!ok) return status::unimplemented;

            const auto &wmd = *this->weights_md();
            bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();
            // bias mask is 2 because batch stride=0, oc stride=1, mb stride=0
            bool gemm_ok = status::success
                    == create_gemm_pd(gemm_pd_, engine,
                            wei_tr ? transpose::trans : transpose::notrans,
                            transpose::notrans, oc, mb, ic_total,
                            wei_tr ? ic_total : oc, ic_total, oc,
                            weights_md()->data_type, src_md()->data_type,
                            dst_md()->data_type, *attr(), with_bias() ? 2 : 0,
                            true);
            if (!gemm_ok) return status::unimplemented;
            init_scratchpad();

            return status::success;
        }

        attr_info_t attr_info_ = {};
        std::unique_ptr<primitive_desc_t> gemm_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    gemm_inner_product_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(gemm_, engine);
        if (gemm_status != status::success) return gemm_status;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {gemm_.get()};
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<primitive_t> gemm_;
};

struct gemm_inner_product_bwd_data_t : public gpu_primitive_t {
    struct pd_t : public gpu_inner_product_bwd_data_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : gpu_inner_product_bwd_data_pd_t(rhs) {
            gemm_pd_.reset(rhs.gemm_pd_->clone());
        }
        ~pd_t() = default;

        DECLARE_COMMON_PD_T("ocl:gemm", gemm_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            assert(engine->kind() == engine_kind::gpu);

            bool ok = this->desc()->prop_kind == backward_data
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && expect_data_types(f32, f32, data_type::undef, f32, f32)
                    && attr()->has_default_values()
                    && dense_consitency_check(
                            diff_src_md(), weights_md(), diff_dst_md())
                    && dense_gemm_consitency_check(
                            diff_src_md(), weights_md(), diff_dst_md());
            if (!ok) return status::unimplemented;

            const auto &wmd = *this->weights_md();
            bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = status::success
                    == create_gemm_pd(gemm_pd_, engine,
                            wei_tr ? transpose::trans : transpose::notrans,
                            transpose::notrans, ic_total, mb, oc,
                            wei_tr ? oc : ic_total, oc, ic_total,
                            weights_md()->data_type, diff_src_md()->data_type,
                            diff_dst_md()->data_type, *attr());
            if (!gemm_ok) return status::unimplemented;
            init_scratchpad();

            return status::success;
        }

        std::unique_ptr<primitive_desc_t> gemm_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    gemm_inner_product_bwd_data_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(gemm_, engine);
        return gemm_status;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {gemm_.get()};
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<primitive_t> gemm_;
};

struct gemm_inner_product_bwd_weights_t : public gpu_primitive_t {
    using gpu_ip_bwd_weights_pd_t = gpu_inner_product_bwd_weights_pd_t;
    struct pd_t : public gpu_ip_bwd_weights_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_ip_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : gpu_ip_bwd_weights_pd_t(rhs) {
            gemm_pd_.reset(rhs.gemm_pd_->clone());
        }
        ~pd_t() = default;

        DECLARE_COMMON_PD_T("gemm:ocl", gemm_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            assert(engine->kind() == engine_kind::gpu);

            bool ok = this->desc()->prop_kind == backward_weights
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && expect_data_types(f32, f32, f32, f32, f32)
                    && attr()->has_default_values()
                    && dense_consitency_check(
                            src_md(), diff_weights_md(), diff_dst_md())
                    && dense_gemm_consitency_check(
                            src_md(), diff_weights_md(), diff_dst_md());
            if (!ok) return status::unimplemented;

            const int mb = this->MB();
            const int oc = this->OC();
            const int ic_total = this->IC_total_padded();

            bool gemm_ok = false;
            if (wei_tr()) {
                gemm_ok = create_gemm_pd(gemm_pd_, engine, transpose::notrans,
                                  transpose::trans, oc, ic_total, mb, oc,
                                  ic_total, oc, src_md()->data_type,
                                  src_md()->data_type, src_md()->data_type,
                                  *attr())
                        == status::success;
            } else {
                gemm_ok = create_gemm_pd(gemm_pd_, engine, transpose::notrans,
                                  transpose::trans, ic_total, oc, mb, ic_total,
                                  oc, ic_total, src_md()->data_type,
                                  src_md()->data_type, src_md()->data_type,
                                  *attr())
                        == status::success;
            }
            if (!gemm_ok) return status::unimplemented;
            init_scratchpad();

            return status::success;
        }

        bool wei_tr() const {
            const auto &wmd = *this->diff_weights_md();
            return wmd.format_desc.blocking.strides[0] == 1;
        }

        std::unique_ptr<primitive_desc_t> gemm_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    gemm_inner_product_bwd_weights_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(gemm_, engine);
        if (gemm_status != status::success) return gemm_status;

        if (pd()->with_bias()) {
            compute::kernel_ctx_t kernel_ctx;

            kernel_ctx.set_data_type(pd()->src_md()->data_type);
            kernel_ctx.define_int("MB", pd()->MB());
            kernel_ctx.define_int("OC", pd()->OC());

            create_kernel(engine, &bias_kernel_,
                    "gemm_inner_product_backward_weights_bias", kernel_ctx);
            if (!bias_kernel_) return status::runtime_error;
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {gemm_.get()};
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> gemm_;
    compute::kernel_t bias_kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
