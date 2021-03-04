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

#ifndef GPU_OCL_SIMPLE_CONCAT_HPP
#define GPU_OCL_SIMPLE_CONCAT_HPP

#include "common/engine.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_concat_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct simple_concat_t : public gpu_primitive_t {
    struct pd_t : public gpu_concat_pd_t {
        using gpu_concat_pd_t::gpu_concat_pd_t;

        DECLARE_CONCAT_PD_T("simple:any", simple_concat_t);

        status_t init(engine_t *engine) {

            bool ok = n_inputs() <= 16 && attr()->has_default_values()
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;

            return init_conf();
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        concat_conf_t conf;
    };

    simple_concat_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        create_kernel(engine, &kernel_, "simple_concat", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_concat(ctx);
    }

private:
    status_t execute_concat(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
