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

#ifndef GPU_OCL_ZERO_PAD_REF_ZERO_PAD_HPP
#define GPU_OCL_ZERO_PAD_REF_ZERO_PAD_HPP

#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_zero_pad_pd.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_zero_pad_t : public gpu_primitive_t {
    struct pd_t : public gpu_zero_pad_pd_t {
        using gpu_zero_pad_pd_t::gpu_zero_pad_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_zero_pad_t);
        status_t init(engine_t *engine) { return status::success; }
    };

    ref_zero_pad_t(const pd_t *apd) : gpu_primitive_t(apd) {};

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        create_kernel(engine, &kernel_, "ref_zero_pad", kernel_ctx);
        if (!kernel_) return status::runtime_error;
        return status::success;
    }
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
