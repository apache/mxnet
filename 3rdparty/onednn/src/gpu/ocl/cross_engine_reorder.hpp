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

#ifndef GPU_OCL_CROSS_ENGINE_REORDER_HPP
#define GPU_OCL_CROSS_ENGINE_REORDER_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Cross-engine reorder manages all reorders between GPU and CPU engines.
//
// For CPU -> GPU reorder, it includes 2 steps:
// 1. CPU -> GPU copying
// 2. GPU reorder
//
// For GPU -> CPU reorder, it includes 2 steps:
// 1. GPU reorder
// 2. GPU -> CPU copying
struct cross_engine_reorder_t : public gpu_primitive_t {
    struct pd_t : public reorder_pd_t {
        using reorder_pd_t::reorder_pd_t;

        pd_t(const pd_t &rhs)
            : reorder_pd_t(rhs)
            , reorder_pd_(rhs.do_reorder_ ? rhs.reorder_pd_->clone() : nullptr)
            , reorder_engine_kind_(rhs.reorder_engine_kind_)
            , do_reorder_(rhs.do_reorder_) {}

        DECLARE_COMMON_PD_T("ocl:cross_engine::any", cross_engine_reorder_t);

        DECLARE_GPU_REORDER_CREATE();

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine);

        std::unique_ptr<primitive_desc_t> reorder_pd_;
        engine_kind_t reorder_engine_kind_ = engine_kind::gpu;
        bool do_reorder_ = true;

    private:
        void init_scratchpad();
    };

    cross_engine_reorder_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        if (!pd()->do_reorder_) return status::success;
        auto status = pd()->reorder_pd_->create_primitive(reorder_, engine);
        return status;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    primitive_list_t nested_primitives() const override {
        return {reorder_.get()};
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> reorder_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
