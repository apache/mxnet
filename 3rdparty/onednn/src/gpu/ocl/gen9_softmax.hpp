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

#ifndef GPU_OCL_GEN9_SOFTMAX_HPP
#define GPU_OCL_GEN9_SOFTMAX_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_softmax_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_softmax_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_softmax_fwd_pd_t {
        pd_t(const softmax_desc_t *adesc, const primitive_attr_t *attr,
                const softmax_fwd_pd_t *hint_fwd_pd)
            : gpu_softmax_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen9", gen9_softmax_fwd_t);

        status_t init(engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const int nelems = desc()->data_desc.dims[desc()->softmax_axis];
            const memory_desc_wrapper src_d(src_md());
            bool ok = true && nelems % 128 == 0
                    && desc()->softmax_axis == src_md()->ndims - 1
                    && src_d.is_plain()
                    && utils::one_of(desc()->prop_kind,
                            prop_kind::forward_inference,
                            prop_kind::forward_training)
                    && utils::one_of(desc()->data_desc.data_type,
                            data_type::f32, data_type::f16, data_type::bf16)
                    && IMPLICATION(
                            desc()->data_desc.data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            gws[0] = gws[1] = gws[2] = 1;
            lws[0] = lws[1] = lws[2] = 1;
            block[0] = block[1] = block[2] = 1;

            for (int i = 0, j = 0; i < src_md()->ndims; ++i) {
                if (i != desc()->softmax_axis) {
                    const auto dim = src_md()->dims[i];
                    gws[j % 3] *= dim;
                    if (j < 3) block[j] = dim;
                    ++j;
                }
            }

            group_size = 16;

            lws[0] = group_size;
            gws[0] *= group_size;

            return status::success;
        }

        size_t gws[3] = {};
        size_t lws[3] = {};
        size_t block[3] = {};
        size_t group_size = 0;
    };

    gen9_softmax_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        if (memory_desc_wrapper(pd()->desc()->data_desc).has_zero_dim())
            return status::success;

        compute::kernel_ctx_t kernel_ctx;

        const auto *desc = pd()->desc();
        kernel_ctx.define_int("SOFTMAX_AXIS_IDX", desc->softmax_axis);
        kernel_ctx.define_int(
                "SOFTMAX_AXIS_SIZE", desc->data_desc.dims[desc->softmax_axis]);
        kernel_ctx.define_int("GROUP_SIZE", pd()->group_size);
        kernel_ctx.define_int("SUB_GROUP_SIZE", pd()->group_size);
        kernel_ctx.define_int("IS_FWD", 1);
        kernel_ctx.add_option("-cl-std=CL2.0");
        kernel_ctx.define_int("LOGSOFTMAX",
                desc->primitive_kind == primitive_kind::logsoftmax ? 1 : 0);

        kernel_ctx.set_data_type(desc->data_desc.data_type);
        set_offsets(kernel_ctx, pd()->dst_md(), "DATA");

        for (int i = 0; i < 3; ++i)
            kernel_ctx.define_int(utils::format("BLOCK_%d", i), pd()->block[i]);

        create_kernel(engine, &kernel_, "gen9_softmax_fwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_generic(ctx);
    }

protected:
    status_t execute_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
