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

#ifndef GPU_OCL_GEN9_BINARY_HPP
#define GPU_OCL_GEN9_BINARY_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_binary_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_binary_t : public gpu_primitive_t {
    struct pd_t : public gpu_binary_pd_t {
        using gpu_binary_pd_t::gpu_binary_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9", gen9_binary_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = sm::post_ops | sm::scales;

            bool ok = set_default_params() == status::success && is_broadcast()
                    && (utils::everyone_is(f32, src_md(0)->data_type,
                                src_md(1)->data_type, dst_md()->data_type)
                            || utils::everyone_is(bf16, src_md(0)->data_type,
                                    src_md(1)->data_type, dst_md()->data_type)
                            || utils::everyone_is(f16, src_md(0)->data_type,
                                    src_md(1)->data_type, dst_md()->data_type)
                            || utils::one_of(src_md(0)->data_type, s8, u8))
                    && (src_md(0)->data_type == dst_md()->data_type)
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            utils::one_of(src_md(0)->data_type, s8, u8)
                                    && utils::one_of(
                                            attr()->output_scales_.mask_, 0,
                                            1 << 1))
                    && attr()->has_default_values(attr_skip_mask)
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(
                            utils::one_of(f16, src_md(1)->data_type,
                                    src_md(0)->data_type, dst_md()->data_type),
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short))
                    && post_ops_with_binary_ok(attr(), dst_md()->data_type);

            if (!ok) return status::unimplemented;

            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);

        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        bool is_broadcast() {
            auto bcast_dims = broadcast_dims();
            for (int i = 0; i < src_md(0)->ndims; ++i) {
                if (bcast_dims[i] != 0) { return true; }
            }
            return false;
        }

        binary_conf_t conf;
    };

    gen9_binary_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        create_kernel(engine, &kernel_, "gen9_binary", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {

        auto &src0 = CTX_IN_STORAGE(DNNL_ARG_SRC_0);
        auto &src1 = CTX_IN_STORAGE(DNNL_ARG_SRC_1);
        auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

        const auto &conf = pd()->conf;

        auto src0_scale = conf.attr_info.src0_scale;
        auto src1_scale = conf.attr_info.src1_scale;

        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src0);
        arg_list.set(1, src1);
        arg_list.set(2, dst);

        unsigned arg_idx = append_post_ops_to_arg_list(
                ctx, arg_list, 3, conf.attr_info.all_post_ops);

        arg_list.set(arg_idx++, src0_scale);
        arg_list.set(arg_idx, src1_scale);

        auto nd_range = conf.dispatch.nd_range();

        status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
        return status;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
