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

#include "gpu/ocl/gemm_inner_product.hpp"

#include "gpu/gemm/gpu_gemm_utils.hpp"
#include "gpu/ocl/ocl_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gemm_inner_product_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    using namespace gemm_utils;

    gemm_exec_args_t gemm_args;
    gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_SRC);
    gemm_args.c = &CTX_OUT_STORAGE(DNNL_ARG_DST);
    gemm_args.bias = &CTX_IN_STORAGE(DNNL_ARG_BIAS);
    memory_storage_t *scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_OUTPUT_SCALES);
    memory_storage_t *a0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);

    memory_storage_t *b0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    memory_storage_t *c0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    gemm_args.a_zero_point = b0;
    gemm_args.b_zero_point = a0;
    gemm_args.c_zero_point = c0;
    gemm_args.output_scales = scales;

    gemm_exec_ctx_t gemm_ctx(ctx, gemm_args);

    nested_scratchpad_t ns(ctx, key_nested, gemm_);
    gemm_ctx.set_scratchpad_grantor(ns.grantor());

    status_t gemm_exec_status = gpu_gemm(gemm_)->execute(gemm_ctx);

    if (gemm_exec_status != status::success) return gemm_exec_status;

    return status::success;
}

status_t gemm_inner_product_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    using namespace gemm_utils;

    gemm_exec_args_t gemm_args;
    gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    gemm_args.c = &CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    gemm_exec_ctx_t gemm_ctx(ctx, gemm_args);

    nested_scratchpad_t ns(ctx, key_nested, gemm_);
    gemm_ctx.set_scratchpad_grantor(ns.grantor());

    status_t gemm_exec_status = gpu_gemm(gemm_)->execute(gemm_ctx);
    if (gemm_exec_status != status::success) return gemm_exec_status;

    return status::success;
}

status_t gemm_inner_product_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    using namespace gemm_utils;

    gemm_exec_args_t gemm_args;
    if (pd()->wei_tr()) {
        gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
        gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_SRC);
    } else {
        gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_SRC);
        gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    }
    gemm_args.c = &CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);

    gemm_exec_ctx_t gemm_ctx(ctx, gemm_args);

    nested_scratchpad_t ns(ctx, key_nested, gemm_);
    gemm_ctx.set_scratchpad_grantor(ns.grantor());

    status_t gemm_exec_status = gpu_gemm(gemm_)->execute(gemm_ctx);
    if (gemm_exec_status != status::success) return gemm_exec_status;

    if (pd()->with_bias()) {
        auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
        auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, diff_dst);
        arg_list.set(1, diff_bias);

        auto nd_range = compute::nd_range_t({pd()->OC()});

        status_t bias_status
                = parallel_for(ctx, nd_range, bias_kernel_, arg_list);
        if (bias_status != status::success) return bias_status;
    }

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
