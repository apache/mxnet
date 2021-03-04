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

#include "gpu/ocl/gemm_matmul.hpp"

#include "gpu/gemm/gpu_gemm_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t gemm_matmul_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    using namespace gemm_utils;

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC);
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS);
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST);
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS);

    memory_storage_t *scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_OUTPUT_SCALES);
    memory_storage_t *a0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);

    memory_storage_t *b0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    memory_storage_t *c0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const bool is_batched = src_d.ndims() == 3;
    const dim_t MB = is_batched ? dst_d.dims()[0] : 1;
    const dim_t M = dst_d.dims()[is_batched + 0];
    const dim_t N = dst_d.dims()[is_batched + 1];
    const dim_t K = src_d.dims()[is_batched + 1];

    const auto &dst_bd = dst_d.blocking_desc();
    const auto &src_strides = &src_d.blocking_desc().strides[0];
    const auto &weights_strides = &weights_d.blocking_desc().strides[0];
    const auto &dst_strides = &dst_d.blocking_desc().strides[0];

    int bias_mask = 0;
    if (is_batched) bias_mask |= (bia_d.dims()[0] > 1) ? 1 << 0 : 0;
    for (int d = is_batched; d < bia_d.ndims(); ++d) {
        bias_mask |= (bia_d.dims()[d] > 1) ? 1 << (bia_d.ndims() - d) : 0;
    }

    const transpose_t transA = src_strides[is_batched + 1] == 1
                    && src_d.dims()[is_batched + 0] > 1
            ? transpose::notrans
            : transpose::trans;
    const transpose_t transB = weights_strides[is_batched + 1] == 1
                    && weights_d.dims()[is_batched + 0] > 1
            ? transpose::notrans
            : transpose::trans;

    const int lda = (int)
            src_strides[is_batched + (transA == transpose::notrans ? 0 : 1)];
    const int ldb = (int)weights_strides[is_batched
            + (transB == transpose::notrans ? 0 : 1)];
    const int ldc = (int)dst_bd.strides[is_batched + 0];

    const auto d = pd()->desc();
    const auto src_dt = d->src_desc.data_type;
    const auto wei_dt = d->weights_desc.data_type;
    const auto bia_dt = d->bias_desc.data_type;
    const auto dst_dt = d->dst_desc.data_type;
    const auto acc_dt = d->accum_data_type;

    const int stride_a = (int)src_strides[0];
    const int stride_b = (int)weights_strides[0];
    const int stride_c = (int)dst_strides[0];

    gemm_exec_args_t gemm_args;
    gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_SRC);
    gemm_args.c = &CTX_OUT_STORAGE(DNNL_ARG_DST);
    gemm_args.bias = &CTX_IN_STORAGE(DNNL_ARG_BIAS);

    gemm_args.a_zero_point = b0;
    gemm_args.b_zero_point = a0;
    gemm_args.c_zero_point = c0;
    gemm_args.output_scales = scales;

    auto gemm_desc = gemm_desc_t();
    gemm_desc.primitive_kind = primitive_kind::gemm;
    gemm_desc.transa = transB;
    gemm_desc.transb = transA;
    gemm_desc.batch = MB;
    gemm_desc.m = N;
    gemm_desc.n = M;
    gemm_desc.k = K;
    gemm_desc.stride_a = stride_b;
    gemm_desc.stride_b = stride_a;
    gemm_desc.stride_c = stride_c;
    gemm_desc.lda = ldb;
    gemm_desc.ldb = lda;
    gemm_desc.ldc = ldc;
    gemm_desc.bias_mask = bias_mask;
    gemm_desc.a_type = wei_dt;
    gemm_desc.b_type = src_dt;
    gemm_desc.c_type = dst_dt;
    gemm_desc.acc_type = acc_dt;
    gemm_desc.bias_type = bia_dt;

    gemm_exec_ctx_t gemm_ctx(ctx, gemm_args, &gemm_desc);

    nested_scratchpad_t ns(ctx, key_nested, gemm_);
    gemm_ctx.set_scratchpad_grantor(ns.grantor());

    status_t gemm_exec_status = gpu_gemm(gemm_)->execute(gemm_ctx);
    if (gemm_exec_status != status::success) return gemm_exec_status;

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
