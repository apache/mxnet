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

#include "gpu/ocl/gemm/ref_gemm.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_gemm_t::execute(const gemm_exec_ctx_t &ctx) const {
    const auto &a = GEMM_CTX_ARG_STORAGE(a);
    const auto &b = GEMM_CTX_ARG_STORAGE(b);
    const auto &bias = GEMM_CTX_ARG_STORAGE(bias);
    auto &c = GEMM_CTX_ARG_STORAGE(c);

    const auto exec_d = ctx.desc() ? ctx.desc() : pd()->desc();

    dim_t off_a0 = a.offset() / types::data_type_size(exec_d->a_type);
    dim_t off_b0 = b.offset() / types::data_type_size(exec_d->b_type);
    dim_t off_c0 = c.offset() / types::data_type_size(exec_d->c_type);
    dim_t off_bias0 = pd()->with_bias()
            ? bias.offset() / types::data_type_size(exec_d->bias_type)
            : 0;

    const memory_storage_t *scales = !pd()->attr()->output_scales_.defined()
            ? &GEMM_CTX_ARG_STORAGE(output_scales)
            : &CTX_GPU_RES_STORAGE(SCALES_);
    const memory_storage_t *a0 = !pd()->attr()->zero_points_.defined(DNNL_ARG_A)
            ? &GEMM_CTX_ARG_STORAGE(a_zero_point)
            : &CTX_GPU_RES_STORAGE(A0_);

    const memory_storage_t *b0 = !pd()->attr()->zero_points_.defined(DNNL_ARG_B)
            ? &GEMM_CTX_ARG_STORAGE(b_zero_point)
            : &CTX_GPU_RES_STORAGE(B0_);

    const memory_storage_t *c0 = !pd()->attr()->zero_points_.defined(DNNL_ARG_C)
            ? &GEMM_CTX_ARG_STORAGE(c_zero_point)
            : &CTX_GPU_RES_STORAGE(C0_);

    int c0_mask = 0;
    pd()->attr()->zero_points_.get(DNNL_ARG_C, nullptr, &c0_mask, nullptr);

    const dim_t MB = exec_d->batch;
    const dim_t M = exec_d->m;
    const dim_t N = exec_d->n;
    const dim_t K = exec_d->k;
    const dim_t stride_a = exec_d->stride_a;
    const dim_t stride_b = exec_d->stride_b;
    const dim_t stride_c = exec_d->stride_c;
    const dim_t lda = exec_d->lda;
    const dim_t ldb = exec_d->ldb;
    const dim_t ldc = exec_d->ldc;

    const dim_t scale_stride = pd()->attr()->output_scales_.mask_ == 0 ? 0 : 1;
    const float eltwise_alpha = pd()->attr_info.eltwise_alpha;
    const float eltwise_beta = pd()->attr_info.eltwise_beta;
    const float eltwise_scale = pd()->attr_info.eltwise_scale;
    const int bias_mask = exec_d->bias_mask;
    const float beta = pd()->attr_info.sum_scale;

    const int tra = exec_d->transa == transpose::trans;
    const int trb = exec_d->transb == transpose::trans;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, a);
    arg_list.set(1, b);
    arg_list.set(2, c);
    arg_list.set(3, bias);
    arg_list.set(4, off_a0);
    arg_list.set(5, off_b0);
    arg_list.set(6, off_c0);
    arg_list.set(7, off_bias0);
    arg_list.set(8, tra);
    arg_list.set(9, trb);
    arg_list.set(10, MB);
    arg_list.set(11, M);
    arg_list.set(12, N);
    arg_list.set(13, K);
    arg_list.set(14, stride_a);
    arg_list.set(15, stride_b);
    arg_list.set(16, stride_c);
    arg_list.set(17, lda);
    arg_list.set(18, ldb);
    arg_list.set(19, ldc);
    arg_list.set(20, eltwise_alpha);
    arg_list.set(21, eltwise_beta);
    arg_list.set(22, eltwise_scale);
    arg_list.set(23, bias_mask);
    arg_list.set(24, *a0);
    arg_list.set(25, *b0);
    arg_list.set(26, *c0);
    arg_list.set(27, c0_mask);
    arg_list.set(28, *scales);
    arg_list.set(29, scale_stride);
    arg_list.set(30, beta);

    const size_t gws[3] = {1, (size_t)N, (size_t)MB};
    const auto nd_range = compute::nd_range_t(gws);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
