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

#include "gpu/ocl/ref_matmul.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    const auto &a = CTX_IN_STORAGE(DNNL_ARG_SRC);
    const auto &b = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    const auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    auto &c = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const memory_storage_t *scales = !pd()->attr()->output_scales_.defined()
            ? &CTX_IN_STORAGE(DNNL_ARG_ATTR_OUTPUT_SCALES)
            : &CTX_GPU_RES_STORAGE(SCALES_);
    const memory_storage_t *a0
            = !pd()->attr()->zero_points_.defined(DNNL_ARG_SRC)
            ? &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC)
            : &CTX_GPU_RES_STORAGE(A0_);
    const memory_storage_t *b0
            = !pd()->attr()->zero_points_.defined(DNNL_ARG_WEIGHTS)
            ? &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS)
            : &CTX_GPU_RES_STORAGE(B0_);
    const memory_storage_t *c0
            = !pd()->attr()->zero_points_.defined(DNNL_ARG_DST)
            ? &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST)
            : &CTX_GPU_RES_STORAGE(C0_);

    const auto a_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto b_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto c_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));
    const bool is_batched = pd()->batched();

    dim_t a_stride_mb, a_stride_m, a_stride_k;
    const auto &a_strides = a_d.blocking_desc().strides;
    a_stride_mb = is_batched && a_d.dims()[0] > 1 ? a_strides[0] : 0;
    a_stride_m = a_strides[is_batched + 0];
    a_stride_k = a_strides[is_batched + 1];

    dim_t b_stride_mb, b_stride_k, b_stride_n;
    const auto &b_strides = b_d.blocking_desc().strides;
    b_stride_mb = is_batched && b_d.dims()[0] > 1 ? b_strides[0] : 0;
    b_stride_k = b_strides[is_batched + 0];
    b_stride_n = b_strides[is_batched + 1];

    dim_t c_stride_mb, c_stride_m, c_stride_n;
    const auto &c_strides = c_d.blocking_desc().strides;
    c_stride_mb = is_batched && c_d.dims()[0] > 1 ? c_strides[0] : 0;
    c_stride_m = c_strides[is_batched + 0];
    c_stride_n = c_strides[is_batched + 1];

    dim_t bia_stride_mb = 0, bia_stride_m = 0, bia_stride_n = 0;
    if (bia_d.data_type() != data_type::undef) {
        const auto &bia_strides = bia_d.blocking_desc().strides;
        bia_stride_mb = is_batched && bia_d.dims()[0] > 1 ? bia_strides[0] : 0;
        bia_stride_m = bia_d.dims()[is_batched + 0] > 1
                ? bia_strides[is_batched + 0]
                : 0;
        bia_stride_n = bia_d.dims()[is_batched + 1] > 1
                ? bia_strides[is_batched + 1]
                : 0;
    }

    const dim_t MB = is_batched ? c_d.dims()[0] : 1;
    const dim_t M = c_d.dims()[is_batched + 0];
    const dim_t N = c_d.dims()[is_batched + 1];
    const dim_t K = a_d.dims()[is_batched + 1];

    const dim_t scale_stride = pd()->attr()->output_scales_.mask_ == 0 ? 0 : 1;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, a);
    arg_list.set(1, b);
    arg_list.set(2, c);
    arg_list.set(3, bias);
    arg_list.set(4, *a0);
    arg_list.set(5, *b0);
    arg_list.set(6, *c0);
    arg_list.set(7, *scales);
    arg_list.set(8, scale_stride);
    arg_list.set(9, MB);
    arg_list.set(10, M);
    arg_list.set(11, N);
    arg_list.set(12, K);
    arg_list.set(13, bia_stride_mb);
    arg_list.set(14, bia_stride_m);
    arg_list.set(15, bia_stride_n);
    arg_list.set(16, a_stride_mb);
    arg_list.set(17, a_stride_m);
    arg_list.set(18, a_stride_k);
    arg_list.set(19, b_stride_mb);
    arg_list.set(20, b_stride_k);
    arg_list.set(21, b_stride_n);
    arg_list.set(22, c_stride_mb);
    arg_list.set(23, c_stride_m);
    arg_list.set(24, c_stride_n);

    append_post_ops_to_arg_list(
            ctx, arg_list, 25, pd()->attr_info_.all_post_ops);

    size_t gws[3] = {1, (size_t)N, (size_t)MB};
    auto nd_range = compute::nd_range_t(gws);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
