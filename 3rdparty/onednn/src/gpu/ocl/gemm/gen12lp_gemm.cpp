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

#include "gpu/ocl/gemm/gen12lp_gemm.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen12lp_gemm_driver_params_t {
    static constexpr auto block_m = 2048;
    static constexpr auto block_n = 2048;
    static constexpr auto block_k = 1024;
};

status_t gen12lp_gemm_t::launch_x8x8s32(gemm_exec_ctx_t ctx,
        compute::compute_stream_t *compute_stream, const memory_storage_t &a,
        const memory_storage_t &b, const memory_storage_t &c, int offset_a,
        int offset_b, int offset_c, int lda, int ldb, int ldc, int m, int n,
        int k, int beta, int ao, int bo, const memory_storage_t &co,
        int offset_co, bool apply_co, bool apply_eltwise, float eltwise_alpha,
        float eltwise_beta, float eltwise_scale, bool aligned) const {

    auto &kernel = compute_x8x8s32_kernel_[aligned];
    assert(kernel);

    int unroll_m, unroll_n, block_m, block_n;
    gen12lp_gemm_x8x8s32_kernel_t::get_unrolls(unroll_m, unroll_n);
    block_m = gen12lp_gemm_driver_params_t::block_m;
    block_n = gen12lp_gemm_driver_params_t::block_n;
    int kk = ((k + 3) & ~3);

    int sizea = block_m * (kk + sizeof(int));
    int sizeb = block_n * (kk + sizeof(int));

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, a);
    arg_list.set(1, b);
    arg_list.set(2, c);
    arg_list.set(3, (int)offset_a);
    arg_list.set(4, (int)offset_b);
    arg_list.set(5, (int)offset_c);
    arg_list.set(6, (int)lda);
    arg_list.set(7, (int)ldb);
    arg_list.set(8, (int)ldc);
    arg_list.set(9, (int)m);
    arg_list.set(10, (int)n);
    arg_list.set(11, (int)k);
    arg_list.set(12, (int)beta);
    arg_list.set(13, (int)ao);
    arg_list.set(14, (int)bo);
    arg_list.set(15, co);
    arg_list.set(16, (int)offset_co);
    arg_list.set(17, (int)apply_co);
    arg_list.set(18, sizea, nullptr);
    arg_list.set(19, sizeb, nullptr);
    arg_list.set(20, (int)apply_eltwise);
    arg_list.set(21, eltwise_alpha);
    arg_list.set(22, eltwise_beta);
    arg_list.set(23, eltwise_scale);

    size_t nthreads_x = (m + unroll_m - 1) / unroll_m;
    size_t nthreads_y = (n + unroll_n - 1) / unroll_n;

    size_t lthreads_x = 2;
    size_t lthreads_y = 8;

// TODO: remove DNNL_SYCL_DPCPP from the condition once non-uniform
// work-groups are fixed in the compiler.
#if !defined(CL_VERSION_2_0) || defined(DNNL_SYCL_COMPUTECPP) \
        || defined(DNNL_SYCL_DPCPP)
    while (nthreads_x % lthreads_x)
        lthreads_x--;
    while (nthreads_y % lthreads_y)
        lthreads_y--;
#endif

    static constexpr size_t subgroup_size = 16;

    size_t gws[3] = {nthreads_x * subgroup_size, nthreads_y, 1};
    size_t lws[3] = {lthreads_x * subgroup_size, lthreads_y, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t gen12lp_gemm_t::launch_scale_x8x8s32(gemm_exec_ctx_t ctx,
        compute::compute_stream_t *compute_stream,
        const memory_storage_t &c_temp, const memory_storage_t &c, char offsetc,
        int offset_c, int m, int n, int ldc, float alpha, float beta,
        const memory_storage_t &co, int offset_co, bool alpha_is_zero,
        bool apply_eltwise, float eltwise_alpha, float eltwise_beta,
        float eltwise_scale) const {

    auto &kernel = scale_x8x8s32_kernel_;

    assert(kernel);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, c_temp);
    arg_list.set(1, c);
    arg_list.set(2, (int)offsetc);
    arg_list.set(3, (int)offset_c);
    arg_list.set(4, (int)m);
    arg_list.set(5, (int)n);
    arg_list.set(6, (int)ldc);
    arg_list.set(7, alpha);
    arg_list.set(8, beta);
    arg_list.set(9, co);
    arg_list.set(10, (int)offset_co);
    arg_list.set(11, (int)alpha_is_zero);
    arg_list.set(12, (int)apply_eltwise);
    arg_list.set(13, eltwise_alpha);
    arg_list.set(14, eltwise_beta);
    arg_list.set(15, eltwise_scale);

    int unroll_m, unroll_n;

    gen12lp_gemm_scale_x8x8s32_kernel_t::get_unrolls(unroll_m, unroll_n);

    size_t nthreads_x = (m + unroll_m - 1) / unroll_m;
    size_t nthreads_y = (n + unroll_n - 1) / unroll_n;

    size_t lthreads_x = 16;
    size_t lthreads_y = 1;

    size_t gws[3] = {nthreads_x * lthreads_x, nthreads_y, 1};
    size_t lws[3] = {lthreads_x, lthreads_y, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t gen12lp_gemm_t::execute(const gemm_exec_ctx_t &ctx) const {
    return execute_standard(ctx);
}

status_t gen12lp_gemm_t::execute_standard(const gemm_exec_ctx_t &ctx) const {
    auto a_type = pd()->desc()->a_type;
    auto b_type = pd()->desc()->b_type;
    auto c_type = pd()->desc()->c_type;
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto m = pd()->desc()->m;
    auto n = pd()->desc()->n;
    auto k = pd()->desc()->k;

    bool transa = (pd()->desc()->transa == dnnl_trans);
    bool transb = (pd()->desc()->transb == dnnl_trans);

    int cmask = 0;
    pd()->attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &cmask, nullptr);

    char offsetc_char;

    if (1 << 1 == cmask)
        offsetc_char = 'C';
    else if (1 << 0 == cmask)
        offsetc_char = 'R';
    else
        offsetc_char = 'F';

    auto lda = pd()->desc()->lda;
    auto ldb = pd()->desc()->ldb;
    auto ldc = pd()->desc()->ldc;

    const int *ao_i32 = nullptr;
    const int *bo_i32 = nullptr;
    pd()->attr()->zero_points_.get(DNNL_ARG_SRC, nullptr, nullptr, &ao_i32);
    pd()->attr()->zero_points_.get(DNNL_ARG_WEIGHTS, nullptr, nullptr, &bo_i32);
    auto ao = *ao_i32;
    auto bo = *bo_i32;

    auto alpha = pd()->alpha();
    auto beta = pd()->beta();

    auto eltwise_alpha = pd()->eltwise_alpha();
    auto eltwise_beta = pd()->eltwise_beta();
    auto eltwise_scale = pd()->eltwise_scale();

    auto &a = GEMM_CTX_ARG_STORAGE(a);
    auto &b = GEMM_CTX_ARG_STORAGE(b);
    auto &co = GEMM_CTX_ARG_STORAGE(c_zero_point);
    auto &c = GEMM_CTX_ARG_STORAGE(c);

    auto temp_buf = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_gemm_tmp_buffer);

    int64_t off_a0
            = a.offset() / types::data_type_size(a_type) + pd()->dyn_offset_a;
    int64_t off_b0
            = b.offset() / types::data_type_size(b_type) + pd()->dyn_offset_b;
    int64_t off_c0
            = c.offset() / types::data_type_size(c_type) + pd()->dyn_offset_c;
    int64_t offset_co
            = co.offset() / types::data_type_size(c_type) + pd()->dyn_offset_co;

    bool do_compute = pd()->do_compute();
    bool do_scale = pd()->do_scale();

    status_t status;

    int unroll_m, unroll_n;
    int block_m, block_n, block_k;

    gen12lp_gemm_x8x8s32_kernel_t::get_unrolls(unroll_m, unroll_n);

    block_m = gen12lp_gemm_driver_params_t::block_m;
    block_n = gen12lp_gemm_driver_params_t::block_n;
    block_k = gen12lp_gemm_driver_params_t::block_k;

    bool apply_co = true;
    bool aligned = false;

    int64_t size_k, size_m, size_n;

    if (do_compute) {
        for (int64_t Bk = 0; Bk < k; Bk += size_k) {
            size_k = k - Bk;
            bool apply_eltwise = (size_k <= block_k);
            if (size_k > block_k) size_k = block_k;
            for (int64_t Bm = 0; Bm < m; Bm += size_m) {
                size_m = m - Bm;
                if (size_m > block_m) size_m = block_m;
                auto off_a_src = off_a0
                        + (!transa ? (Bm + Bk * lda) : (Bk + Bm * lda));
                for (int64_t Bn = 0; Bn < n; Bn += size_n) {
                    size_n = n - Bn;
                    if (size_n > block_n) size_n = block_n;
                    auto off_b_src = off_b0
                            + (!transb ? (Bk + Bn * ldb) : (Bn + Bk * ldb));
                    apply_co = !co.is_null() && !(do_scale || (Bk > 0));
                    auto offset_co_src = offset_co
                            + ((offsetc_char == 'C') ? Bm : 0)
                            + ((offsetc_char == 'R') ? Bn : 0);
                    int eff_beta = ((Bk > 0) || (!do_scale && (beta == 1.0f)))
                            ? 1
                            : 0;
                    if (!do_scale) {
                        auto off_c = off_c0 + Bm + Bn * ldc;
                        if ((lda & 3) || (ldb & 3) || (ldc & 3)
                                || (off_a_src & 3) || (off_b_src & 3)
                                || (off_c & 3))
                            aligned = false;
                        else
                            aligned = true;
                        status = launch_x8x8s32(ctx, compute_stream, a, b, c,
                                off_a_src, off_b_src, off_c, lda, ldb, ldc,
                                size_m, size_n, size_k, eff_beta, ao, bo, co,
                                offset_co_src, apply_co, apply_eltwise,
                                eltwise_alpha, eltwise_beta, eltwise_scale,
                                aligned);

                        if (status) return status;
                    } else if (do_scale) {
                        auto off_c = 0 + Bm + Bn * m;
                        if ((lda & 3) || (ldb & 3) || (ldc & 3)
                                || (off_a_src & 3) || (off_b_src & 3)
                                || (off_c & 3))
                            aligned = false;
                        else
                            aligned = true;
                        status = launch_x8x8s32(ctx, compute_stream, a, b,
                                *temp_buf, off_a_src, off_b_src, off_c, lda,
                                ldb, m, size_m, size_n, size_k, eff_beta, ao,
                                bo, co, offset_co_src, apply_co, false,
                                eltwise_alpha, eltwise_beta, eltwise_scale,
                                aligned);
                        if (status) return status;
                    }
                }
            }
        }
    }
    bool alpha_is_zero = false;
    if (do_scale) {
        status = launch_scale_x8x8s32(ctx, compute_stream, *temp_buf, c,
                offsetc_char, off_c0, m, n, ldc, alpha, beta, co, offset_co,
                (int)alpha_is_zero, true, eltwise_alpha, eltwise_beta,
                eltwise_scale);
        if (status) return status;
    }
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
