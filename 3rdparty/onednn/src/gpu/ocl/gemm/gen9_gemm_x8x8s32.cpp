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

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/type_helpers.hpp"

#include "gpu/ocl/gemm/gen9_gemm_x8x8s32.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_gemm_x8x8s32_driver_params_t {
    //unroll_m = 32, unroll_n = 16
    static constexpr auto block_m = 6 * 32;
    static constexpr auto block_n = 4 * 16;
    static constexpr auto block_k
            = ((32768 / ((block_m >= block_n) ? block_m : block_n)
                       - sizeof(int))
                    & ~3);
};

status_t gen9_gemm_x8x8s32_t::launch_x8x8s32(const gemm_exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, const memory_storage_t &a,
        const memory_storage_t &b, const memory_storage_t &c, int64_t offset_a,
        int64_t offset_b, int64_t offset_c, int64_t lda, int64_t ldb,
        int64_t ldc, int64_t m, int64_t n, int64_t k, int64_t beta, int32_t ao,
        int32_t bo, const memory_storage_t &co, int64_t offset_co,
        bool apply_co, bool apply_eltwise, float eltwise_alpha,
        float eltwise_beta, float eltwise_scale) const {

    int unroll_m, unroll_n, block_m, block_n;
    gen9_gemm_x8x8s32_kernel_t::get_unrolls(unroll_m, unroll_n);
    block_m = gen9_gemm_x8x8s32_driver_params_t::block_m;
    block_n = gen9_gemm_x8x8s32_driver_params_t::block_n;
    int kk = ((k + 3) & ~3);

    int sizea = block_m * (kk + sizeof(int));
    int sizeb = block_n * (kk + sizeof(int));

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, a);
    arg_list.set(1, b);
    arg_list.set(2, c);
    arg_list.set(3, offset_a);
    arg_list.set(4, offset_b);
    arg_list.set(5, offset_c);
    arg_list.set(6, lda);
    arg_list.set(7, ldb);
    arg_list.set(8, ldc);
    arg_list.set(9, m);
    arg_list.set(10, n);
    arg_list.set(11, k);
    arg_list.set(12, (int)beta);
    arg_list.set(13, ao);
    arg_list.set(14, bo);
    arg_list.set(15, co);
    arg_list.set(16, offset_co);
    arg_list.set(17, (int)apply_co);
    arg_list.set(18, sizea, nullptr);
    arg_list.set(19, sizeb, nullptr);
    arg_list.set(20, (int)apply_eltwise);
    arg_list.set(21, eltwise_alpha);
    arg_list.set(22, eltwise_beta);
    arg_list.set(23, eltwise_scale);

    size_t nthreads_x = (m + block_m - 1) / block_m;
    size_t nthreads_y = (n + block_n - 1) / block_n;

    int GRX = 8;

    size_t lthreads0 = GRX; //8
    size_t lthreads1 = block_m / unroll_m; //6
    size_t lthreads2 = block_n / unroll_n; //4

    size_t gthreads0 = lthreads0; //8
    size_t gthreads1 = lthreads1 * nthreads_x;
    size_t gthreads2 = lthreads2 * nthreads_y;

    size_t gws[3] = {gthreads0, gthreads1, gthreads2};
    size_t lws[3] = {lthreads0, lthreads1, lthreads2};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, compute_x8x8s32_kernel_, arg_list);
}

status_t gen9_gemm_x8x8s32_t::launch_scale_x8x8s32(const gemm_exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream,
        const memory_storage_t &c_temp, const memory_storage_t &c, char offsetc,
        int64_t offset_c, int64_t m, int64_t n, int64_t ldc, float alpha,
        float beta, const memory_storage_t &co, int64_t offset_co,
        bool alpha_is_zero, bool apply_eltwise, float eltwise_alpha,
        float eltwise_beta, float eltwise_scale) const {

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, c_temp);
    arg_list.set(1, c);
    arg_list.set(2, (int8_t)offsetc);
    arg_list.set(3, offset_c);
    arg_list.set(4, m);
    arg_list.set(5, n);
    arg_list.set(6, ldc);
    arg_list.set(7, alpha);
    arg_list.set(8, beta);
    arg_list.set(9, co);
    arg_list.set(10, offset_co);
    arg_list.set(11, (int)alpha_is_zero);
    arg_list.set(12, (int)apply_eltwise);
    arg_list.set(13, eltwise_alpha);
    arg_list.set(14, eltwise_beta);
    arg_list.set(15, eltwise_scale);

    int unroll_m, unroll_n;

    gen9_gemm_scale_x8x8s32_kernel_t::get_unrolls(unroll_m, unroll_n);

    size_t nthreads_x = (m + unroll_m - 1) / unroll_m;
    size_t nthreads_y = (n + unroll_n - 1) / unroll_n;

    size_t lthreads_x = 16;
    size_t lthreads_y = 1;

    size_t gws[3] = {nthreads_x * lthreads_x, nthreads_y, 1};
    size_t lws[3] = {lthreads_x, lthreads_y, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, scale_x8x8s32_kernel_, arg_list);
}

status_t gen9_gemm_x8x8s32_t::execute(const gemm_exec_ctx_t &ctx) const {
    return execute_standard(ctx);
}

status_t gen9_gemm_x8x8s32_t::execute_standard(
        const gemm_exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
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

    auto eltwise_alpha = pd()->attr_info.eltwise_alpha;
    auto eltwise_beta = pd()->attr_info.eltwise_beta;
    auto eltwise_scale = pd()->attr_info.eltwise_scale;

    auto &a = GEMM_CTX_ARG_STORAGE(a);
    auto &b = GEMM_CTX_ARG_STORAGE(b);
    auto &c = GEMM_CTX_ARG_STORAGE(c);
    auto &co = GEMM_CTX_ARG_STORAGE(c_zero_point);

    size_t off_a0
            = a.offset() / types::data_type_size(a_type) + pd()->dyn_offset_a;
    size_t off_b0
            = b.offset() / types::data_type_size(b_type) + pd()->dyn_offset_b;
    size_t off_c0
            = c.offset() / types::data_type_size(c_type) + pd()->dyn_offset_c;
    size_t offset_co
            = co.offset() / types::data_type_size(c_type) + pd()->dyn_offset_co;

    bool do_compute = ((k > 0) && (alpha != 0.0f));
    bool do_scale = !(
            (k > 0) && (alpha == 1.0f) && ((beta == 0.0f) || (beta == 1.0f)));

    status_t status;

    int unroll_m, unroll_n;
    int block_m, block_n, block_k;
    int slices;

    gen9_gemm_x8x8s32_kernel_t::get_unrolls(unroll_m, unroll_n);

    block_m = gen9_gemm_x8x8s32_driver_params_t::block_m;
    block_n = gen9_gemm_x8x8s32_driver_params_t::block_n;
    block_k = gen9_gemm_x8x8s32_driver_params_t::block_k;

    slices = eu_count_ / 24; //24EUs per slice

    bool apply_co = true;
    int64_t size_k, size_m, size_n;

    std::unique_ptr<memory_storage_t> acc;
    if (do_scale)
        acc = ctx.get_scratchpad_grantor().get_memory_storage(
                key_gemm_int_c_in_acc_dt);

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
                    if (size_n > block_n * slices) size_n = block_n * slices;
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
                        status = launch_x8x8s32(ctx, compute_stream, a, b, c,
                                off_a_src, off_b_src, off_c, lda, ldb, ldc,
                                size_m, size_n, size_k, eff_beta, ao, bo, co,
                                offset_co_src, (int)apply_co,
                                (int)apply_eltwise, eltwise_alpha, eltwise_beta,
                                eltwise_scale);
                        if (status) return status;
                    } else if (do_scale) {
                        auto off_c = 0 + Bm + Bn * m;
                        status = launch_x8x8s32(ctx, compute_stream, a, b, *acc,
                                off_a_src, off_b_src, off_c, lda, ldb, m,
                                size_m, size_n, size_k, eff_beta, ao, bo, co,
                                offset_co_src, apply_co, false, eltwise_alpha,
                                eltwise_beta, eltwise_scale);
                        if (status) return status;
                    }
                }
            }
        }
    }
    bool alpha_is_zero = false;
    if (do_scale) {
        status = launch_scale_x8x8s32(ctx, compute_stream, *acc, c,
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
