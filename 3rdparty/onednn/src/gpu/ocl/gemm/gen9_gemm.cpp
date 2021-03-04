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

#include "gpu/ocl/gemm/gen9_gemm.hpp"
#include "gpu/gpu_resource.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/float16.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct driver_params_f32_copy_t {
    static constexpr int block_m = 512 * 16;
    static constexpr int block_n = 64 * 32;
    static constexpr int block_k = 1024;
};

struct driver_params_f16_copy_t {
    static constexpr int block_m = 512 * 16;
    static constexpr int block_n = 64 * 32;
    static constexpr int block_k = 2048;
};

struct driver_params_f32_nocopy_t {
    static constexpr int block_m = 4096;
    static constexpr int block_n = 2048;
    static constexpr int block_k = 2048;
};

static_assert(sizeof(plan_element_t) == 8,
        "Plan element structure has been padded by the compiler.");

status_t gen9_gemm_t::launch_beta(const gemm_exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, int64_t m, int64_t n,
        float alpha, const memory_storage_t &a, int64_t offset_a,
        int64_t lda) const {

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, m);
    arg_list.set(1, n);
    arg_list.set(2, alpha);
    arg_list.set(3, a);
    arg_list.set(4, offset_a);
    arg_list.set(5, lda);

    size_t gws[3] = {1, size_t(n), 1};
    size_t lws[3] = {1, 1, 1};
    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, beta_kernel_, arg_list);
}

status_t gen9_gemm_t::launch_copy(const gemm_exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, int64_t x, int64_t y,
        const memory_storage_t &a, int64_t offset_a, int64_t lda, float alpha,
        const memory_storage_t &b, int64_t offset_b, bool outer,
        bool trans) const {

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, x);
    arg_list.set(1, y);
    arg_list.set(2, a);
    arg_list.set(3, offset_a);
    arg_list.set(4, lda);
    arg_list.set(5, alpha);
    arg_list.set(6, b);
    arg_list.set(7, offset_b);

    int unroll_m, unroll_n;
    gen9_gemm_compute_kernel_t::get_unrolls(unroll_m, unroll_n);

    auto unroll = outer ? unroll_n : unroll_m;

    size_t gws[3] = {size_t(x), size_t((y + unroll - 1) / unroll), 1};
    size_t lws[3] = {1, 1, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, copy_kernel_[outer][trans], arg_list);
}

status_t gen9_gemm_t::launch_compute(const gemm_exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, int64_t m, int64_t n,
        int64_t k, const memory_storage_t &base, int32_t offset_a,
        int32_t offset_b, const memory_storage_t &c, int64_t offset_c,
        int64_t ldc, int last_k_block, float eltwise_alpha, float eltwise_beta,
        float eltwise_scale, bool beta0) const {

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, m);
    arg_list.set(1, n);
    arg_list.set(2, k);
    arg_list.set(3, base);
    arg_list.set(4, offset_a);
    arg_list.set(5, offset_b);
    arg_list.set(6, c);
    arg_list.set(7, offset_c);
    arg_list.set(8, ldc);
    arg_list.set(9, last_k_block);
    arg_list.set(10, eltwise_alpha);
    arg_list.set(11, eltwise_beta);
    arg_list.set(12, eltwise_scale);

    int unroll_m, unroll_n;
    gen9_gemm_compute_kernel_t::get_unrolls(unroll_m, unroll_n);

    int nthreads_x = (m + unroll_m - 1) / unroll_m;
    int nthreads_y = (n + unroll_n - 1) / unroll_n;

    int lws_y = 8;
    while (nthreads_y % lws_y)
        lws_y--;

    size_t gws[3] = {size_t(nthreads_x) * 8, size_t(nthreads_y), 1};
    size_t lws[3] = {8, size_t(lws_y), 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, compute_kernel_[beta0], arg_list);
}

status_t gen9_gemm_t::launch_nocopy(const gemm_exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, const memory_storage_t &a,
        const memory_storage_t &b, const memory_storage_t &c, int64_t offset_a,
        int64_t offset_b, int64_t offset_c, int32_t lda, int32_t ldb,
        int32_t ldc, int32_t m, int32_t n, int32_t k, float alpha, float beta,
        int last_k_block, float eltwise_alpha, float eltwise_beta,
        float eltwise_scale, memory_storage_t &flag) const {

    int64_t offset_f = 0;

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
    arg_list.set(12, alpha);
    arg_list.set(13, beta);
    arg_list.set(14, last_k_block);
    arg_list.set(15, eltwise_alpha);
    arg_list.set(16, eltwise_beta);
    arg_list.set(17, eltwise_scale);
    if (pd()->gemm_type_ == type::no_copy_k_unroll) {
        arg_list.set(18, flag);
        arg_list.set(19, offset_f);
    }

    bool transa = (pd()->desc()->transa == dnnl_trans);
    bool transb = (pd()->desc()->transb == dnnl_trans);

    int unroll_m, unroll_n, unroll_k;

    gen9_gemm_nocopy_kernel_t::get_unrolls(
            transa, transb, unroll_m, unroll_n, unroll_k, pd()->desc()->c_type);

    size_t nthreads_x = (n + unroll_n - 1) / nstl::max(unroll_n, 1);
    size_t nthreads_y = (m + unroll_m - 1) / nstl::max(unroll_m, 1);
    size_t nthreads_z;
    if (pd()->gemm_type_ == type::no_copy_k_unroll)
        nthreads_z = (k + unroll_k - 1) / nstl::max(unroll_k, 1);
    else
        nthreads_z = 1;

    size_t lthreads_x = 2;
    size_t lthreads_y = 8;
    size_t lthreads_z = 1;

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
    size_t gws[3] = {nthreads_x * subgroup_size, nthreads_y, nthreads_z};
    size_t lws[3] = {lthreads_x * subgroup_size, lthreads_y, lthreads_z};

    auto nd_range = compute::nd_range_t(gws, lws);
    return parallel_for(ctx, nd_range, nocopy_kernel_, arg_list);
}

status_t gen9_gemm_t::launch_nocopy_superkernel(const gemm_exec_ctx_t &ctx,
        compute::compute_stream_t *compute_stream, const memory_storage_t &plan,
        int32_t threads, const memory_storage_t &a, const memory_storage_t &b,
        const memory_storage_t &c, int64_t offset_a, int64_t offset_b,
        int64_t offset_c, int32_t lda, int32_t ldb, int32_t ldc, int32_t m,
        int32_t n, int32_t k, float alpha, float beta, int last_k_block,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale) const {

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, plan);
    arg_list.set(1, threads);
    arg_list.set(2, a);
    arg_list.set(3, b);
    arg_list.set(4, c);
    arg_list.set(5, offset_a);
    arg_list.set(6, offset_b);
    arg_list.set(7, offset_c);
    arg_list.set(8, lda);
    arg_list.set(9, ldb);
    arg_list.set(10, ldc);
    arg_list.set(11, m);
    arg_list.set(12, n);
    arg_list.set(13, k);
    arg_list.set(14, alpha);
    arg_list.set(15, beta);
    arg_list.set(16, last_k_block);
    arg_list.set(17, eltwise_alpha);
    arg_list.set(18, eltwise_beta);
    arg_list.set(19, eltwise_scale);

    size_t lthreads = nstl::min(pd()->hw_threads_, threads);

    static constexpr size_t subgroup_size = 16;
    size_t lws[3] = {subgroup_size, 1, 1};
    size_t gws[3] = {lthreads * subgroup_size, 1, 1};

    auto nd_range = compute::nd_range_t(gws, lws);
    return parallel_for(ctx, nd_range, nocopy_superkernel_, arg_list);
}

status_t gen9_gemm_t::execute(const gemm_exec_ctx_t &ctx) const {
    if (pd()->gemm_type_ == type::no_copy_superkernel)
        return execute_superkernel(ctx);
    else
        return execute_standard(ctx);
}

status_t gen9_gemm_t::execute_standard(const gemm_exec_ctx_t &ctx) const {
    auto a_type = pd()->desc()->a_type;
    auto b_type = pd()->desc()->b_type;
    auto c_type = pd()->desc()->c_type;

    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto mb = pd()->desc()->batch;
    auto m = pd()->desc()->m;
    auto n = pd()->desc()->n;
    auto k = pd()->desc()->k;

    bool transa = (pd()->desc()->transa == dnnl_trans);
    bool transb = (pd()->desc()->transb == dnnl_trans);

    auto lda = pd()->desc()->lda;
    auto ldb = pd()->desc()->ldb;
    auto ldc = pd()->desc()->ldc;

    auto stride_a = pd()->desc()->stride_a;
    auto stride_b = pd()->desc()->stride_b;
    auto stride_c = pd()->desc()->stride_c;

    auto alpha = pd()->alpha();
    auto beta = pd()->beta();

    auto eltwise_alpha = pd()->eltwise_alpha();
    auto eltwise_beta = pd()->eltwise_beta();
    auto eltwise_scale = pd()->eltwise_scale();

    auto alpha_native = alpha;
    auto beta_native = beta;
    auto one_native = 1.0f;

    auto &a = GEMM_CTX_ARG_STORAGE(a);
    auto &b = GEMM_CTX_ARG_STORAGE(b);
    auto &c = GEMM_CTX_ARG_STORAGE(c);

    size_t off_a0
            = a.offset() / types::data_type_size(a_type) + pd()->dyn_offset_a;
    size_t off_b0
            = b.offset() / types::data_type_size(b_type) + pd()->dyn_offset_b;
    size_t off_c0
            = c.offset() / types::data_type_size(c_type) + pd()->dyn_offset_c;

    bool nocopy = (pd()->gemm_type_ == type::no_copy)
            || (pd()->gemm_type_ == type::no_copy_if_even_off && !(off_a0 & 1)
                    && !(off_b0 & 1))
            || (pd()->gemm_type_ == type::no_copy_k_unroll);

    status_t status;
    constexpr int64_t align = 0x1000;
    int block_m, block_n, block_k;
    if (!nocopy) {
        if (pd()->desc()->acc_type == data_type::f16) {
            block_m = driver_params_f16_copy_t::block_m;
            block_n = driver_params_f16_copy_t::block_n;
            block_k = driver_params_f16_copy_t::block_k;
        } else {
            block_m = driver_params_f32_copy_t::block_m;
            block_n = driver_params_f32_copy_t::block_n;
            block_k = driver_params_f32_copy_t::block_k;
        }
    } else {
        block_m = driver_params_f32_nocopy_t::block_m;
        block_n = driver_params_f32_nocopy_t::block_n;
        block_k = driver_params_f32_nocopy_t::block_k;
    }

    if (!nocopy && beta != 0. && beta != 1.) {
        status = launch_beta(
                ctx, compute_stream, m, n, beta_native, c, off_c0, ldc);
        if (status) return status;
    }

    int64_t off_b_packed = 0;
    int64_t off_a_packed
            = ((off_b_packed + block_n * block_k) + align - 1) & -align;

    auto temp_buf = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_gemm_tmp_buffer);

    for (int64_t batch = 0; batch < mb; batch++) {
        for (int64_t Bk = 0; Bk < k; Bk += block_k) {
            int64_t size_k = k - Bk;
            bool last_k_block = (size_k <= block_k);
            if (!last_k_block) size_k = block_k;

            for (int64_t Bm = 0; Bm < m; Bm += block_m) {
                int64_t size_m = m - Bm;
                if (size_m > block_m) size_m = block_m;

                auto off_a_src = off_a0 + batch * stride_a
                        + (!transa ? (Bm + Bk * lda) : (Bk + Bm * lda));

                if (!nocopy) {
                    status = launch_copy(ctx, compute_stream, size_k, size_m, a,
                            off_a_src, lda, alpha_native, *temp_buf,
                            off_a_packed, false, !transa);
                    if (status) return status;
                }

                for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                    int64_t size_n = n - Bn;
                    if (size_n > block_n) size_n = block_n;

                    auto off_b_src = off_b0 + batch * stride_b
                            + (!transb ? (Bk + Bn * ldb) : (Bn + Bk * ldb));

                    if (!nocopy && ((Bn == 0) || (n > block_n))) {
                        status = launch_copy(ctx, compute_stream, size_k,
                                size_n, b, off_b_src, ldb, one_native,
                                *temp_buf, off_b_packed, true, transb);
                        if (status) return status;
                    }

                    auto off_c = off_c0 + batch * stride_c + Bm + Bn * ldc;

                    if (nocopy) {
                        auto flag
                                = ctx.get_scratchpad_grantor()
                                          .get_memory_storage(memory_tracking::
                                                          names::key_gemm_flag);
                        float eff_beta = (Bk == 0) ? beta : 1.0f;
                        status = launch_nocopy(ctx, compute_stream, a, b, c,
                                off_a_src, off_b_src, off_c, lda, ldb, ldc,
                                size_m, size_n, size_k, alpha, eff_beta,
                                (int)last_k_block, eltwise_alpha, eltwise_beta,
                                eltwise_scale, *flag);
                    } else {
                        bool beta0 = (beta == 0) && (Bk == 0);
                        status = launch_compute(ctx, compute_stream, size_m,
                                size_n, size_k, *temp_buf, off_a_packed,
                                off_b_packed, c, off_c, ldc, (int)last_k_block,
                                eltwise_alpha, eltwise_beta, eltwise_scale,
                                beta0);
                    }
                    if (status) return status;
                }
            }
        }
    }

    return status::success;
}

status_t gen9_gemm_t::execute_superkernel(const gemm_exec_ctx_t &ctx) const {
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

    auto lda = pd()->desc()->lda;
    auto ldb = pd()->desc()->ldb;
    auto ldc = pd()->desc()->ldc;

    auto alpha = pd()->alpha();
    auto beta = pd()->beta();

    auto eltwise_alpha = pd()->eltwise_alpha();
    auto eltwise_beta = pd()->eltwise_beta();
    auto eltwise_scale = pd()->eltwise_scale();

    auto &a = GEMM_CTX_ARG_STORAGE(a);
    auto &b = GEMM_CTX_ARG_STORAGE(b);
    auto &c = GEMM_CTX_ARG_STORAGE(c);

    size_t off_a0
            = a.offset() / types::data_type_size(a_type) + pd()->dyn_offset_a;
    size_t off_b0
            = b.offset() / types::data_type_size(b_type) + pd()->dyn_offset_b;
    size_t off_c
            = c.offset() / types::data_type_size(c_type) + pd()->dyn_offset_c;

    status_t status;
    auto block_k = driver_params_f32_nocopy_t::block_k;

    auto temp_buf = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_gemm_tmp_buffer);
    auto temp_buf_size
            = ctx.get_scratchpad_grantor()
                      .get_registry()
                      .get(memory_tracking::names::key_gemm_tmp_buffer)
                      .size;

    int unroll_m[2], unroll_n;
    gen9_gemm_nocopy_superkernel_t::get_unrolls(
            transa, transb, unroll_m, unroll_n);

    int km = utils::div_up(m, unroll_m[0]);
    int kn = utils::div_up(n, unroll_n);
    int last_ldispatch = 0;
    int km_large = km;
    int best_km_large = 0;

    auto good_enough = [=](int ldispatch, int threads) -> bool {
        return (threads < pd()->hw_threads_)
                && (ldispatch >= pd()->eu_count_ * 3);
    };

    while (km_large >= 0) {
        int km_small = utils::div_up(m - (km_large * unroll_m[0]), unroll_m[1]);
        km_small = nstl::max(0, km_small);

        auto threads = (km_large + km_small) * kn;
        auto ldispatch = threads % pd()->hw_threads_;

        if (ldispatch == 0 || good_enough(ldispatch, threads)) {
            best_km_large = km_large;
            break;
        } else if (ldispatch < last_ldispatch)
            break;
        else if (ldispatch > last_ldispatch)
            best_km_large = km_large;

        last_ldispatch = ldispatch;
        km_large--;
    }

    km_large = best_km_large;

    int km_small = utils::div_up(m - (km_large * unroll_m[0]), unroll_m[1]);
    km_small = nstl::max(0, km_small);

    km = km_small + km_large;

    int threads_ = km * kn;

    auto n_block_target = nstl::max<int>(1, 128 / unroll_n);
    auto columns = utils::div_up(kn, n_block_target);
    auto kn_left = (n_block_target - kn) % n_block_target;
    if (kn_left < 0) kn_left += n_block_target;
    auto spread = nstl::min(kn_left, columns);

    int bn0, bn1, columns_small;
    if (spread == columns) {
        bn0 = utils::div_up(kn, columns);
        bn1 = bn0 - 1;
        columns_small = (bn0 * columns) - kn;
    } else {
        bn0 = n_block_target;
        bn1 = n_block_target - 1;
        columns_small = spread;
    }

    void *plan_void = nullptr;
    temp_buf->map_data(&plan_void, nullptr, temp_buf_size);

    if (!plan_void) return status::runtime_error;

    auto plan = (plan_element_t *)plan_void;

    plan[0].next_id = pd()->hw_threads_;
    plan[0].done_count = 0;

    int p = 1, j0 = 0;
    for (int column = 0; column < columns; column++) {
        auto bn = (column >= (columns - columns_small)) ? bn1 : bn0;
        int i0 = 0;
        for (int ki = 0; ki < km; ki++) {
            int m_idx = (ki >= (km - km_small));
            for (int bj = 0; bj < bn; bj++, p++) {
                plan[p].i0 = i0;
                plan[p].j0 = j0 + bj * unroll_n;
                plan[p].kid0 = m_idx;
                plan[p].kid1 = 0;
            }
            auto um = m_idx ? unroll_m[1] : unroll_m[0];
            i0 += um;
        }
        j0 += bn * unroll_n;
    }

    temp_buf->unmap_data(plan_void, nullptr);

    for (int64_t Bk = 0; Bk < k; Bk += block_k) {
        int64_t size_k = k - Bk;
        bool last_k_block = (size_k <= block_k);
        if (!last_k_block) size_k = block_k;

        auto off_a = off_a0 + (!transa ? Bk * lda : Bk);
        auto off_b = off_b0 + (!transb ? Bk : Bk * ldb);

        auto this_beta = (Bk == 0) ? beta : 1.0f;

        status = launch_nocopy_superkernel(ctx, compute_stream, *temp_buf,
                threads_, a, b, c, off_a, off_b, off_c, lda, ldb, ldc, m, n,
                size_k, alpha, this_beta, (int)last_k_block, eltwise_alpha,
                eltwise_beta, eltwise_scale);

        if (status) return status;
    }

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
