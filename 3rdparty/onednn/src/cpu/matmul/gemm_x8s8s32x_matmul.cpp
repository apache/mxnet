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

#include <atomic>

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/matmul/gemm_x8s8s32x_matmul.hpp"
#include "cpu/matmul/matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

using namespace data_type;

namespace {
template <typename pd_t>
bool need_post_processing(const pd_t *pd, float runtime_dst_zero_point = 0.f) {
    return pd->with_bias() || pd->dst_md()->data_type != s32
            || !pd->params().dst_is_acc_
            || !pd->params().pp_attr_.has_default_values()
            || !pd->params().pp_attr_.zero_points_.has_default_values(
                    DNNL_ARG_DST)
            || runtime_dst_zero_point != 0.f;
}
} // namespace

template <data_type_t src_type, data_type_t weights_type, data_type_t dst_type>
status_t gemm_x8s8s32x_matmul_t<src_type, weights_type, dst_type>::pd_t::init(
        engine_t *engine) {
    using namespace utils;

    auto check_bias = [&]() -> bool {
        return !with_bias()
                || (utils::one_of(weights_md(1)->data_type, f32, s32, s8, u8)
                        && is_bias_1xN());
    };

    auto check_attr_oscale = [&]() -> bool {
        const auto &oscale = attr()->output_scales_;
        return oscale.mask_ == 0
                || (oscale.mask_ == (1 << 1) && batched() == false);
    };

    auto check_attr_zero_points
            = [&]() -> bool { return attr()->zero_points_.common(); };

    auto check_attr_post_ops = [&]() -> bool {
        using namespace primitive_kind;
        const auto &p = attr()->post_ops_;
        switch (p.len()) {
            case 0: return true;
            case 1: return p.contain(sum, 0) || p.contain(eltwise, 0);
            case 2: return p.contain(sum, 0) && p.contain(eltwise, 1);
            default: return false;
        }
    };

    bool ok = src_md()->data_type == src_type
            && weights_md()->data_type == weights_type
            && desc()->accum_data_type == acc_type
            && dst_md()->data_type == dst_type && check_bias()
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::zero_points_runtime
                    | primitive_attr_t::skip_mask_t::post_ops)
            && check_attr_oscale() && check_attr_zero_points()
            && check_attr_post_ops() && set_default_formats()
            && gemm_based::check_gemm_compatible_formats(*this);
    if (!ok) return status::unimplemented;

    // set states

    // copy attributes and drop src and weights zero points
    CHECK(params_.pp_attr_.copy_from(*attr()));
    params_.pp_attr_.zero_points_.set(DNNL_ARG_SRC, 0);
    params_.pp_attr_.zero_points_.set(DNNL_ARG_WEIGHTS, 0);

    params_.gemm_applies_output_scales_ = false;
    params_.gemm_beta_ = 0.f;

    bool do_sum = params_.pp_attr_.post_ops_.find(primitive_kind::sum) >= 0;
    params_.dst_is_acc_ = utils::one_of(dst_type, s32, f32) && !do_sum;

    params_.has_pp_kernel_ = need_post_processing(this);

    gemm_based::book_acc_scratchpad(*this, params_, sizeof(acc_data_t));

    return status::success;
}

template <data_type_t src_type, data_type_t weights_type, data_type_t dst_type>
void gemm_x8s8s32x_matmul_t<src_type, weights_type, dst_type>::
        post_process_src_and_weights_zero_points(
                std::vector<acc_data_t> &src_comp,
                std::vector<acc_data_t> &wei_comp, dim_t M, dim_t N, dim_t K,
                const src_data_t *src, dim_t src_s0, dim_t src_s1,
                const weights_data_t *wei, dim_t wei_s0, dim_t wei_s1,
                acc_data_t *acc, int ldc, acc_data_t src_zero_point,
                acc_data_t wei_zero_point) const {
    if (wei_zero_point) {
        for_(dim_t m = 0; m < M; ++m)
        for (dim_t k = 0; k < K; ++k) {
            if (k == 0) src_comp[m] = acc_data_t(0);
            src_comp[m] += src[src_s0 * m + src_s1 * k];
        }
    }

    if (src_zero_point) {
        for_(dim_t k = 0; k < K; ++k)
        for (dim_t n = 0; n < N; ++n) {
            if (k == 0) wei_comp[n] = acc_data_t(0);
            wei_comp[n] += wei[wei_s0 * k + wei_s1 * n];
        }
    }

    for_(dim_t m = 0; m < M; ++m)
    for (dim_t n = 0; n < N; ++n)
        acc[m * ldc + n] += 0 - src_zero_point * wei_comp[n]
                - wei_zero_point * src_comp[m]
                + src_zero_point * wei_zero_point * (int)K;
}

template <data_type_t src_type, data_type_t weights_type, data_type_t dst_type>
status_t gemm_x8s8s32x_matmul_t<src_type, weights_type, dst_type>::execute_ref(
        const exec_ctx_t &ctx) const {
    using math::get_bias;

    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const weights_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    DEFINE_SCALES_BUFFER(scales);
    DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(weights_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, DNNL_ARG_DST);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    src_data_t gemm_off_a = (src_data_t)src_zero_point;
    weights_data_t gemm_off_b = (weights_data_t)weights_zero_point;
    bool post_process_src_and_weights_zero_points_outside_of_gemm = false;
    if (gemm_off_a != src_zero_point || gemm_off_b != weights_zero_point) {
        post_process_src_and_weights_zero_points_outside_of_gemm = true;
        gemm_off_a = gemm_off_b = 0;
    }
    const float dst_zero_point_f32 = (float)dst_zero_point;

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const int batch_ndims = ndims - 2;
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();
    const char transA = helper.transA();
    const char transB = helper.transB();
    const dim_t lda = helper.lda();
    const dim_t ldb = helper.ldb();
    const dim_t ldc = helper.ldc();
    const dim_t acc_batch_stride = M * N;
    const int ldx_dim_idx = pd()->ndims() - 2;
    const dim_t *src_strides = &src_d.blocking_desc().strides[ldx_dim_idx];
    const dim_t *weights_strides
            = &weights_d.blocking_desc().strides[ldx_dim_idx];

    const gemm_based::params_t &params = pd()->params();
    bool dst_is_acc = params.dst_is_acc_;
    acc_data_t *acc = dst_is_acc
            ? (acc_data_t *)dst
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    memory_tracking::names::key_matmul_dst_in_acc_dt);
    // case: dynamic sizes
    bool need_free_acc = false;
    if (acc == nullptr) {
        acc = (acc_data_t *)malloc(sizeof(acc_data_t)
                        * nstl::min(batch, (dim_t)dnnl_get_max_threads()) * M
                        * N,
                64);
        if (acc == nullptr) return status::out_of_memory;
        need_free_acc = true;
    }

    const float alpha = params.get_gemm_alpha(scales);
    const float beta = params.gemm_beta_;
    const dim_t acc_ldc = dst_is_acc ? ldc : N;

    std::atomic<status_t> st(status::success);
    const bool parallel_over_batch = batch > 1;
    if (parallel_over_batch) {
        const int src_mask
                = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
        const int wei_mask
                = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);
        // NOTE: inside lambda, type cast variables captured by reference using
        // either c-like "(type)var" or functional "type(var)" notation in order
        // to avoid gcc bug with c++14 standard. Otherwise, capture by value.
        parallel(0, [=, &st](int ithr, int nthr) {
            size_t batch_start {0}, batch_end {0};
            balance211((size_t)(batch), nthr, ithr, batch_start, batch_end);
            dims_t s_dims_idx, w_dims_idx, d_dims_idx;
            // account for M, N dims for index calculations
            utils::l_dims_by_l_offset(
                    d_dims_idx, batch_start * M * N, dst_d.dims(), ndims);

            const bool reuse_acc = acc != (acc_data_t *)dst;
            acc_data_t *curr_acc
                    = reuse_acc ? acc + ithr * acc_batch_stride : nullptr;

            std::vector<acc_data_t> src_compensation(M, 0);
            std::vector<acc_data_t> weights_compensation(N, 0);

            // icc 17.0 has a bug with capturing const variables with value known
            // at compilation time in lambdas
            const int32_t gemm_off_c = 0;

            for (size_t b = batch_start; b < batch_end; ++b) {
                utils::copy_dims_with_mask(
                        s_dims_idx, d_dims_idx, ndims, src_mask);
                utils::copy_dims_with_mask(
                        w_dims_idx, d_dims_idx, ndims, wei_mask);
                const src_data_t *curr_src = src + src_d.off_v(s_dims_idx);
                const weights_data_t *curr_weights
                        = weights + weights_d.off_v(w_dims_idx);
                const dim_t dst_off = dst_d.off_v(d_dims_idx);
                dst_data_t *curr_dst = dst + dst_off;
                if (!reuse_acc) curr_acc = acc + dst_off;

                status_t st_thr = gemm_s8x8s32(&transB, &transA, "F", &N, &M,
                        &K, &alpha, curr_weights, &ldb, &gemm_off_b, curr_src,
                        &lda, &gemm_off_a, &beta, curr_acc, &acc_ldc,
                        &gemm_off_c);
                if (st_thr != status::success) {
                    st = st_thr;
                    return;
                }

                // if igemm cannot handle src and weights zero points
                if (post_process_src_and_weights_zero_points_outside_of_gemm) {
                    post_process_src_and_weights_zero_points(src_compensation,
                            weights_compensation, M, N, K, curr_src,
                            src_strides[0], src_strides[1], curr_weights,
                            weights_strides[0], weights_strides[1], curr_acc,
                            acc_ldc, src_zero_point, weights_zero_point);
                }

                bool postops_in_matmul
                        = need_post_processing(pd(), dst_zero_point_f32);
                assert(IMPLICATION(postops_in_matmul, params.has_pp_kernel_));

                if (postops_in_matmul) {
                    (*pp_kernel_)(curr_dst, curr_acc, bias, scales, 0, M * N,
                            (size_t)N, ldc, &dst_zero_point_f32);
                }
                utils::dim_iterator(dst_d.dims(), d_dims_idx, batch_ndims);
            }
        });
    } else {
        // icc 17.0 has a bug with capturing const variables with value known
        // at compilation time in lambdas
        const int32_t gemm_off_c = 0;

        status_t st = gemm_s8x8s32(&transB, &transA, "F", &N, &M, &K, &alpha,
                weights, &ldb, &gemm_off_b, src, &lda, &gemm_off_a, &beta, acc,
                &acc_ldc, &gemm_off_c);
        if (st != status::success) return st;

        std::vector<acc_data_t> src_compensation(M, 0);
        std::vector<acc_data_t> weights_compensation(N, 0);

        // if igemm cannot handle src and weights zero points
        if (post_process_src_and_weights_zero_points_outside_of_gemm) {
            post_process_src_and_weights_zero_points(src_compensation,
                    weights_compensation, M, N, K, src, src_strides[0],
                    src_strides[1], weights, weights_strides[0],
                    weights_strides[1], acc, acc_ldc, src_zero_point,
                    weights_zero_point);
        }

        bool postops_in_matmul = need_post_processing(pd(), dst_zero_point_f32);
        assert(IMPLICATION(postops_in_matmul, params.has_pp_kernel_));

        if (postops_in_matmul) {
            const bool force_sequential = pp_kernel_->sequential_kernel();

            parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
                size_t start {}, end {};
                balance211((size_t)(M * N), nthr, ithr, start, end);
                (*pp_kernel_)(dst, acc, bias, scales, start, end, (size_t)N,
                        ldc, &dst_zero_point_f32);
            });
        }
    }
    if (need_free_acc) free(acc);

    return st;
}

template struct gemm_x8s8s32x_matmul_t<s8, s8, f32>;
template struct gemm_x8s8s32x_matmul_t<s8, s8, s32>;
template struct gemm_x8s8s32x_matmul_t<s8, s8, s8>;
template struct gemm_x8s8s32x_matmul_t<s8, s8, u8>;
template struct gemm_x8s8s32x_matmul_t<u8, s8, f32>;
template struct gemm_x8s8s32x_matmul_t<u8, s8, s32>;
template struct gemm_x8s8s32x_matmul_t<u8, s8, s8>;
template struct gemm_x8s8s32x_matmul_t<u8, s8, u8>;

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
