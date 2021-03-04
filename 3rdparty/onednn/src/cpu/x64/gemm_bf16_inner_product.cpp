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

#include <algorithm>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/gemm_bf16_inner_product.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::primitive_kind;
using namespace memory_tracking::names;
using namespace dnnl::impl::cpu::x64::bf16_support;

template <data_type_t dst_data_type>
status_t gemm_bf16_inner_product_fwd_t<dst_data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const dim_t M = pd()->OC();
    const dim_t N = pd()->MB();
    const dim_t K = pd()->IC_total_padded();

    const auto &wmd = *pd()->weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

    acc_data_t *acc = pd()->dst_is_acc_
            ? (acc_data_t *)dst
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0;
    status_t st = gemm_bf16bf16f32(wei_tr ? "T" : "N", "N", &M, &N, &K, &alpha,
            weights, wei_tr ? &K : &M, src, &K, &beta_, acc, &M);
    if (st != status::success) return st;

    const float *scales = pd()->attr()->output_scales_.scales_;
    if (postops_in_ip_) {
        const bool force_sequential = pp_kernel_->sequential_kernel();
        parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
            size_t start = 0, end = 0;
            size_t work_size = M * N;
            balance211(work_size, nthr, ithr, start, end);
            (*pp_kernel_)(dst, acc, bias, scales, start, end, 0, 0, nullptr);
        });
    }

    return st;
}

template <data_type_t diff_src_data_type>
status_t
gemm_bf16_inner_product_bwd_data_t<diff_src_data_type>::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const dim_t M = pd()->IC_total_padded();
    const dim_t N = pd()->MB();
    const dim_t K = pd()->OC();

    const auto &wmd = *pd()->weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

    acc_data_t *acc = pd()->diff_src_is_acc_
            ? (acc_data_t *)diff_src
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    status_t st = gemm_bf16bf16f32(wei_tr ? "T" : "N", "N", &M, &N, &K, &alpha,
            weights, wei_tr ? &K : &M, diff_dst, &K, &beta, acc, &M);
    if (st != status::success) return st;

    if (!pd()->diff_src_is_acc_) {
        parallel(0, [&](int ithr, int nthr) {
            size_t start = 0, end = 0;
            size_t work_size = M * N;
            balance211(work_size, nthr, ithr, start, end);
            if (end > start)
                cvt_float_to_bfloat16((bfloat16_t *)&diff_src[start],
                        (const float *)&acc[start], end - start);
        });
    }

    return status::success;
}

template <data_type_t diff_wei_data_type>
status_t gemm_bf16_inner_product_bwd_weights_t<diff_wei_data_type>::
        execute_backward_weights(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(diff_wei_data_t *, DNNL_ARG_DIFF_WEIGHTS);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    diff_dst += diff_dst_d.offset0();

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t IC = pd()->IC_total_padded();

    const auto &wmd = *pd()->diff_weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

    const dim_t M = wei_tr ? OC : IC;
    const dim_t N = wei_tr ? IC : OC;
    const dim_t K = MB;

    acc_data_t *acc = pd()->diff_wei_is_acc_
            ? (acc_data_t *)diff_weights
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    key_iprod_int_dat_in_acc_dt);

    float alpha = 1.0, beta = 0.0;
    status_t st = gemm_bf16bf16f32("N", "T", &M, &N, &K, &alpha,
            wei_tr ? diff_dst : src, &M, wei_tr ? src : diff_dst, &N, &beta,
            acc, &M);
    if (st != status::success) return st;

    if (!pd()->diff_wei_is_acc_) {
        parallel(0, [&](int ithr, int nthr) {
            constexpr size_t blksize = 64;
            size_t start = 0, end = 0;
            size_t work_size = M * N;
            balance211(
                    utils::div_up(work_size, blksize), nthr, ithr, start, end);
            start = std::min(work_size, start * blksize);
            end = std::min(work_size, end * blksize);
            if (end > start) {
                cvt_float_to_bfloat16((bfloat16_t *)&diff_weights[start],
                        (const float *)&acc[start], end - start);
            }
        });
    }

    execute_backward_bias(ctx);

    return status::success;
}

template <data_type_t diff_wei_data_type>
void gemm_bf16_inner_product_bwd_weights_t<diff_wei_data_type>::
        execute_backward_bias(const exec_ctx_t &ctx) const {
    if (!pd()->with_bias()) return;

    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto diff_bias = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_BIAS);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    diff_dst += diff_dst_d.offset0();
    diff_bias += diff_bias_d.data_type_size() * diff_bias_d.offset0();

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();

    constexpr dim_t blksize = pd_t::bias_blksize;
    const dim_t OCB = utils::div_up(OC, blksize);

    dim_t OC_per_thread {0};
    int nthr_OCB {0}, nthr_MB {0};
    pd()->get_bias_partitioning(OC_per_thread, nthr_OCB, nthr_MB);

    const bool diff_bias_is_acc
            = nthr_MB == 1 && diff_bias_d.data_type() == data_type::f32;
    float *diff_bias_acc = diff_bias_is_acc
            ? (float *)diff_bias
            : (float *)ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    key_iprod_bias_bf16_convert_wsp);

    parallel(pd()->bias_reduction_nthr_, [&](int ithr, int nthr) {
        if (ithr < nthr_OCB * nthr_MB) {
            const int ithr_MB = ithr / nthr_OCB;
            const int ithr_OCB = ithr % nthr_OCB;

            dim_t ocb_s {0}, ocb_e {0};
            balance211(OCB, nthr_OCB, ithr_OCB, ocb_s, ocb_e);
            const dim_t oc_s = std::min(ocb_s * blksize, OC);
            const dim_t oc_e = std::min(ocb_e * blksize, OC);
            const dim_t oc_len = oc_e - oc_s;

            dim_t mb_s {0}, mb_e {0};
            balance211(MB, nthr_MB, ithr_MB, mb_s, mb_e);
            const dim_t mb_len = mb_e - mb_s;

            const dim_t db_offset = diff_bias_is_acc
                    ? oc_s
                    : (ithr_OCB * nthr_MB + ithr_MB) * OC_per_thread;
            float *db = diff_bias_acc + db_offset;

            PRAGMA_OMP_SIMD()
            for (dim_t oc = 0; oc < oc_len; ++oc)
                db[oc] = 0;

            (*bias_reduction_)(db, &((bfloat16_t *)diff_dst)[mb_s * OC + oc_s],
                    (size_t)oc_len, (size_t)mb_len);

            if (!diff_bias_is_acc && nthr_MB == 1)
                cvt_float_to_bfloat16(
                        &((bfloat16_t *)diff_bias)[oc_s], db, oc_len);
        }
    });

    if (nthr_MB == 1) return; // no reduction required

    parallel(pd()->bias_reduction_nthr_, [&](int ithr, int nthr) {
        if (ithr < nthr_OCB) {
            const int ithr_OCB = ithr;

            dim_t ocb_s {0}, ocb_e {0};
            balance211(OCB, nthr_OCB, ithr_OCB, ocb_s, ocb_e);
            const dim_t oc_s = std::min(ocb_s * blksize, OC);
            const dim_t oc_e = std::min(ocb_e * blksize, OC);
            const dim_t oc_len = oc_e - oc_s;

            float *db = diff_bias_acc + ithr_OCB * nthr_MB * OC_per_thread;

            for (dim_t thr_MB = 1; thr_MB < nthr_MB; ++thr_MB) {
                const float *thr_db = db + thr_MB * OC_per_thread;

                PRAGMA_OMP_SIMD()
                for (dim_t oc = 0; oc < oc_len; ++oc)
                    db[oc] += thr_db[oc];
            }

            if (diff_bias_d.data_type() == data_type::f32) {
                float *res = &((float *)diff_bias)[oc_s];

                PRAGMA_OMP_SIMD()
                for (dim_t oc = 0; oc < oc_len; ++oc)
                    res[oc] = db[oc];
            } else {
                cvt_float_to_bfloat16(
                        &((bfloat16_t *)diff_bias)[oc_s], db, oc_len);
            }
        }
    });
}

template struct gemm_bf16_inner_product_fwd_t<data_type::f32>;
template struct gemm_bf16_inner_product_fwd_t<data_type::bf16>;
template struct gemm_bf16_inner_product_bwd_data_t<data_type::f32>;
template struct gemm_bf16_inner_product_bwd_data_t<data_type::bf16>;
template struct gemm_bf16_inner_product_bwd_weights_t<data_type::f32>;
template struct gemm_bf16_inner_product_bwd_weights_t<data_type::bf16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
