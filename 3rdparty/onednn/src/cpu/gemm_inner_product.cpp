/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/gemm_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::primitive_kind;

template <impl::data_type_t data_type>
status_t gemm_inner_product_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t IC = pd()->IC_total_padded();

    const auto &wmd = *pd()->weights_md();
    // check if OC is NOT the leading dimension
    bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

    const float *scales = pd()->attr()->output_scales_.scales_;

    float alpha = 1.;
    status_t st = extended_sgemm(wei_tr ? "T" : "N", "N", &OC, &MB, &IC, &alpha,
            weights, wei_tr ? &IC : &OC, src, &IC, &beta_, dst, &OC,
            postops_in_ip_ ? nullptr : bias);

    if (st != status::success) return st;

    if (postops_in_ip_) {
        const bool force_sequential = pp_kernel_->sequential_kernel();
        parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)(OC * MB), nthr, ithr, start, end);
            (*pp_kernel_)(
                    dst, dst, (char *)bias, scales, start, end, 0, 0, nullptr);
        });
    }

    return status::success;
}

template <impl::data_type_t data_type>
status_t gemm_inner_product_bwd_data_t<data_type>::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t IC = pd()->IC_total_padded();

    const auto &wmd = *pd()->weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

    float alpha = 1.0, beta = 0.0;
    status_t st = extended_sgemm(wei_tr ? "T" : "N", "N", &IC, &MB, &OC, &alpha,
            weights, wei_tr ? &OC : &IC, diff_dst, &OC, &beta, diff_src, &IC);

    return st;
}

template <impl::data_type_t data_type>
status_t gemm_inner_product_bwd_weights_t<data_type>::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_BIAS);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    diff_dst += diff_dst_d.offset0();

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t IC = pd()->IC_total_padded();

    const auto &wmd = *pd()->diff_weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] == 1;

    float alpha = 1.0, beta = 0.0;
    status_t st;
    if (wei_tr)
        st = extended_sgemm("N", "T", &OC, &IC, &MB, &alpha, diff_dst, &OC, src,
                &IC, &beta, diff_weights, &OC);
    else
        st = extended_sgemm("N", "T", &IC, &OC, &MB, &alpha, src, &IC, diff_dst,
                &OC, &beta, diff_weights, &IC);

    if (st != status::success) return st;

    if (diff_bias) {
        diff_bias += diff_bias_d.offset0();
        constexpr dim_t blksize = 8;
        const dim_t OC_blocks = utils::div_up(OC, blksize);
        parallel(0, [&](const int ithr, const int nthr) {
            dim_t oc_s {0}, oc_e {0};
            balance211(OC_blocks, nthr, ithr, oc_s, oc_e);
            oc_s = std::min(oc_s * blksize, OC);
            oc_e = std::min(oc_e * blksize, OC);

            PRAGMA_OMP_SIMD()
            for (dim_t oc = oc_s; oc < oc_e; ++oc) {
                diff_bias[oc] = diff_dst[oc];
            }

            for (dim_t mb = 1; mb < MB; ++mb) {
                PRAGMA_OMP_SIMD()
                for (dim_t oc = oc_s; oc < oc_e; ++oc) {
                    diff_bias[oc] += diff_dst[mb * OC + oc];
                }
            }
        });
    }

    return status::success;
}

template struct gemm_inner_product_fwd_t<data_type::f32>;
template struct gemm_inner_product_bwd_data_t<data_type::f32>;
template struct gemm_inner_product_bwd_weights_t<data_type::f32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
