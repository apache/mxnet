/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm_x8s8s32x_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace math;
using namespace format_tag;
using namespace memory_tracking::names;

template <data_type_t src_type, data_type_t dst_type>
status_t gemm_x8s8s32x_inner_product_fwd_t<src_type, dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();

    const auto &wmd = *pd()->weights_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;

    const dim_t M = OC;
    const dim_t N = MB;
    const dim_t K = pd()->IC_total_padded();
    const int8_t off_a = 0;
    const src_data_t off_b = 0;
    const int32_t off_c = 0;

    const float *scales = pd()->attr()->output_scales_.scales_;

    acc_data_t *acc = pd()->dst_is_acc_
            ? (acc_data_t *)dst
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    key_iprod_int_dat_in_acc_dt);

    const float onef = 1.0, zerof = 0.0;
    status_t st = gemm_s8x8s32(wei_tr ? "T" : "N", "N", "F", &M, &N, &K, &onef,
            weights, wei_tr ? &K : &M, &off_a, src, &K, &off_b, &zerof, acc, &M,
            &off_c);
    if (st != status::success) return st;

    if (!pd()->attr()->has_default_values() || dst_type != data_type::s32
            || pd()->with_bias()) {
        const bool force_sequential
                = pp_kernel_->sequential_kernel() || MB * OC < 2000;
        parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)(OC * MB), nthr, ithr, start, end);
            (*pp_kernel_)(dst, acc, bias, scales, start, end, 0, 0, nullptr);
        });
    }

    return st;
}

using namespace data_type;

template struct gemm_x8s8s32x_inner_product_fwd_t<u8, f32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<u8, s32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<u8, s8>;
template struct gemm_x8s8s32x_inner_product_fwd_t<u8, u8>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, f32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, s32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, s8>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, u8>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
