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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/matmul/ref_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

template <data_type_t src_type, data_type_t weights_type, data_type_t dst_type,
        data_type_t acc_type>
status_t ref_matmul_t<src_type, weights_type, dst_type, acc_type>::execute_ref(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const weights_data_t *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    DEFINE_SCALES_BUFFER(scales);
    DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(weights_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, DNNL_ARG_DST);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    const bool non_default_attrs = !pd()->attr()->has_default_values();

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const int batch_ndims = ndims - 2;
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();

    const int src_mask
            = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
    const int wei_mask
            = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);
    const int bia_mask
            = utils::get_dims_mask(dst_d.dims(), bia_d.dims(), ndims);

    // mm kernel
    auto ker = [&](const dims_t dst_dims_idx, dim_t m, dim_t n) {
        acc_data_t acc = 0;
        dims_t src_dims_idx, weights_dims_idx;
        utils::copy_dims_with_mask(src_dims_idx, dst_dims_idx, ndims, src_mask);
        utils::copy_dims_with_mask(
                weights_dims_idx, dst_dims_idx, ndims, wei_mask);
        src_dims_idx[ndims - 2] = m;
        weights_dims_idx[ndims - 1] = n;
        auto &src_k_dim = src_dims_idx[ndims - 1];
        auto &wei_k_dim = weights_dims_idx[ndims - 2];
        for (dim_t k = 0; k < K; ++k) {
            src_k_dim = k;
            wei_k_dim = k;
            acc += (src[src_d.off_v(src_dims_idx)] - src_zero_point)
                    * (weights[weights_d.off_v(weights_dims_idx)]
                            - weights_zero_point);
        }
        return acc;
    };

    // bias section
    const data_type_t bia_dt = pd()->desc()->bias_desc.data_type;
    auto get_bias = [&](const dims_t &dst_dims_idx) -> float {
        dims_t bia_dims_idx;
        utils::copy_dims_with_mask(bia_dims_idx, dst_dims_idx, ndims, bia_mask);
        dim_t off = bia_d.off_v(bia_dims_idx);
        return math::get_bias(bias, off, bia_dt);
    };

    // output scale section
    const dim_t scale_stride = pd()->attr()->output_scales_.mask_ == 0 ? 0 : 1;

    // computations
    parallel_nd(batch, M, N, [&](dim_t mb, dim_t m, dim_t n) {
        dims_t dst_dims_idx;
        // account for M, N dims for index calculations
        const size_t l_offset = mb * M * N + m * N + n;
        utils::l_dims_by_l_offset(dst_dims_idx, l_offset, dst_d.dims(), ndims);
        auto &dst_value = dst[dst_d.off_v(dst_dims_idx)];
        acc_data_t acc = ker(dst_dims_idx, m, n);
        float res = acc;
        if (bias || non_default_attrs) {
            if (bias) res += get_bias(dst_dims_idx);
            res *= scales[scale_stride * n];

            ref_post_ops_t::args_t args;
            args.dst_val = dst_value;
            args.ctx = &ctx;
            args.l_offset = l_offset;
            args.dst_md = pd()->dst_md();
            ref_post_ops->execute(res, args);

            res += (float)dst_zero_point;
        }
        dst_value = cpu::saturate_and_round<dst_data_t>(res);
        utils::dim_iterator(dst_d.dims(), dst_dims_idx, batch_ndims);
    });

    return status::success;
}

using namespace data_type;
template struct ref_matmul_t<f32, f32, f32, f32>;
template struct ref_matmul_t<bf16, bf16, f32, f32>;
template struct ref_matmul_t<bf16, bf16, bf16, f32>;
template struct ref_matmul_t<s8, s8, f32, s32>;
template struct ref_matmul_t<s8, s8, s32, s32>;
template struct ref_matmul_t<s8, s8, s8, s32>;
template struct ref_matmul_t<s8, s8, u8, s32>;
template struct ref_matmul_t<u8, s8, f32, s32>;
template struct ref_matmul_t<u8, s8, s32, s32>;
template struct ref_matmul_t<u8, s8, s8, s32>;
template struct ref_matmul_t<u8, s8, u8, s32>;

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
