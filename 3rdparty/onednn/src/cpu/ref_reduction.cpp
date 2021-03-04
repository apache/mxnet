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

#include <math.h>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/ref_reduction.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t src_type, data_type_t dst_type, data_type_t acc_type>
void ref_reduction_t<src_type, dst_type, acc_type>::init_acc(
        acc_t &acc, alg_kind_t alg) const {
    using namespace alg_kind;
    using namespace nstl;

    switch (alg) {
        case reduction_max:
            acc = static_cast<acc_t>(numeric_limits<src_t>::lowest());
            break;
        case reduction_min:
            acc = static_cast<acc_t>(numeric_limits<src_t>::max());
            break;
        case reduction_mean:
        case reduction_sum: acc = acc_t(0); break;
        case reduction_mul: acc = acc_t(1); break;
        case reduction_norm_lp_max:
        case reduction_norm_lp_sum:
        case reduction_norm_lp_power_p_max:
        case reduction_norm_lp_power_p_sum: acc = acc_t(0); break;
        default: assert(!"unknown alg");
    }
}

template <data_type_t src_type, data_type_t dst_type, data_type_t acc_type>
void ref_reduction_t<src_type, dst_type, acc_type>::accumulate(
        acc_t &acc, const src_t &src, alg_kind_t alg, float p) const {
    using namespace alg_kind;

    acc_t src_ = static_cast<acc_t>(src);

    switch (alg) {
        case reduction_max: acc = nstl::max(acc, src_); break;
        case reduction_min: acc = nstl::min(acc, src_); break;
        case reduction_mean:
        case reduction_sum: acc += src_; break;
        case reduction_mul: acc *= src_; break;
        case reduction_norm_lp_max:
        case reduction_norm_lp_sum:
        case reduction_norm_lp_power_p_max:
        case reduction_norm_lp_power_p_sum:
            acc += powf(nstl::abs(src_), p);
            break;
        default: assert(!"unknown alg");
    }
}

template <data_type_t src_type, data_type_t dst_type, data_type_t acc_type>
void ref_reduction_t<src_type, dst_type, acc_type>::finalize(
        acc_t &acc, alg_kind_t alg, float p, float eps, dim_t n) const {
    using namespace alg_kind;

    float acc_f32 = static_cast<float>(acc);
    switch (alg) {
        case reduction_mean: acc_f32 /= n; break;
        case reduction_norm_lp_max:
            acc_f32 = nstl::max(acc_f32, eps);
            acc_f32 = powf(acc_f32, 1.0f / p);
            break;
        case reduction_norm_lp_sum:
            acc_f32 += eps;
            acc_f32 = powf(acc_f32, 1.0f / p);
            break;
        case reduction_norm_lp_power_p_max:
            acc_f32 = nstl::max(acc_f32, eps);
            break;
        case reduction_norm_lp_power_p_sum: acc_f32 += eps; break;
        default: break;
    }
    acc = static_cast<acc_t>(acc_f32);
}

template <data_type_t src_type, data_type_t dst_type, data_type_t acc_type>
status_t ref_reduction_t<src_type, dst_type, acc_type>::execute_ref(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(dst_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_mdw(pd()->src_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());

    const int ndims = src_mdw.ndims();
    const auto &src_dims = src_mdw.dims();
    const auto &dst_dims = dst_mdw.dims();

    const auto alg = pd()->desc()->alg_kind;
    const auto p = pd()->desc()->p;
    const auto eps = pd()->desc()->eps;

    dims_t reduce_dims;
    dim_t reduce_size {1}, idle_size = dst_mdw.nelems();

    for (int d = 0; d < ndims; ++d) {
        reduce_dims[d] = dim_t {1};
        const bool is_reduction_dim = src_dims[d] != dst_dims[d];
        if (is_reduction_dim) {
            reduce_dims[d] = src_dims[d];
            reduce_size *= reduce_dims[d];
        }
    }

    parallel_nd(idle_size, [&](dim_t f) {
        dims_t idle_pos, reduce_pos;
        utils::l_dims_by_l_offset(idle_pos, f, dst_mdw.dims(), ndims);
        const dim_t dst_off = dst_mdw.off_v(idle_pos);
        const dim_t src_idle_off = src_mdw.off_v(idle_pos);
        acc_t acc {0};
        init_acc(acc, alg);
        for (dim_t r = 0; r < reduce_size; ++r) {
            utils::l_dims_by_l_offset(reduce_pos, r, reduce_dims, ndims);
            const dim_t src_reduce_off = src_mdw.off_v(reduce_pos);
            const dim_t src_off = src_idle_off + src_reduce_off;
            accumulate(acc, src[src_off], alg, p);
        }
        finalize(acc, alg, p, eps, reduce_size);
        dst[dst_off] = saturate_and_round<dst_t>(acc);
    });

    return status::success;
}

using namespace data_type;
template struct ref_reduction_t<f32, f32, f32>;
template struct ref_reduction_t<bf16, bf16, f32>;
template struct ref_reduction_t<bf16, f32, f32>;
template struct ref_reduction_t<s8, s8, s32>;
template struct ref_reduction_t<s8, s32, s32>;
template struct ref_reduction_t<s8, f32, f32>;
template struct ref_reduction_t<u8, u8, s32>;
template struct ref_reduction_t<u8, s32, s32>;
template struct ref_reduction_t<u8, f32, f32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
