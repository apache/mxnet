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

#include <cmath>

#include "primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace alg_kind;
using namespace math;

float compute_binary_scalar(alg_kind_t alg, float x, float y) {
    switch (alg) {
        case binary_add: return x + y;
        case binary_div: return x / y;
        case binary_max: return nstl::max(x, y);
        case binary_min: return nstl::min(x, y);
        case binary_mul: return x * y;
        default: assert(!"not supported operation!"); return NAN;
    }
}

float compute_eltwise_scalar_fwd(
        const alg_kind_t alg, float s, float alpha, float beta) {
    // don't compute on nan, propagate it
    if (std::isnan(s)) return s;

    float d = 0.f;
    switch (alg) {
        case eltwise_relu: d = relu_fwd(s, alpha); break;
        case eltwise_tanh: d = tanh_fwd(s); break;
        case eltwise_elu: d = elu_fwd(s, alpha); break;
        case eltwise_square: d = square_fwd(s); break;
        case eltwise_abs: d = abs_fwd(s); break;
        case eltwise_sqrt: d = sqrt_fwd(s); break;
        case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
        case eltwise_bounded_relu: d = bounded_relu_fwd(s, alpha); break;
        case eltwise_soft_relu: d = soft_relu_fwd(s); break;
        case eltwise_logistic: d = logistic_fwd(s); break;
        case eltwise_exp: d = exp_fwd(s); break;
        case eltwise_gelu_tanh: d = gelu_tanh_fwd(s); break;
        case eltwise_swish: d = swish_fwd(s, alpha); break;
        case eltwise_log: d = log_fwd(s); break;
        case eltwise_clip: d = clip_fwd(s, alpha, beta); break;
        case eltwise_pow: d = pow_fwd(s, alpha, beta); break;
        case eltwise_gelu_erf: d = gelu_erf_fwd(s); break;
        case eltwise_round: d = round_fwd(s); break;

        case eltwise_relu_use_dst_for_bwd: d = relu_fwd(s, alpha); break;
        case eltwise_tanh_use_dst_for_bwd: d = tanh_fwd(s); break;
        case eltwise_elu_use_dst_for_bwd: d = elu_fwd(s, alpha); break;
        case eltwise_sqrt_use_dst_for_bwd: d = sqrt_fwd(s); break;
        case eltwise_logistic_use_dst_for_bwd: d = logistic_fwd(s); break;
        case eltwise_exp_use_dst_for_bwd: d = exp_fwd(s); break;

        default: assert(!"unknown eltwise alg_kind");
    }
    return d;
}

float compute_eltwise_scalar_bwd(
        const alg_kind_t alg, float dd, float s, float alpha, float beta) {
    float ds = 0.f;
    switch (alg) {
        case eltwise_relu: ds = relu_bwd(dd, s, alpha); break;
        case eltwise_tanh: ds = tanh_bwd(dd, s); break;
        case eltwise_elu: ds = elu_bwd(dd, s, alpha); break;
        case eltwise_square: ds = square_bwd(dd, s); break;
        case eltwise_abs: ds = abs_bwd(dd, s); break;
        case eltwise_sqrt: ds = sqrt_bwd(dd, s); break;
        case eltwise_linear: ds = linear_bwd(dd, s, alpha, beta); break;
        case eltwise_bounded_relu: ds = bounded_relu_bwd(dd, s, alpha); break;
        case eltwise_soft_relu: ds = soft_relu_bwd(dd, s); break;
        case eltwise_logistic: ds = logistic_bwd(dd, s); break;
        case eltwise_exp: ds = exp_bwd(dd, s); break;
        case eltwise_gelu_tanh: ds = gelu_tanh_bwd(dd, s); break;
        case eltwise_swish: ds = swish_bwd(dd, s, alpha); break;
        case eltwise_log: ds = log_bwd(dd, s); break;
        case eltwise_clip: ds = clip_bwd(dd, s, alpha, beta); break;
        case eltwise_pow: ds = pow_bwd(dd, s, alpha, beta); break;
        case eltwise_gelu_erf: ds = gelu_erf_bwd(dd, s); break;

        case eltwise_relu_use_dst_for_bwd:
            ds = relu_bwd_use_dst(dd, s, alpha);
            break;
        case eltwise_tanh_use_dst_for_bwd: ds = tanh_bwd_use_dst(dd, s); break;
        case eltwise_elu_use_dst_for_bwd:
            ds = elu_bwd_use_dst(dd, s, alpha);
            break;
        case eltwise_sqrt_use_dst_for_bwd: ds = sqrt_bwd_use_dst(dd, s); break;
        case eltwise_logistic_use_dst_for_bwd:
            ds = logistic_bwd_use_dst(dd, s);
            break;
        case eltwise_exp_use_dst_for_bwd: ds = exp_bwd_use_dst(dd, s); break;

        default: assert(!"unknown eltwise alg_kind");
    }
    return ds;
}

ref_binary_scalar_t::ref_binary_scalar_t(alg_kind_t alg) : alg_(alg) {
    assert(utils::one_of(alg_, alg_kind::binary_add, alg_kind::binary_max,
            alg_kind::binary_min, alg_kind::binary_mul));
}

ref_binary_scalar_t::ref_binary_scalar_t(
        const post_ops_t::entry_t::binary_t &binary)
    : ref_binary_scalar_t(binary.alg) {}

float ref_binary_scalar_t::compute_scalar(float src0, float src1) const {
    return compute_binary_scalar(alg_, src0, src1);
}

ref_eltwise_scalar_fwd_t::ref_eltwise_scalar_fwd_t(
        alg_kind_t alg, float alpha, float beta, float scale)
    : alg_(alg), alpha_(alpha), beta_(beta), scale_(scale) {
    assert(utils::one_of(alg_, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
            eltwise_exp, eltwise_gelu_tanh, eltwise_swish, eltwise_log,
            eltwise_clip, eltwise_pow, eltwise_gelu_erf, eltwise_round,
            eltwise_relu_use_dst_for_bwd, eltwise_tanh_use_dst_for_bwd,
            eltwise_elu_use_dst_for_bwd, eltwise_sqrt_use_dst_for_bwd,
            eltwise_logistic_use_dst_for_bwd, eltwise_exp_use_dst_for_bwd));
}

ref_eltwise_scalar_fwd_t::ref_eltwise_scalar_fwd_t(
        const post_ops_t::entry_t::eltwise_t &eltwise)
    : ref_eltwise_scalar_fwd_t(
            eltwise.alg, eltwise.alpha, eltwise.beta, eltwise.scale) {}

float ref_eltwise_scalar_fwd_t::compute_scalar(float s) const {
    return compute_eltwise_scalar_fwd(alg_, s, alpha_, beta_) * scale_;
}

ref_post_ops_t::ref_post_ops_t(const post_ops_t &po) : po_(po) {
    for (auto idx = 0; idx < po_.len(); ++idx) {
        const auto &e = po_.entry_[idx];
        if (po_.contain(primitive_kind::eltwise, idx)) {
            eltwise_po_.emplace_back(e.eltwise);
        } else if (po_.contain(primitive_kind::binary, idx)) {
            binary_po_.emplace_back(e.binary);
        }
    }
}

status_t ref_post_ops_t::execute(float &res, const args_t &args) const {
    if (po_.len() == 0) return status::success;

    auto it_eltwise_po = eltwise_po_.begin();
    auto it_binary_po = binary_po_.begin();
    for (auto idx = 0; idx < po_.len(); ++idx) {
        const auto &e = po_.entry_[idx];
        switch (e.kind) {
            case primitive_kind::sum: res += e.sum.scale * args.dst_val; break;
            case primitive_kind::eltwise:
                res = it_eltwise_po->compute_scalar(res);
                it_eltwise_po++;
                break;
            case primitive_kind::binary: {
                assert(args.ctx);
                assert(args.l_offset >= 0);
                assert(args.dst_md);

                const exec_ctx_t &ctx = *args.ctx;
                const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, args.dst_md);
                const auto &dst_dims = dst_d.dims();
                const auto dst_ndims = dst_d.ndims();

                const auto &b = e.binary;
                const memory_desc_wrapper src1_binary_po_d(b.src1_desc);
                dims_t l_dims_binary_po;
                utils::l_dims_by_l_offset(
                        l_dims_binary_po, args.l_offset, dst_dims, dst_ndims);
                int mask_binary_po = utils::get_dims_mask(
                        dst_dims, src1_binary_po_d.dims(), dst_ndims);
                utils::apply_mask_on_dims(
                        l_dims_binary_po, dst_ndims, mask_binary_po);

                const auto off = src1_binary_po_d.off_v(l_dims_binary_po);
                const auto src1_binary_po = CTX_IN_MEM(const void *,
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1));
                float val_po = types::get_float_value(
                        src1_binary_po_d.data_type(), src1_binary_po, off);
                res = it_binary_po->compute_scalar(res, val_po);
                it_binary_po++;
            } break;
            default: assert(!"unsupported post op primitive kind!");
        }
    }
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
