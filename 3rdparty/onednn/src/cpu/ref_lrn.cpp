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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_lrn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {

using acc_data_t = float;

inline acc_data_t fast_negative_powf(acc_data_t omega, acc_data_t beta) {
    acc_data_t Y;
    /*
         * Y = omega^(-3/4) =
         * = 1.0f / sqrtf(omega) * sqrtf(1.0f / sqrtf(omega))
         * = sqrtf(1.0f / sqrtf(omega)) * 1.0f / sqrtf(omega)
         * = sqrtf(1.0f / sqrtf(omega)) / sqrtf(omega)
         * = sqrtf(1.0f / sqrtf(omega) / omega)
         * = sqrtf(1.0f / (sqrtf(omega) * omega))
         */
    if (beta == 0.75f) {
        Y = sqrtf(1.0f / (sqrtf(omega) * omega));
    } else {
        Y = 1.0f / powf(omega, beta);
    }
    return Y;
};
} // namespace

// Forward LRN formula:
// y_i = x_i * (k + a / n * Sum:j [x_j^2])^-b, where
// k, a(alpha), b(beta), n(local_size) - lrn hyperparameters;
// j - kernel points, j in [i - n/2, i + n/2] for ACROSS, 2d-shape for WITHIN;

template <impl::data_type_t d_type>
template <impl::format_tag_t tag>
void ref_lrn_fwd_t<d_type>::execute_forward(const exec_ctx_t &ctx) const {
    using namespace alg_kind;
    using namespace format_tag;

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());

    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const auto stride_mb = data_d.blocking_desc().strides[0];
    const bool across_channels = pd()->desc()->alg_kind == lrn_across_channels;
    static constexpr dim_t blksize = tag == nChw16c ? 16 : 8;
    const auto ndims = data_d.ndims();

    auto compute_n_summands = [&](dim_t size) {
        if (across_channels) {
            return size;
        } else { // within_channel
            dim_t n_summands = 1;
            for (auto d = ndims - 2; d > 0; --d)
                n_summands *= size;
            return n_summands;
        }
    };

    const acc_data_t alpha = static_cast<acc_data_t>(pd()->desc()->lrn_alpha);
    const acc_data_t beta = static_cast<acc_data_t>(pd()->desc()->lrn_beta);
    const acc_data_t k = static_cast<acc_data_t>(pd()->desc()->lrn_k);
    const dim_t size = pd()->desc()->local_size;
    const dim_t half_size = (size - 1) / 2;
    const dim_t summands = compute_n_summands(size);

    auto data_off = [&](dim_t mb, dim_t c, dim_t d, dim_t h, dim_t w) -> dim_t {
        switch (tag) {
            case nChw16c:
            case nChw8c:
                return mb * stride_mb + (c / blksize) * H * W * blksize
                        + h * W * blksize + w * blksize + c % blksize;
            case nchw: return mb * stride_mb + c * H * W + h * W + w;
            case nhwc: return mb * stride_mb + h * W * C + w * C + c;
            default:
                if (ndims >= 5) return data_d.off(mb, c, d, h, w);
                if (ndims >= 4) return data_d.off(mb, c, h, w);
                if (ndims >= 3) return data_d.off(mb, c, w);
                return data_d.off(mb, c);
        }
    };

    // pass by value due to icc170 and icc180 problem on KNL
    auto ker = [=](data_t *d, dim_t mb, dim_t oc, dim_t od, dim_t oh,
                       dim_t ow) {
        acc_data_t sum = 0;
        if (across_channels) {
            const dim_t c_st = nstl::max(oc - half_size + 0, (dim_t)0);
            const dim_t c_en = nstl::min(oc + half_size + 1, C);

            for (dim_t c = c_st; c < c_en; ++c) {
                const acc_data_t s = src[data_off(mb, c, od, oh, ow)];
                sum += s * s;
            }
        } else {
            dim_t d_st = nstl::max(od - half_size + 0, (dim_t)0);
            dim_t d_en = nstl::min(od + half_size + 1, D);
            dim_t h_st = nstl::max(oh - half_size + 0, (dim_t)0);
            dim_t h_en = nstl::min(oh + half_size + 1, H);
            dim_t w_st = nstl::max(ow - half_size + 0, (dim_t)0);
            dim_t w_en = nstl::min(ow + half_size + 1, W);
            for_(dim_t d = d_st; d < d_en; ++d)
            for_(dim_t h = h_st; h < h_en; ++h)
            for (dim_t w = w_st; w < w_en; ++w) {
                const acc_data_t s = src[data_off(mb, oc, d, h, w)];
                sum += s * s;
            }
        }
        sum = k + alpha * sum / summands;
        const acc_data_t s = src[data_off(mb, oc, od, oh, ow)];
        d[0] = static_cast<data_t>(s * fast_negative_powf(sum, beta));
    };

    const dim_t MB = pd()->MB();
    if (tag == nChw16c || tag == nChw8c) {
        parallel_nd(MB, utils::div_up(C, blksize), H, W,
                [&](dim_t mb, dim_t c_blk, dim_t h, dim_t w) {
                    dim_t c = c_blk * blksize;
                    const dim_t off = mb * stride_mb + c * H * W
                            + (h * W + w) * blksize;
                    PRAGMA_OMP_SIMD()
                    for (dim_t cc = 0; cc < nstl::min(blksize, C - c); ++cc)
                        ker(&dst[off + cc], mb, c + cc, 0, h, w);
                });
    } else if (tag == nhwc) {
        parallel_nd(MB, H, W, C, [&](dim_t mb, dim_t h, dim_t w, dim_t c) {
            const dim_t off = mb * stride_mb + h * W * C + w * C + c;
            ker(&dst[off], mb, c, 0, h, w);
        });
    } else {
        parallel_nd(MB, C, D, H, W,
                [&](dim_t mb, dim_t c, dim_t d, dim_t h, dim_t w) {
                    const dim_t off = data_off(mb, c, d, h, w);
                    ker(&dst[off], mb, c, d, h, w);
                });
    }
}

// Backward LRN formula (refer to Forward LRN formula):
// Partial derivatives:
// dy_i/dx_j =         - 2*a*b/n * x_i * O(i)^-b / O(i) * x_j, i != j
//             O(i)^-b - 2*a*b/n * x_i * O(i)^-b / O(i) * x_j, i == j, where
// O(i) = (k + a / n * Sum:j [x_j^2]), j in [i - n/2, i + n/2]. Note: j depends
//     on i, which means that O(i) may use more points than local_size.
// Now, z_i = Sum:k [dE/dy_k * dy_k/dx_j], where k in [i - n/2, i + n/2]
//     for ACROSS. 2d-shape for WITHIN.
// Then, dE/dy_k = diffDst_k. Finally,
// z_i = Sum:k [dd_k * dy_k/dx_j] = A - B (code variables) =
//     = dd_i * O(i)^-b - 2*a*b/n * x_i * Sum:k {O(k)^-b / O(k) * x_k * dd_k};

template <impl::data_type_t d_type>
template <dnnl_format_tag_t tag>
void ref_lrn_bwd_t<d_type>::execute_backward(const exec_ctx_t &ctx) const {
    using namespace alg_kind;
    using namespace format_tag;

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->src_md());

    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const auto stride_mb = data_d.blocking_desc().strides[0];
    const bool across_channels = pd()->desc()->alg_kind == lrn_across_channels;
    static constexpr dim_t blksize = tag == nChw16c ? 16 : 8;
    const auto ndims = data_d.ndims();

    auto compute_n_summands = [&](dim_t size) {
        if (across_channels) {
            return size;
        } else { // within_channel
            dim_t n_summands = 1;
            for (auto d = ndims - 2; d > 0; --d)
                n_summands *= size;
            return n_summands;
        }
    };

    const acc_data_t alpha = static_cast<acc_data_t>(pd()->desc()->lrn_alpha);
    const acc_data_t beta = static_cast<acc_data_t>(pd()->desc()->lrn_beta);
    const acc_data_t k = static_cast<acc_data_t>(pd()->desc()->lrn_k);
    const dim_t size = pd()->desc()->local_size;
    const dim_t half_size = (size - 1) / 2;
    const dim_t summands = compute_n_summands(size);

    auto data_off = [&](dim_t mb, dim_t c, dim_t d, dim_t h, dim_t w) -> dim_t {
        switch (tag) {
            case nChw16c:
            case nChw8c:
                return mb * stride_mb + (c / blksize) * H * W * blksize
                        + h * W * blksize + w * blksize + c % blksize;
            case nchw: return mb * stride_mb + c * H * W + h * W + w;
            case nhwc: return mb * stride_mb + h * W * C + w * C + c;
            default:
                if (ndims >= 5) return data_d.off(mb, c, d, h, w);
                if (ndims >= 4) return data_d.off(mb, c, h, w);
                if (ndims >= 3) return data_d.off(mb, c, w);
                return data_d.off(mb, c);
        }
    };

    // pass by value due to icc170 and icc180 problem on KNL
    auto get_omega = [=](dim_t mb, dim_t oc, dim_t od, dim_t oh, dim_t ow) {
        acc_data_t sum = 0;
        if (across_channels) {
            const dim_t c_st = nstl::max(oc - half_size + 0, (dim_t)0);
            const dim_t c_en = nstl::min(oc + half_size + 1, C);

            for (dim_t c = c_st; c < c_en; ++c) {
                const acc_data_t s = src[data_off(mb, c, od, oh, ow)];
                sum += s * s;
            }
        } else {
            dim_t d_st = nstl::max(od - half_size + 0, (dim_t)0);
            dim_t d_en = nstl::min(od + half_size + 1, D);
            dim_t h_st = nstl::max(oh - half_size + 0, (dim_t)0);
            dim_t h_en = nstl::min(oh + half_size + 1, H);
            dim_t w_st = nstl::max(ow - half_size + 0, (dim_t)0);
            dim_t w_en = nstl::min(ow + half_size + 1, W);
            for_(dim_t d = d_st; d < d_en; ++d)
            for_(dim_t h = h_st; h < h_en; ++h)
            for (dim_t w = w_st; w < w_en; ++w) {
                const acc_data_t s = src[data_off(mb, oc, d, h, w)];
                sum += s * s;
            }
        }
        return (acc_data_t)(k + alpha * sum / summands);
    };

    // pass by value due to icc170 and icc180 problem on KNL
    auto ker = [=](data_t *d, dim_t mb, dim_t oc, dim_t od, dim_t oh,
                       dim_t ow) {
        acc_data_t A = 0, B = 0;
        if (across_channels) {
            const dim_t c_st = nstl::max(oc - half_size + 0, (dim_t)0);
            const dim_t c_en = nstl::min(oc + half_size + 1, C);

            for (dim_t c = c_st; c < c_en; c++) {
                const auto off = data_off(mb, c, od, oh, ow);
                const acc_data_t omega = get_omega(mb, c, od, oh, ow);
                const acc_data_t omega_in_beta
                        = fast_negative_powf(omega, beta);
                const acc_data_t tmp
                        = omega_in_beta * (acc_data_t)diff_dst[off];
                if (c == oc) A = tmp;
                B += (src[off] * tmp / omega);
            }
        } else {
            dim_t d_st = nstl::max(od - half_size + 0, (dim_t)0);
            dim_t d_en = nstl::min(od + half_size + 1, D);
            dim_t h_st = nstl::max(oh - half_size + 0, (dim_t)0);
            dim_t h_en = nstl::min(oh + half_size + 1, H);
            dim_t w_st = nstl::max(ow - half_size + 0, (dim_t)0);
            dim_t w_en = nstl::min(ow + half_size + 1, W);
            for_(dim_t d = d_st; d < d_en; ++d)
            for_(dim_t h = h_st; h < h_en; ++h)
            for (dim_t w = w_st; w < w_en; ++w) {
                const auto off = data_off(mb, oc, d, h, w);
                const acc_data_t omega = get_omega(mb, oc, d, h, w);
                const acc_data_t omega_in_beta
                        = fast_negative_powf(omega, beta);
                const acc_data_t tmp
                        = omega_in_beta * (acc_data_t)diff_dst[off];
                if (d == od && h == oh && w == ow) A = tmp;
                B += (src[off] * tmp / omega);
            }
        }
        const auto off = data_off(mb, oc, od, oh, ow);
        B *= (2.0f * alpha * beta * src[off] / summands);
        *d = static_cast<data_t>(A - B);
    };

    const dim_t MB = pd()->MB();
    if (tag == nChw16c || tag == nChw8c) {
        parallel_nd(MB, utils::div_up(C, blksize), H, W,
                [&](dim_t mb, dim_t c_blk, dim_t h, dim_t w) {
                    dim_t c = c_blk * blksize;
                    const dim_t off = mb * stride_mb + c * H * W
                            + (h * W + w) * blksize;
                    PRAGMA_OMP_SIMD()
                    for (dim_t cc = 0; cc < nstl::min(blksize, C - c); ++cc)
                        ker(&diff_src[off + cc], mb, c + cc, 0, h, w);
                });
    } else if (tag == nhwc) {
        parallel_nd(MB, H, W, C, [&](dim_t mb, dim_t h, dim_t w, dim_t c) {
            const dim_t off = mb * stride_mb + h * W * C + w * C + c;
            ker(&diff_src[off], mb, c, 0, h, w);
        });
    } else {
        parallel_nd(MB, C, D, H, W,
                [&](dim_t mb, dim_t c, dim_t d, dim_t h, dim_t w) {
                    const dim_t off = data_off(mb, c, d, h, w);
                    ker(&diff_src[off], mb, c, d, h, w);
                });
    }
}

template void
ref_lrn_fwd_t<data_type::f32>::execute_forward<format_tag::nChw16c>(
        const exec_ctx_t &ctx) const;
template void
ref_lrn_fwd_t<data_type::f32>::execute_forward<format_tag::nChw8c>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_fwd_t<data_type::f32>::execute_forward<format_tag::nchw>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_fwd_t<data_type::f32>::execute_forward<format_tag::nhwc>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_fwd_t<data_type::f32>::execute_forward<format_tag::any>(
        const exec_ctx_t &ctx) const;
template void
ref_lrn_bwd_t<data_type::f32>::execute_backward<format_tag::nChw16c>(
        const exec_ctx_t &ctx) const;
template void
ref_lrn_bwd_t<data_type::f32>::execute_backward<format_tag::nChw8c>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_bwd_t<data_type::f32>::execute_backward<format_tag::nchw>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_bwd_t<data_type::f32>::execute_backward<format_tag::nhwc>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_bwd_t<data_type::f32>::execute_backward<format_tag::any>(
        const exec_ctx_t &ctx) const;

template void
ref_lrn_fwd_t<data_type::bf16>::execute_forward<format_tag::nChw16c>(
        const exec_ctx_t &ctx) const;
template void
ref_lrn_fwd_t<data_type::bf16>::execute_forward<format_tag::nChw8c>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_fwd_t<data_type::bf16>::execute_forward<format_tag::nchw>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_fwd_t<data_type::bf16>::execute_forward<format_tag::nhwc>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_fwd_t<data_type::bf16>::execute_forward<format_tag::any>(
        const exec_ctx_t &ctx) const;
template void
ref_lrn_bwd_t<data_type::bf16>::execute_backward<format_tag::nChw16c>(
        const exec_ctx_t &ctx) const;
template void
ref_lrn_bwd_t<data_type::bf16>::execute_backward<format_tag::nChw8c>(
        const exec_ctx_t &ctx) const;
template void
ref_lrn_bwd_t<data_type::bf16>::execute_backward<format_tag::nchw>(
        const exec_ctx_t &ctx) const;
template void
ref_lrn_bwd_t<data_type::bf16>::execute_backward<format_tag::nhwc>(
        const exec_ctx_t &ctx) const;
template void ref_lrn_bwd_t<data_type::bf16>::execute_backward<format_tag::any>(
        const exec_ctx_t &ctx) const;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
