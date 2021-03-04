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

#include <atomic>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/gemm_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

namespace {
struct im_pos_t {
    im_pos_t() : n {0}, g {0}, od {0}, sp {0}, ic {0}, oc {0} {}
    int n, g, od, sp, ic, oc;
    bool do_im2col(const im_pos_t &prev) const {
        return true
                && (n != prev.n || g != prev.g || od != prev.od || sp != prev.sp
                        || ic != prev.ic);
    }
};
} // namespace

status_t gemm_convolution_fwd_t::execute_forward_nspc(
        const exec_ctx_t &ctx) const {
    auto src_base = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst_base = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();
    const conv_gemm_conf_t &jcp = pd()->jcp_;
    std::atomic<status_t> st(status::success);

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        status_t st_thr = execute_forward_thr_nspc(
                ithr, nthr, src_base, wei_base, bia_base, dst_base, scratchpad);
        if (st_thr != status::success) st = st_thr;
    });

    return st;
}

status_t gemm_convolution_fwd_t::execute_forward_thr_nspc(const int ithr,
        const int nthr, const data_t *src_base, const data_t *wei_base,
        const data_t *bia_base, data_t *dst_base,
        const memory_tracking::grantor_t &scratchpad) const {
    const conv_gemm_conf_t &jcp = pd()->jcp_;

    // Src Format: mb-spatial-groups-input_channels
    const size_t src_mb_stride = static_cast<size_t>(jcp.id) * jcp.ih * jcp.iw
            * jcp.ngroups * jcp.ic;
    const size_t src_g_stride = jcp.ic;
    // Wei Format: spatial-input_channels-groups-output_channels
    const size_t wei_g_stride = pd()->with_groups() ? jcp.oc : 0;

    // Dst Format: mb-spatial-groups-output_channels
    const size_t dst_mb_stride = static_cast<size_t>(jcp.od) * jcp.oh * jcp.ow
            * jcp.ngroups * jcp.oc;
    const size_t dst_g_stride = jcp.oc;
    const size_t dst_os_stride = jcp.ngroups * jcp.oc;

    data_t *__restrict col = scratchpad.get<data_t>(key_conv_gemm_col)
            + (ptrdiff_t)ithr * jcp.im2col_sz;
    data_t *__restrict imtr = scratchpad.get<data_t>(key_conv_gemm_imtr)
            + (ptrdiff_t)ithr * jcp.is * jcp.ic;

    int g {0}, n {0}, ohb {0}, owb {0};
    size_t start = 0, end = 0;
    const bool is_problem_3d = pd()->ndims() == 5;

    assert(IMPLICATION(is_problem_3d,
            jcp.oh_block == jcp.oh && jcp.ow_block == jcp.ow
                    && jcp.ic_block == jcp.ic));
    assert(IMPLICATION(jcp.ow_block != jcp.ow, jcp.oh_block == 1));

    const int nb_oh = div_up(jcp.oh, jcp.oh_block);
    const int nb_ow = div_up(jcp.ow, jcp.ow_block);
    // threads share work across mini-batch, groups, and blocked width/height
    const size_t work_amount
            = static_cast<size_t>(jcp.mb) * jcp.ngroups * nb_oh * nb_ow;
    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ohb, nb_oh, owb, nb_ow);

    if (jcp.im2col_sz && is_problem_3d) {
        // jit_gemm_convolution_utils::im2col_dt_3d() requires external
        // data initialization by zeroes
        PRAGMA_OMP_SIMD()
        for (ptrdiff_t i = 0; i < jcp.im2col_sz; i++)
            col[i] = 0.0f;
    }
    for (size_t iwork = start; iwork < end; ++iwork) {
        int oh = ohb * jcp.oh_block;
        int ow = owb * jcp.ow_block;
        const data_t *__restrict src
                = src_base + n * src_mb_stride + g * src_g_stride;
        const data_t *__restrict wei = wei_base + g * wei_g_stride;

        const int h_step = nstl::min(jcp.oh_block, jcp.oh - oh);
        const int w_step = nstl::min(jcp.ow_block, jcp.ow - ow);
        if (jcp.im2col_sz && is_problem_3d) {
            jit_gemm_convolution_utils::transpose_dt(jcp, src, imtr);
        }

        for (int od = 0; od < jcp.od; od++) {
            data_t *__restrict dst = dst_base + n * dst_mb_stride
                    + g * dst_g_stride
                    + ((od * jcp.oh + oh) * jcp.ow + ow) * dst_os_stride;
            if (jcp.im2col_sz) {
                if (is_problem_3d)
                    jit_gemm_convolution_utils::im2col_dt_3d<data_t, data_t>(
                            jcp, imtr, col, od);
                else
                    jit_gemm_convolution_utils::im2col_dt<data_t, data_t>(
                            jcp, src, imtr, col, oh, h_step, ow, w_step);
            }

            const dim_t M = jcp.oc;
            const dim_t K = jcp.ks * jcp.ic;
            const dim_t N = h_step * w_step;
            const dim_t LDA = M * jcp.ngroups;
            const dim_t LDB = jcp.im2col_sz ? N : K * jcp.ngroups;
            const dim_t LDC = M * jcp.ngroups;
            const char *BT = jcp.im2col_sz ? "T" : "N";
            const data_t onef = 1.f;
            const float beta = this->beta_;
            const data_t *__restrict src_od
                    = src + od * jcp.oh * jcp.ow * jcp.ngroups * jcp.ic;
            status_t st = extended_sgemm("N", BT, &M, &N, &K, &onef, wei, &LDA,
                    jcp.im2col_sz ? col : (data_t *)src_od, &LDB, &beta, dst,
                    &LDC);
            if (st != status::success) return st;

            if (jcp.with_bias || eltwise_) {
                parallel(0, [&](int ithr, int nthr) {
                    size_t start, end;
                    balance211((size_t)N * jcp.oc, nthr, ithr, start, end);

                    const size_t first_oc = start % jcp.oc;
                    const size_t last_oc = (end - 1) % jcp.oc;
                    const size_t first_os = start / jcp.oc;
                    const size_t last_os = (end - 1) / jcp.oc;

                    for (size_t os = first_os; os <= last_os; ++os) {
                        const size_t start_oc = (os == first_os) ? first_oc : 0;
                        const size_t end_oc
                                = (os == last_os) ? last_oc : jcp.oc - 1;

                        const data_t *__restrict bia_arr
                                = bia_base + g * jcp.oc;
                        data_t *__restrict dst_arr = dst + os * dst_os_stride;

                        if (jcp.with_bias) {
                            PRAGMA_OMP_SIMD()
                            for (size_t oc = start_oc; oc <= end_oc; oc++) {
                                dst_arr[oc] += bia_arr[oc];
                            }
                        }

                        // fast branch for ReLU case
                        if (eltwise_
                                && eltwise_->alg_ == alg_kind::eltwise_relu) {
                            const auto alpha = eltwise_->alpha_;
                            const auto scale = eltwise_->scale_;
                            PRAGMA_OMP_SIMD()
                            for (size_t oc = start_oc; oc <= end_oc; oc++) {
                                if (dst_arr[oc] < 0) dst_arr[oc] *= alpha;
                                dst_arr[oc] *= scale;
                            }
                        } else if (eltwise_) {
                            for (size_t oc = start_oc; oc <= end_oc; oc++) {
                                dst_arr[oc]
                                        = eltwise_->compute_scalar(dst_arr[oc]);
                            }
                        }
                    }
                });
            }
        }
        nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ohb, nb_oh, owb, nb_ow);
    }
    return status::success;
}

status_t gemm_convolution_fwd_t::execute_forward_ncsp(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    auto col = ctx.get_scratchpad_grantor().get<data_t>(key_conv_gemm_col);

    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t weights_oc_size = jcp.ic * jcp.ks;
    const size_t weights_g_size = weights_oc_size * jcp.oc;
    const bool is_problem_3d = pd()->ndims() == 5;

    assert(IMPLICATION(is_problem_3d,
            jcp.os_block == jcp.os && jcp.ic_block == jcp.ic
                    && jcp.os_nb_block == 1));

    status_t st = status::success;
    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

        // non-blocked jit_gemm_convolution_utils::im2col_3d() requires
        // external data initialization by zeroes
        const bool outer_padding = jcp.os_nb_block == 1;
        if (outer_padding && is_problem_3d) {
            for (ptrdiff_t i = 0; i < jcp.im2col_sz; i++)
                _col[i] = (data_t)0;
        }
        auto inner_ker = [&](int spatial, const im_pos_t &curr, im_pos_t &prev,
                                 im_pos_t &step, const im_pos_t &end) {
            const data_t *_src
                    = src + (curr.n * jcp.ngroups + curr.g) * src_step;
            step.oc = nstl::min(
                    jcp.oc_block, nstl::min(jcp.oc, end.oc) - curr.oc);
            step.sp = nstl::min(jcp.os_block,
                    nstl::min(jcp.os - curr.sp, end.sp - spatial));
            step.ic = nstl::min(
                    jcp.ic_block, nstl::min(jcp.ic, end.ic) - curr.ic);
            bool do_im2col = curr.do_im2col(prev);
            prev = curr;

            if (jcp.im2col_sz && do_im2col) {
                if (!is_problem_3d)
                    jit_gemm_convolution_utils::im2col<float>(jcp, _src, _col,
                            curr.sp, step.sp, curr.ic, step.ic);
                else
                    jit_gemm_convolution_utils::im2col_3d<float>(
                            jcp, _src, _col, curr.od, 0, jcp.os);
            }
            const data_t one = 1.0;

            const dim_t M = jcp.os * jcp.od;
            const size_t dst_step = jcp.oc * M;
            const dim_t m = step.sp;
            const dim_t LDA = jcp.im2col_sz ? m : M;
            data_t *_dst = dst + (curr.n * jcp.ngroups + curr.g) * dst_step
                    + curr.oc * M + curr.od * jcp.os + curr.sp;
            const dim_t K = step.ic * jcp.ks;
            const dim_t LDB = jcp.ic * jcp.ks;
            const dim_t N = step.oc;

            // TODO: what if this->beta_ != 0 && != 1 ?
            const float beta = (curr.ic == 0) ? this->beta_ : one;
            const float *_source = jcp.im2col_sz
                    ? _col
                    : _src + curr.ic * M + curr.od * jcp.os + curr.sp;
            const data_t *_weights = weights + curr.g * weights_g_size
                    + curr.oc * weights_oc_size + curr.ic * jcp.ks;

            status_t st = extended_sgemm("N", "N", &m, &N, &K, &one, _source,
                    &LDA, _weights, &LDB, &beta, _dst, &M);
            if (st != status::success) return st;

            if (curr.ic == jcp.ic - step.ic) {
                // TODO: for "outer threading" we have parallel section within
                // outermost "parallel". It is not good. Consider to use
                // "parallel" here with number of threads passed as parameter
                const int oc_start = curr.g * jcp.oc + curr.oc;
                if (eltwise_) {
                    // fast branch for ReLU case
                    if (eltwise_->alg_ == alg_kind::eltwise_relu) {
                        parallel_nd(step.oc, [&](const int oc) {
                            data_t b = jcp.with_bias ? bias[oc_start + oc] : 0;
                            data_t *d_ = _dst + oc * M;
                            PRAGMA_OMP_SIMD()
                            for (int oS = 0; oS < m; ++oS) {
                                d_[oS] += b;
                                if (d_[oS] < 0) d_[oS] *= eltwise_->alpha_;
                                d_[oS] *= eltwise_->scale_;
                            }
                        });
                    } else {
                        parallel_nd(step.oc, [&](const int oc) {
                            data_t b = jcp.with_bias ? bias[oc_start + oc] : 0;
                            data_t *d_ = _dst + oc * M;
                            PRAGMA_OMP_SIMD()
                            for (int oS = 0; oS < m; ++oS) {
                                d_[oS] += b;
                                d_[oS] = eltwise_->compute_scalar(d_[oS]);
                            }
                        });
                    }
                } else if (jcp.with_bias) {
                    parallel_nd(step.oc, [&](const int oc) {
                        data_t b = bias[oc_start + oc];
                        data_t *d_ = _dst + oc * M;
                        PRAGMA_OMP_SIMD()
                        for (int oS = 0; oS < m; ++oS) {
                            d_[oS] += b;
                        }
                    });
                }
            }

            return status::success;
        };
        im_pos_t start, end;
        end.ic = jcp.ic;

        if (!is_problem_3d) {
            const int sp_work = jcp.mb * jcp.ngroups * jcp.od * jcp.os;
            balance2D(nthr, ithr, sp_work, start.sp, end.sp, jcp.oc, start.oc,
                    end.oc, jcp.nthr_oc);
        } else {
            const int sp_work = jcp.mb * jcp.ngroups * jcp.od;
            balance2D(nthr, ithr, sp_work, start.sp, end.sp, jcp.oc, start.oc,
                    end.oc, jcp.nthr_oc);
            start.sp *= jcp.os;
            end.sp *= jcp.os;
        }

        im_pos_t curr, prev, step;
        prev.n = prev.g = prev.od = prev.sp = prev.ic = -1;
        step.oc = jcp.oc_block;
        step.sp = jcp.os_block;
        step.ic = jcp.ic_block;

        if (jcp.loop_order == gemm_loop_rlb)
            for (curr.ic = 0; curr.ic < jcp.ic; curr.ic += step.ic)
                for (int spatial = start.sp; spatial < end.sp;
                        spatial += step.sp) {
                    nd_iterator_init(spatial, curr.n, jcp.mb, curr.g,
                            jcp.ngroups, curr.od, jcp.od, curr.sp, jcp.os);
                    for (curr.oc = start.oc; curr.oc < end.oc;
                            curr.oc += step.oc) {
                        status_t st_thr
                                = inner_ker(spatial, curr, prev, step, end);
                        if (st_thr != status::success) {
                            st = st_thr;
                            return;
                        }
                    }
                }
        else if (jcp.loop_order == gemm_loop_lrb)
            for (int spatial = start.sp; spatial < end.sp; spatial += step.sp) {
                nd_iterator_init(spatial, curr.n, jcp.mb, curr.g, jcp.ngroups,
                        curr.od, jcp.od, curr.sp, jcp.os);
                for (curr.ic = 0; curr.ic < jcp.ic; curr.ic += step.ic)
                    for (curr.oc = start.oc; curr.oc < end.oc;
                            curr.oc += step.oc) {
                        status_t st_thr
                                = inner_ker(spatial, curr, prev, step, end);
                        if (st_thr != status::success) {
                            st = st_thr;
                            return;
                        }
                    }
            }
        else
            st = status::unimplemented;
    });

    return st;
}

status_t gemm_convolution_bwd_data_t::execute_backward_data_nspc(
        const exec_ctx_t &ctx) const {

    auto diff_dst_base = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto wei_base = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto diff_src_base = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    auto scratchpad = ctx.get_scratchpad_grantor();
    const conv_gemm_conf_t &jcp = pd()->jcp_;
    std::atomic<status_t> st(status::success);

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        status_t st_thr = execute_backward_data_thr_nspc(ithr, nthr,
                diff_dst_base, wei_base, bia_base, diff_src_base, scratchpad);
        if (st_thr != status::success) st = st_thr;
    });

    return st;
}

status_t gemm_convolution_bwd_data_t::execute_backward_data_thr_nspc(
        const int ithr, const int nthr, const data_t *diff_dst_base,
        const data_t *wei_base, const data_t *bia_base, data_t *diff_src_base,
        const memory_tracking::grantor_t &scratchpad) const {
    const conv_gemm_conf_t &jcp = pd()->jcp_;

    // Diff_dst Format: mb-spatial-groups-output_channels
    const size_t diff_dst_mb_stride = static_cast<size_t>(jcp.od) * jcp.oh
            * jcp.ow * jcp.ngroups * jcp.oc;
    const size_t diff_dst_g_stride = jcp.oc;

    // Wei Format: spatial-input_channels-groups-output_channels
    const size_t wei_g_stride = pd()->with_groups() ? jcp.oc : 0;

    // Diff_src Format: mb-spatial-groups-input_channels
    const size_t diff_src_mb_stride = static_cast<size_t>(jcp.id) * jcp.ih
            * jcp.iw * jcp.ngroups * jcp.ic;
    const size_t diff_src_g_stride = jcp.ic;
    const size_t diff_src_os_stride = jcp.ngroups * jcp.ic;

    // threads share work across mini-batch and groups
    const size_t work_amount = jcp.ngroups * jcp.mb;

    data_t *__restrict col = scratchpad.get<data_t>(key_conv_gemm_col)
            + (ptrdiff_t)ithr * jcp.im2col_sz;
    const bool acc_needed = jcp.ngroups > 1;
    data_t *__restrict acc = acc_needed
            ? scratchpad.get<data_t>(key_conv_gemm_acc)
                    + (ptrdiff_t)ithr * jcp.is * jcp.id * jcp.ic
            : nullptr;

    int n {0}, g {0};
    size_t start = 0, end = 0;

    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups);

    for (size_t iwork = start; iwork < end; ++iwork) {
        const data_t *__restrict diff_dst = diff_dst_base
                + n * diff_dst_mb_stride + g * diff_dst_g_stride;
        const data_t *__restrict wei = wei_base + g * wei_g_stride;
        data_t *__restrict diff_src = diff_src_base + n * diff_src_mb_stride
                + g * diff_src_g_stride;

        const dim_t M = jcp.ks * jcp.ic;
        const dim_t N = jcp.os * jcp.od;
        const dim_t K = jcp.oc;

        const data_t onef = 1.0f, zerof = 0.0f;
        const dim_t LD = K * jcp.ngroups;

        status_t st = extended_sgemm("T", "N", &M, &N, &K, &onef, wei, &LD,
                diff_dst, &LD, &zerof,
                jcp.im2col_sz ? col : (acc_needed ? acc : diff_src), &M);
        if (st != status::success) return st;

        if (jcp.im2col_sz)
            jit_gemm_convolution_utils::col2im_dt<data_t>(
                    jcp, col, (acc_needed ? acc : diff_src));

        if (acc_needed) {
            parallel_nd(static_cast<size_t>(jcp.is) * jcp.id, [&](size_t is) {
                data_t *__restrict diff_src_arr
                        = diff_src + is * diff_src_os_stride;
                const data_t *__restrict acc_arr = acc + is * jcp.ic;
                PRAGMA_OMP_SIMD()
                for (int ic = 0; ic < jcp.ic; ic++) {
                    diff_src_arr[ic] = acc_arr[ic];
                }
            });
        }
        nd_iterator_step(n, jcp.mb, g, jcp.ngroups);
    }
    return status::success;
}

status_t gemm_convolution_bwd_data_t::execute_backward_data_ncsp(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    auto col = ctx.get_scratchpad_grantor().get<data_t>(key_conv_gemm_col);

    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    const dim_t M = jcp.os * jcp.od;
    const size_t src_step = (size_t)jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = (size_t)jcp.oc * M;
    const size_t weights_g_size = (size_t)jcp.ic * jcp.oc * jcp.ks;

    const dim_t m = jcp.os_block;
    const dim_t K = jcp.oc;
    const dim_t N = jcp.ic * jcp.ks;

    const size_t work_amount = (size_t)jcp.ngroups * jcp.mb;
    const bool is_problem_3d = pd()->ndims() == 5;

    std::atomic<status_t> st(status::success);
    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

        int g {0}, n {0};
        size_t start = 0, end = 0;
        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb);
        for (size_t iwork = start; iwork < end; ++iwork) {

            data_t *_diff_src = diff_src + (n * jcp.ngroups + g) * src_step;
            if (is_problem_3d && jcp.im2col_sz > 0) {
                // jit_gemm_convolution_utils::col2im_3d() assumes that the
                // accumulator is initialized by zeroes
                for (size_t i = 0; i < src_step; i++)
                    _diff_src[i] = (data_t)0;
            }

            const data_t *_weights = weights + g * weights_g_size;
            for_(int od = 0; od < jcp.od; ++od)
            for (int os_nb = 0; os_nb < jcp.os_nb_block; ++os_nb) {
                auto out_off = os_nb * m + od * jcp.os;
                const data_t *_diff_dst
                        = diff_dst + (n * jcp.ngroups + g) * dst_step + out_off;
                const dim_t os_block
                        = nstl::min((dim_t)jcp.os_block, jcp.os - os_nb * m);
                const dim_t LDC = jcp.im2col_sz ? os_block : M;

                const data_t zero = 0.0, one = 1.0;
                status_t st_thr = extended_sgemm("N", "T", &os_block, &N, &K,
                        &one, _diff_dst, &M, _weights, &N, &zero,
                        jcp.im2col_sz ? _col : _diff_src + out_off, &LDC);
                if (st_thr != status::success) {
                    st = st_thr;
                    return;
                }

                if (jcp.im2col_sz) {
                    if (!is_problem_3d)
                        jit_gemm_convolution_utils::col2im(jcp, _col, _diff_src,
                                os_nb * jcp.os_block, os_block);
                    else {
                        jit_gemm_convolution_utils::col2im_3d(jcp, _col,
                                _diff_src, od, os_nb * jcp.os_block, os_block);
                    }
                }
            }
            nd_iterator_step(g, jcp.ngroups, n, jcp.mb);
        }
    });

    return st;
}

status_t gemm_convolution_bwd_weights_t::execute_backward_weights_nspc(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_BIAS);

    auto col = ctx.get_scratchpad_grantor().get<data_t>(key_conv_gemm_col);
    const conv_gemm_conf_t &jcp = pd()->jcp_;

    auto wei_reduction
            = ctx.get_scratchpad_grantor().get<data_t>(key_conv_wei_reduction);

    const dim_t K = jcp.os * static_cast<size_t>(jcp.od);
    const size_t src_step
            = static_cast<size_t>(jcp.ic) * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * K;
    const size_t weights_g_size = jcp.oc;

    const dim_t k = jcp.os;
    const dim_t M = jcp.oc;
    const dim_t N = static_cast<dim_t>(jcp.ic) * jcp.ks;
    const dim_t LDB = jcp.ngroups * jcp.oc;
    const dim_t LDA = jcp.im2col_sz ? jcp.oh * jcp.ow : jcp.ngroups * jcp.ic;
    const bool is_problem_3d = pd()->ndims() == 5;

    std::atomic<status_t> st(status::success);
    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int ithr_g, nthr_g, ithr_mb, nthr_mb;
        size_t g_start {0}, g_end {0}, mb_start {0}, mb_end {0};

        const int mb_for_balance = jcp.need_wei_reduction ? jcp.mb : 1;
        jit_gemm_convolution_utils::bwd_weights_balance(ithr, nthr, jcp.ngroups,
                mb_for_balance, ithr_g, nthr_g, ithr_mb, nthr_mb);

        assert(IMPLICATION(!jcp.need_wei_reduction, nthr_mb == 1));

        const int need_reduction = nthr_mb != 1;
        const dim_t LDC = need_reduction ? jcp.oc : jcp.ngroups * jcp.oc;
        data_t *__restrict imtr
                = ctx.get_scratchpad_grantor().get<data_t>(key_conv_gemm_imtr)
                + (ptrdiff_t)ithr * jcp.id * jcp.ic * jcp.is;

        if (ithr_g != -1 && ithr_mb != -1) {
            balance211((size_t)jcp.ngroups, nthr_g, ithr_g, g_start, g_end);
            balance211((size_t)jcp.mb, nthr_mb, ithr_mb, mb_start, mb_end);

            assert(IMPLICATION((g_end - g_start) > 1, need_reduction == 0));

            data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;
            if (is_problem_3d) {
                // jit_gemm_convolution_utils::im2col_3d() requires external
                // data initialization by zeroes
                PRAGMA_OMP_SIMD()
                for (ptrdiff_t i = 0; i < jcp.im2col_sz; i++)
                    _col[i] = 0.0f;
            }

            data_t *weights_reduce_base = wei_reduction
                    + ithr_g * nthr_mb * weights_g_size * jcp.ks * jcp.ic;
            data_t *weights_reduce = weights_reduce_base
                    + ithr_mb * weights_g_size * jcp.ks * jcp.ic;

            for (size_t g = g_start; g < g_end; ++g) {
                data_t *_diff_weights = need_reduction
                        ? weights_reduce
                        : diff_weights + g * weights_g_size;
                for (size_t mb = mb_start; mb < mb_end; ++mb) {
                    const data_t *_src
                            = src + mb * jcp.ngroups * src_step + g * jcp.ic;
                    if (jcp.im2col_sz && is_problem_3d)
                        jit_gemm_convolution_utils::transpose_dt(
                                jcp, _src, imtr);
                    for (int od = 0; od < jcp.od; ++od) {
                        const data_t *_diff_dst = diff_dst
                                + mb * jcp.ngroups * dst_step
                                + od * k * jcp.ngroups * jcp.oc + g * jcp.oc;

                        if (jcp.im2col_sz) {
                            if (is_problem_3d)
                                jit_gemm_convolution_utils::im2col_dt_3d<data_t,
                                        data_t>(jcp, imtr, _col, od);
                            else
                                jit_gemm_convolution_utils::im2col_dt<data_t,
                                        data_t>(jcp, _src, imtr, _col, 0,
                                        jcp.oh, 0, jcp.ow);
                        }
                        const data_t zero = 0.0f, one = 1.0f;
                        status_t st_thr = extended_sgemm("N",
                                jcp.im2col_sz ? "N" : "T", &M, &N, &k, &one,
                                _diff_dst, &LDB,
                                jcp.im2col_sz
                                        ? _col
                                        : _src + od * k * jcp.ngroups * jcp.ic,
                                &LDA, mb == mb_start && od == 0 ? &zero : &one,
                                _diff_weights, &LDC);
                        if (st_thr != status::success) {
                            st = st_thr;
                            // Finish the loops early if failure occured.
                            g = g_end;
                            mb = mb_end;
                            od = jcp.od;
                        }
                    }
                }
            }
            if (need_reduction && dnnl_thr_syncable()) {
                dnnl_thr_barrier();
                if (st != status::success) return;
                jit_gemm_convolution_utils::bwd_weights_reduction_par_nspc(
                        ithr_mb, nthr_mb, g_start, g_end, jcp,
                        weights_reduce_base, diff_weights);
            }
        } else {
            if (need_reduction && dnnl_thr_syncable()) dnnl_thr_barrier();
        }
    });

    if (jcp.need_wei_reduction && !dnnl_thr_syncable()) {
        parallel(jcp.nthr, [&](const int ithr, const int nthr) {
            int ithr_g, nthr_g, ithr_mb, nthr_mb;
            size_t g_start {0}, g_end {0};
            size_t mb_start {0}, mb_end {0};
            const int mb_for_balance = jcp.need_wei_reduction ? jcp.mb : 1;
            jit_gemm_convolution_utils::bwd_weights_balance(ithr, nthr,
                    jcp.ngroups, mb_for_balance, ithr_g, nthr_g, ithr_mb,
                    nthr_mb);

            assert(IMPLICATION(!jcp.need_wei_reduction, nthr_mb == 1));
            const int need_reduction = nthr_mb != 1;

            if (need_reduction && ithr_g != -1 && ithr_mb != -1) {
                balance211((size_t)jcp.ngroups, nthr_g, ithr_g, g_start, g_end);
                balance211((size_t)jcp.mb, nthr_mb, ithr_mb, mb_start, mb_end);

                assert(IMPLICATION((g_end - g_start) > 1, need_reduction == 0));

                data_t *weights_reduce_base = wei_reduction
                        + ithr_g * nthr_mb * weights_g_size * jcp.ic * jcp.ks;

                jit_gemm_convolution_utils::bwd_weights_reduction_par_nspc(
                        ithr_mb, nthr_mb, g_start, g_end, jcp,
                        weights_reduce_base, diff_weights);
            }
        });
    }

    if (jcp.with_bias) {
        parallel_nd(jcp.ngroups, jcp.oc, [&](int g, int oc) {
            data_t db = 0;
            const size_t offset_base = g * jcp.oc + oc;
            for_(int mb = 0; mb < jcp.mb; ++mb)
            for_(int od = 0; od < jcp.od; ++od)
            for (int oh = 0; oh < jcp.oh; ++oh) {
                const data_t *__restrict diff_dst_arr = diff_dst + offset_base
                        + ((static_cast<size_t>(mb) * jcp.od + od) * jcp.oh
                                  + oh)
                                * jcp.ow * jcp.ngroups * jcp.oc;
                const int width_stride = jcp.ngroups * jcp.oc;

                PRAGMA_OMP_SIMD(reduction(+ : db))
                for (int ow = 0; ow < jcp.ow; ++ow) {
                    db += diff_dst_arr[ow * width_stride];
                }
            }
            diff_bias[g * jcp.oc + oc] = db;
        });
    }
    return st;
}

status_t gemm_convolution_bwd_weights_t::execute_backward_weights_ncsp(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_BIAS);

    auto col = ctx.get_scratchpad_grantor().get<data_t>(key_conv_gemm_col);
    auto wei_reduction
            = ctx.get_scratchpad_grantor().get<data_t>(key_conv_wei_reduction);

    const conv_gemm_conf_t &jcp = this->pd()->jcp_;

    const dim_t K = jcp.os * jcp.od;
    const size_t src_step = jcp.ic * jcp.ih * jcp.iw * jcp.id;
    const size_t dst_step = jcp.oc * K;
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    const dim_t k = jcp.os_block;
    const dim_t N = jcp.oc;
    const dim_t M = jcp.ic * jcp.ks;
    const bool is_problem_3d = pd()->ndims() == 5;

    std::atomic<status_t> st(status::success);
    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int ithr_g, nthr_g, ithr_mb, nthr_mb;
        size_t g_start {0}, g_end {0}, mb_start {0}, mb_end {0};

        const int mb_for_balance = jcp.need_wei_reduction ? jcp.mb : 1;
        jit_gemm_convolution_utils::bwd_weights_balance(ithr, nthr, jcp.ngroups,
                mb_for_balance, ithr_g, nthr_g, ithr_mb, nthr_mb);

        assert(IMPLICATION(!jcp.need_wei_reduction, nthr_mb == 1));
        const int need_reduction = nthr_mb != 1;

        if (ithr_g != -1 && ithr_mb != -1) {
            balance211((size_t)jcp.ngroups, nthr_g, ithr_g, g_start, g_end);
            balance211((size_t)jcp.mb, nthr_mb, ithr_mb, mb_start, mb_end);

            assert(IMPLICATION((g_end - g_start) > 1, need_reduction == 0));

            data_t *_col = col + (ptrdiff_t)ithr * jcp.im2col_sz;

            // non-blocked jit_gemm_convolution_utils::im2col_3d() requires
            // external data initialization by zeroes
            const bool outer_padding = jcp.os_nb_block == 1;
            if (outer_padding && is_problem_3d) {
                for (ptrdiff_t i = 0; i < jcp.im2col_sz; i++)
                    _col[i] = (data_t)0;
            }
            data_t *weights_reduce_base
                    = wei_reduction + ithr_g * nthr_mb * weights_g_size;
            data_t *weights_reduce
                    = weights_reduce_base + ithr_mb * weights_g_size;

            for (size_t g = g_start; g < g_end; ++g) {
                data_t *_diff_weights = need_reduction
                        ? weights_reduce
                        : (diff_weights + g * weights_g_size);
                for (size_t mb = mb_start; mb < mb_end; ++mb) {
                    const data_t *_src
                            = src + (mb * jcp.ngroups + g) * src_step;
                    for_(int od = 0; od < jcp.od; ++od)
                    for (int os_nb = 0; os_nb < jcp.os_nb_block; ++os_nb) {
                        auto out_off = os_nb * k + od * jcp.os;
                        const dim_t os_block = nstl::min(
                                (dim_t)jcp.os_block, jcp.os - os_nb * k);
                        const data_t *_diff_dst = diff_dst
                                + (mb * jcp.ngroups + g) * dst_step + out_off;

                        if (jcp.im2col_sz) {
                            if (!is_problem_3d)
                                jit_gemm_convolution_utils::im2col<float>(jcp,
                                        _src, _col, os_nb * jcp.os_block,
                                        os_block, 0, jcp.ic);
                            else
                                jit_gemm_convolution_utils::im2col_3d<float>(
                                        jcp, _src, _col, od,
                                        os_nb * jcp.os_block, os_block);
                        }
                        const dim_t LDA = jcp.im2col_sz ? os_block : K;
                        const data_t zero = 0.0, one = 1.0;
                        status_t st_thr = extended_sgemm("T", "N", &M, &N,
                                &os_block, &one,
                                jcp.im2col_sz ? _col : _src + out_off, &LDA,
                                _diff_dst, &K,
                                mb == mb_start && os_nb == 0 && od == 0 ? &zero
                                                                        : &one,
                                _diff_weights, &M);
                        if (st_thr != status::success) {
                            st = st_thr;
                            // Finish the loops early if failure occured.
                            g = g_end;
                            mb = mb_end;
                            od = jcp.od;
                            os_nb = jcp.os_nb_block;
                        }
                    }
                }
            }
            if (need_reduction && dnnl_thr_syncable()) {
                dnnl_thr_barrier();
                if (st != status::success) return;
                data_t *weights_base = diff_weights + g_start * weights_g_size;
                jit_gemm_convolution_utils::bwd_weights_reduction_par_ncsp(
                        ithr_mb, nthr_mb, jcp, weights_reduce_base,
                        weights_base);
            }
        } else {
            if (need_reduction && dnnl_thr_syncable()) dnnl_thr_barrier();
        }
    });

    if (st != status::success) return st;

    if (jcp.need_wei_reduction && !dnnl_thr_syncable()) {
        parallel(jcp.nthr, [&](const int ithr, const int nthr) {
            int ithr_g, nthr_g, ithr_mb, nthr_mb;
            size_t g_start {0}, g_end {0};
            const int mb_for_balance = jcp.need_wei_reduction ? jcp.mb : 1;
            jit_gemm_convolution_utils::bwd_weights_balance(ithr, nthr,
                    jcp.ngroups, mb_for_balance, ithr_g, nthr_g, ithr_mb,
                    nthr_mb);

            assert(IMPLICATION(!jcp.need_wei_reduction, nthr_mb == 1));
            const int need_reduction = nthr_mb != 1;

            if (need_reduction && ithr_g != -1 && ithr_mb != -1) {
                balance211((size_t)jcp.ngroups, nthr_g, ithr_g, g_start, g_end);

                assert(IMPLICATION((g_end - g_start) > 1, need_reduction == 0));

                data_t *weights_reduce_base
                        = wei_reduction + ithr_g * nthr_mb * weights_g_size;
                data_t *weights_base = diff_weights + g_start * weights_g_size;

                jit_gemm_convolution_utils::bwd_weights_reduction_par_ncsp(
                        ithr_mb, nthr_mb, jcp, weights_reduce_base,
                        weights_base);
            }
        });
    }

    if (jcp.with_bias) {
        parallel_nd(jcp.ngroups, jcp.oc, [&](int g, int oc) {
            data_t db = 0;
            size_t offset_ = (size_t)g * dst_step + (size_t)oc * K;
            for (int mb = 0; mb < jcp.mb; ++mb) {
                size_t offset = offset_ + (size_t)mb * jcp.ngroups * dst_step;
                for_(int od = 0; od < jcp.od; ++od)
                for (int oh = 0; oh < jcp.oh; ++oh)
                    PRAGMA_OMP_SIMD(reduction(+ : db))
                for (int ow = 0; ow < jcp.ow; ++ow) {
                    db += diff_dst[offset];
                    offset++;
                }
            }
            diff_bias[g * jcp.oc + oc] = db;
        });
    }

    return st;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
