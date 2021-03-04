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
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/matmul/gemm_f32_matmul.hpp"
#include "cpu/matmul/matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

using namespace data_type;

status_t gemm_f32_matmul_t::pd_t::init(engine_t *engine) {
    auto check_bias = [&]() -> bool {
        return !with_bias()
                || (weights_md(1)->data_type == f32 && is_bias_1xN());
    };

    bool ok = src_md()->data_type == src_type
            && weights_md()->data_type == weights_type
            && desc()->accum_data_type == acc_type
            && dst_md()->data_type == dst_type && check_bias()
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::post_ops)
            && set_default_formats()
            && gemm_based::check_gemm_compatible_formats(*this);

    if (!ok) return status::unimplemented;

    // set state
    params_.dst_is_acc_ = true;

    return check_and_configure_attributes();
}

status_t gemm_f32_matmul_t::pd_t::check_and_configure_attributes() {
    auto check_attr_oscale = [&]() -> bool {
        const auto &oscale = attr()->output_scales_;
        return oscale.mask_ == 0
                || (oscale.mask_ == (1 << 1) && batched() == false);
    };

    auto check_attr_post_ops = [&]() -> bool {
        using namespace primitive_kind;
        const auto &p = attr()->post_ops_;
        auto check_sum = [&](int idx) -> bool {
            return p.contain(sum, idx) && params_.gemm_applies_output_scales_;
        };
        switch (p.len()) {
            case 0: return true;
            case 1: return check_sum(0) || p.contain(eltwise, 0);
            case 2: return check_sum(0) && p.contain(eltwise, 1);
            default: return false;
        }
    };

    // check basic attributes
    if (!check_attr_oscale()) return status::unimplemented;

    // set state
    CHECK(params_.pp_attr_.copy_from(*attr()));
    params_.gemm_applies_output_scales_
            = attr()->output_scales_.mask_ == 0 && !with_bias();
    if (params_.gemm_applies_output_scales_)
        params_.pp_attr_.output_scales_.set(1.f);

    // check post-ops
    if (check_attr_post_ops()) {
        auto &po = params_.pp_attr_.post_ops_;
        const int sum_idx = 0;
        if (po.len() > 0 && po.contain(primitive_kind::sum, sum_idx)) {
            // set state
            params_.gemm_beta_ = po.entry_[sum_idx].sum.scale;
            // drop sum from pp_attributes, as it will be applied by gemm
            po.entry_.erase(po.entry_.begin());
        }
    } else {
        return status::unimplemented;
    }

    // set state
    params_.has_pp_kernel_
            = with_bias() || !params_.pp_attr_.has_default_values();

    return status::success;
}

status_t gemm_f32_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const weights_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    DEFINE_SCALES_BUFFER(scales);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

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

    const gemm_based::params_t &params = pd()->params();
    const float alpha = params.get_gemm_alpha(scales);
    const float beta = params.gemm_beta_;

    std::atomic<status_t> st(status::success);
    const bool parallel_over_batch = batch > 1;
    if (parallel_over_batch) {
        const int src_mask
                = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
        const int wei_mask
                = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);

        parallel(0, [&](int ithr, int nthr) {
            size_t batch_start {0}, batch_end {0};
            balance211((size_t)(batch), nthr, ithr, batch_start, batch_end);
            dims_t s_dims_idx, w_dims_idx, d_dims_idx;
            // account for M, N dims for index calculations
            utils::l_dims_by_l_offset(
                    d_dims_idx, batch_start * M * N, dst_d.dims(), ndims);

            for (size_t b = batch_start; b < batch_end; ++b) {
                utils::copy_dims_with_mask(
                        s_dims_idx, d_dims_idx, ndims, src_mask);
                utils::copy_dims_with_mask(
                        w_dims_idx, d_dims_idx, ndims, wei_mask);
                const src_data_t *curr_src = src + src_d.off_v(s_dims_idx);
                const weights_data_t *curr_weights
                        = weights + weights_d.off_v(w_dims_idx);
                dst_data_t *curr_dst = dst + dst_d.off_v(d_dims_idx);

                status_t st_thr = extended_sgemm(&transB, &transA, &N, &M, &K,
                        &alpha, curr_weights, &ldb, curr_src, &lda, &beta,
                        curr_dst, &ldc, nullptr, false);
                if (st_thr != status::success) {
                    st = st_thr;
                    return;
                }

                if (params.has_pp_kernel_) {
                    const float *pp_scales
                            = params.get_post_processing_scales(scales);
                    (*pp_kernel_)(curr_dst, curr_dst, bias, pp_scales, 0, M * N,
                            (size_t)N, ldc, nullptr);
                }
                utils::dim_iterator(dst_d.dims(), d_dims_idx, batch_ndims);
            }
        });
    } else {
        st = extended_sgemm(&transB, &transA, &N, &M, &K, &alpha, weights, &ldb,
                src, &lda, &beta, dst, &ldc, nullptr, false);
        if (st != status::success) return st;

        if (params.has_pp_kernel_) {
            const bool force_sequential = pp_kernel_->sequential_kernel();
            const float *pp_scales = params.get_post_processing_scales(scales);
            parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
                size_t start {}, end {};
                balance211((size_t)(M * N), nthr, ithr, start, end);
                (*pp_kernel_)(dst, dst, bias, pp_scales, start, end, (size_t)N,
                        ldc, nullptr);
            });
        }
    }

    return st;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
