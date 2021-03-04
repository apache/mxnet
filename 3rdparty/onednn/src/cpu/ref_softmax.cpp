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
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::execute_forward_dense(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const auto ou_stride = pd()->outer_stride();

    parallel_nd(outer_size_, [&](int ou) {
        const data_t *src_data = src + ou * ou_stride;
        data_t *dst_data = dst + ou * ou_stride;
        float space_max = -FLT_MAX;
        float space_denom = 0;
        constexpr int unroll_factor = 32;

// Intel(R) C++ Compiler generates the maxps + shuffle pattern
// for the max search which works faster
#if !defined(__INTEL_COMPILER)
        // The code below makes the compiler generate maxps instruction.
        // rather than maxss, which is generated for the 'else' code path
        auto max_wrapper = [](float a, float b) { return nstl::max(a, b); };
        auto min_wrapper = [](int a, int b) { return nstl::min(a, b); };

        if (channels_ < unroll_factor) {
            float max_val = -FLT_MAX;
            for (int i = 0; i < channels_; i++) {
                max_val = max_wrapper(max_val, src_data[i]);
            }
            space_max = max_val;
        } else {
            float max_values[unroll_factor];

            for (int i = 0; i < unroll_factor; i++) {
                max_values[i] = src_data[i];
            }
            for (int i = unroll_factor; i < channels_; i += unroll_factor) {
                int offset = min_wrapper(i, channels_ - unroll_factor);
                for (int j = 0; j < unroll_factor; j++) {
                    max_values[j]
                            = max_wrapper(max_values[j], src_data[offset + j]);
                }
            }
            float max_val = -FLT_MAX;
            for (int i = 0; i < unroll_factor; i++) {
                max_val = max_wrapper(max_val, max_values[i]);
            }
            space_max = max_val;
        }
#else
        for (int c = 0; c < channels_; ++c)
            space_max = nstl::max(space_max, (float)src_data[c]);
#endif

        // sub + exp + sum
        int tail = channels_ % unroll_factor;
        for (int i = 0; i < channels_ - tail; i += unroll_factor) {
            PRAGMA_OMP_SIMD()
            for (int j = 0; j < unroll_factor; j++) {
                if (pd()->is_softmax()) {
                    float D = expf(src_data[i + j] - space_max);
                    space_denom += D;
                    dst_data[i + j] = D;
                } else if (pd()->is_logsoftmax()) {
                    float D = src_data[i + j] - space_max;
                    space_denom += expf(D);
                    dst_data[i + j] = D;
                }
            }
        }
        for (int i = channels_ - tail; i < channels_; i++) {
            if (pd()->is_softmax()) {
                float D = expf(src_data[i] - space_max);
                space_denom += D;
                dst_data[i] = D;
            } else if (pd()->is_logsoftmax()) {
                float D = src_data[i] - space_max;
                space_denom += expf(D);
                dst_data[i] = D;
            }
        }

        // scal
        if (pd()->is_softmax()) {
            space_denom = space_denom ? (1.f / space_denom) : 1.f;
        } else if (pd()->is_logsoftmax()) {
            space_denom = logf(space_denom);
        }
        for (int c = 0; c < channels_; ++c) {
            if (pd()->is_softmax()) {
                dst_data[c] = dst_data[c] * space_denom;
            } else if (pd()->is_logsoftmax()) {
                dst_data[c] = dst_data[c] - space_denom;
            }
        }
    });
}

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::execute_forward_generic(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());

    parallel_nd(outer_size_, [&](int ou) {
        float space_max_val = 0, space_denom_val = 0;
        float *space_max = &space_max_val, *space_denom = &space_denom_val;
        if (inner_size_ > 1) {
            using namespace memory_tracking::names;
            space_max = ctx.get_scratchpad_grantor().template get<float>(
                                key_softmax_reduction)
                    + ou * 2 * inner_size_;
            space_denom = space_max + inner_size_;
        }

        utils::array_set(space_max, -FLT_MAX, inner_size_);
        utils::array_set(space_denom, 0, inner_size_);

        for (int in = 0; in < inner_size_; in++) {
            dim_t ou_in_offset = ou * channels_ * inner_size_ + in;

            for (int c = 0; c < channels_; c++) {
                size_t off = data_d.off_l(ou_in_offset + c * inner_size_);
                space_max[in] = nstl::max(space_max[in], (float)src[off]);
            }

            for (int c = 0; c < channels_; c++) {
                size_t off = data_d.off_l(ou_in_offset + c * inner_size_);
                if (pd()->is_softmax()) {
                    float D = expf(src[off] - space_max[in]);
                    space_denom[in] += D;
                    dst[off] = D;
                } else if (pd()->is_logsoftmax()) {
                    float D = src[off] - space_max[in];
                    space_denom[in] += expf(D);
                    dst[off] = D;
                }
            }

            if (pd()->is_logsoftmax()) {
                space_denom[in] = logf(space_denom[in]);
            }

            for (int c = 0; c < channels_; c++) {
                size_t off = data_d.off_l(ou_in_offset + c * inner_size_);
                if (pd()->is_softmax()) {
                    dst[off] = dst[off] / space_denom[in];
                } else if (pd()->is_logsoftmax()) {
                    dst[off] = dst[off] - space_denom[in];
                }
            }
        }
    });
}

template struct ref_softmax_fwd_t<data_type::bf16>;
template struct ref_softmax_fwd_t<data_type::f32>;

// softmax along last physical dimension
template <impl::data_type_t data_type>
void ref_softmax_bwd_t<data_type>::execute_backward_dense(
        const exec_ctx_t &ctx) const {
    auto dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DST);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const auto ou_stride = pd()->outer_stride();

    parallel_nd(outer_size_, [&](int ou) {
        float sbr = 0;
        size_t off = ou * ou_stride;
        if (pd()->is_softmax()) {
            for (size_t loff = off; loff < off + channels_; ++loff)
                sbr += diff_dst[loff] * dst[loff];
            for (size_t loff = off; loff < off + channels_; ++loff)
                diff_src[loff] = dst[loff] * (diff_dst[loff] - sbr);
        } else if (pd()->is_logsoftmax()) {
            for (size_t loff = off; loff < off + channels_; ++loff)
                sbr += diff_dst[loff];
            for (size_t loff = off; loff < off + channels_; ++loff)
                diff_src[loff] = diff_dst[loff] - expf(dst[loff]) * sbr;
        }
    });
}

template <impl::data_type_t data_type>
void ref_softmax_bwd_t<data_type>::execute_backward_generic(
        const exec_ctx_t &ctx) const {
    auto dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DST);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_d(pd()->diff_src_md());
    const memory_desc_wrapper data_d(pd()->dst_md());

    parallel_nd(outer_size_, inner_size_, [&](int ou, int in) {
        dim_t ou_in_offset = ou * channels_ * inner_size_ + in;
        float sbr = 0;
        for (int c = 0; c < channels_; ++c) {
            auto off_diff = diff_d.off_l(ou_in_offset + c * inner_size_);
            if (pd()->is_softmax()) {
                auto off_data = data_d.off_l(ou_in_offset + c * inner_size_);
                sbr += diff_dst[off_diff] * dst[off_data];
            } else if (pd()->is_logsoftmax()) {
                sbr += diff_dst[off_diff];
            }
        }

        for (int c = 0; c < channels_; ++c) {
            auto off_diff = diff_d.off_l(ou_in_offset + c * inner_size_);
            auto off_data = data_d.off_l(ou_in_offset + c * inner_size_);
            if (pd()->is_softmax()) {
                diff_src[off_diff] = dst[off_data] * (diff_dst[off_diff] - sbr);
            } else if (pd()->is_logsoftmax()) {
                diff_src[off_diff]
                        = diff_dst[off_diff] - expf(dst[off_data]) * sbr;
            }
        }
    });
}

template struct ref_softmax_bwd_t<data_type::bf16>;
template struct ref_softmax_bwd_t<data_type::f32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
