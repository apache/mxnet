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

#include "tests/test_thread.hpp"

#include "softmax/softmax.hpp"

namespace softmax {

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src, dnn_mem_t &dst) {
    int64_t outer_size {0}, inner_size {0}, axis_size {0};
    get_sizes(prb, outer_size, inner_size, axis_size);

    const float *src_ptr = (const float *)src;
    float *dst_ptr = (float *)dst;
    const auto alg = prb->alg;

    dnnl::impl::parallel_nd(
            outer_size, inner_size, [&](int64_t ou, int64_t in) {
                float space_denom = 0.;
                float space_max = -FLT_MAX;
                int64_t ou_in_offset = ou * axis_size * inner_size + in;

                for (int64_t as = 0; as < axis_size; ++as) {
                    int64_t idx = ou_in_offset + as * inner_size;
                    space_max = MAX2(space_max, src_ptr[idx]);
                }

                for (int64_t as = 0; as < axis_size; ++as) {
                    int64_t idx = ou_in_offset + as * inner_size;
                    if (alg == SOFTMAX) {
                        float D = dst_ptr[idx] = expf(src_ptr[idx] - space_max);
                        space_denom += D;
                    } else if (alg == LOGSOFTMAX) {
                        float D = dst_ptr[idx] = src_ptr[idx] - space_max;
                        space_denom += expf(D);
                    }
                }

                if (alg == SOFTMAX) {
                    space_denom = space_denom ? (1.f / space_denom) : 1.f;
                } else if (alg == LOGSOFTMAX) {
                    space_denom = logf(space_denom);
                }

                for (int64_t as = 0; as < axis_size; ++as) {
                    int64_t idx = ou_in_offset + as * inner_size;
                    if (alg == SOFTMAX) {
                        dst_ptr[idx] *= space_denom;
                    } else if (alg == LOGSOFTMAX) {
                        dst_ptr[idx] -= space_denom;
                    }
                }
            });
}

void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &dst,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_src) {
    int64_t outer_size {0}, inner_size {0}, axis_size {0};
    get_sizes(prb, outer_size, inner_size, axis_size);

    const float *dst_ptr = (const float *)dst;
    const float *d_dst_ptr = (const float *)diff_dst;
    float *d_src_ptr = (float *)diff_src;
    const auto alg = prb->alg;

    dnnl::impl::parallel_nd(
            outer_size, inner_size, [&](int64_t ou, int64_t in) {
                float part_deriv_sum = 0.;
                int64_t ou_in_offset = ou * axis_size * inner_size + in;

                for (int64_t as = 0; as < axis_size; ++as) {
                    int64_t idx = ou_in_offset + as * inner_size;
                    if (alg == SOFTMAX) {
                        part_deriv_sum += d_dst_ptr[idx] * dst_ptr[idx];
                    } else if (alg == LOGSOFTMAX) {
                        part_deriv_sum += d_dst_ptr[idx];
                    }
                }

                for (int64_t as = 0; as < axis_size; ++as) {
                    int64_t idx = ou_in_offset + as * inner_size;
                    if (alg == SOFTMAX) {
                        d_src_ptr[idx] = dst_ptr[idx]
                                * (d_dst_ptr[idx] - part_deriv_sum);
                    } else if (alg == LOGSOFTMAX) {
                        d_src_ptr[idx] = d_dst_ptr[idx]
                                - expf(dst_ptr[idx]) * part_deriv_sum;
                    }
                }
            });
}

} // namespace softmax
