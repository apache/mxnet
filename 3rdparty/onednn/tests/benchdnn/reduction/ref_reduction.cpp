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

#include <limits>
#include <math.h>

#include "tests/test_thread.hpp"

#include "common.hpp"
#include "dnnl_memory.hpp"

#include "reduction.hpp"

namespace reduction {

void init_acc(float &acc, alg_t alg) {
    switch (alg) {
        case MAX: acc = std::numeric_limits<float>::lowest(); break;
        case MIN: acc = std::numeric_limits<float>::max(); break;
        case SUM: acc = 0.0f; break;
        case MUL: acc = 1.0f; break;
        case MEAN:
        case NORM_LP_MAX:
        case NORM_LP_SUM:
        case NORM_LP_POWER_P_MAX:
        case NORM_LP_POWER_P_SUM: acc = 0.0f; break;
        default: assert(!"unknown algorithm");
    }
}

void accumulate(float &dst, const float src, alg_t alg, float p, float eps) {
    switch (alg) {
        case MAX: dst = MAX2(dst, src); break;
        case MIN: dst = MIN2(dst, src); break;
        case MEAN:
        case SUM: dst += src; break;
        case MUL: dst *= src; break;
        case NORM_LP_MAX:
        case NORM_LP_SUM:
        case NORM_LP_POWER_P_MAX:
        case NORM_LP_POWER_P_SUM: dst += pow(fabs(src), p); break;
        default: assert(!"unknown algorithm");
    }
}

void finalize(float &dst, alg_t alg, float p, float eps, dnnl_dim_t n) {
    switch (alg) {
        case MEAN: dst /= n; break;
        case NORM_LP_MAX:
            dst = MAX2(dst, eps);
            dst = pow(dst, 1.0f / p);
            break;
        case NORM_LP_SUM:
            dst += eps;
            dst = pow(dst, 1.0f / p);
            break;
        case NORM_LP_POWER_P_MAX: dst = MAX2(dst, eps); break;
        case NORM_LP_POWER_P_SUM: dst += eps; break;
        default: break;
    }
}

void compute_ref(const prb_t *prb, const dnn_mem_t &src, dnn_mem_t &dst) {
    float *dst_ptr = (float *)dst;
    const float *src_ptr = (const float *)src;

    const auto &ndims = prb->ndims;
    const auto &src_dims = prb->src_dims;
    const auto &dst_dims = prb->dst_dims;

    const auto alg = prb->alg;
    const auto p = prb->p;
    const auto eps = prb->eps;

    dims_t reduce_dims(ndims, 1);
    int64_t reduce_size {1}, idle_size {1};

    for (int d = 0; d < ndims; ++d) {
        const bool is_reduction_dim = src_dims[d] != dst_dims[d];
        if (is_reduction_dim) {
            reduce_dims[d] = src_dims[d];
            reduce_size *= reduce_dims[d];
        } else {
            idle_size *= dst_dims[d];
        }
    }

    if (reduce_size == 1) return;

    dnnl::impl::parallel_nd(idle_size, [&](int64_t f) {
        dims_t idle_pos = off2dims_idx(dst_dims, f);
        const int64_t dst_off = md_off_v(dst.md_, idle_pos.data());
        const int64_t src_idle_off = md_off_v(src.md_, idle_pos.data());
        float acc {0.0f};
        init_acc(acc, alg);
        for (int64_t r = 0; r < reduce_size; ++r) {
            dims_t reduce_pos = off2dims_idx(reduce_dims, r);
            const int64_t src_reduce_off = md_off_v(src.md_, reduce_pos.data());
            const int64_t src_off = src_idle_off + src_reduce_off;
            accumulate(acc, src_ptr[src_off], alg, p, eps);
        }
        finalize(acc, alg, p, eps, reduce_size);
        dst_ptr[dst_off] = acc;
    });
}

} // namespace reduction
