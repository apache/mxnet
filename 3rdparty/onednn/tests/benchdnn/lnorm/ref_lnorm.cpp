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

#include "lnorm/lnorm.hpp"

namespace lnorm {

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src, dnn_mem_t &mean,
        dnn_mem_t &var, const dnn_mem_t &ss, dnn_mem_t &dst) {
    dnnl::impl::parallel_nd(prb->n, [&](int64_t n) {
        float smean = ((float *)mean)[n];
        float svar = ((float *)var)[n];
        float sqrt_var = sqrtf(svar + prb->eps);

        for (int64_t c = 0; c < prb->c; ++c) {
            float gamma
                    = (prb->flags & USE_SCALESHIFT ? ((float *)ss)[c] : 1.0f)
                    / sqrt_var;
            float beta = prb->flags & USE_SCALESHIFT ? ((float *)ss)[prb->c + c]
                                                     : 0;
            auto off = n * prb->c + c;
            float res = gamma * (((float *)src)[off] - smean) + beta;
            dst.set_elem(off, res);
        }
    });
}

void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &mean, const dnn_mem_t &var, const dnn_mem_t &d_dst,
        const dnn_mem_t &ss, dnn_mem_t &d_src, dnn_mem_t &d_ss) {

    if ((prb->flags & USE_SCALESHIFT) && (prb->dir & FLAG_WEI)) {
        dnnl::impl::parallel_nd(prb->c, [&](int64_t c) {
            float d_gamma = 0;
            float d_beta = 0;

            for (int64_t n = 0; n < prb->n; ++n) {
                float smean = ((float *)mean)[n];
                float svar = ((float *)var)[n];
                float rcp_denom = 1.f / sqrtf(svar + prb->eps);
                auto off = n * prb->c + c;
                float dd = ((float *)d_dst)[off];
                d_gamma += dd * (((float *)src)[off] - smean) * rcp_denom;
                d_beta += dd;
            }

            ((float *)d_ss)[c] = d_gamma;
            ((float *)d_ss)[prb->c + c] = d_beta;
        });
    }

    dnnl::impl::parallel_nd(prb->n, [&](int64_t n) {
        float smean = ((float *)mean)[n];
        float svar = ((float *)var)[n];
        float rcp_denom = 1.f / sqrtf(svar + prb->eps);
        float dd_gamma = 0, dd_gamma_x = 0;
        if (!(prb->flags & GLOB_STATS)) {
            for (int64_t c = 0; c < prb->c; ++c) {
                auto off = n * prb->c + c;
                float ds = ((float *)d_dst)[off];
                const float x = ((float *)src)[off] - smean;
                float gamma
                        = prb->flags & USE_SCALESHIFT ? ((float *)ss)[c] : 1;
                dd_gamma += gamma * ds;
                dd_gamma_x += gamma * ds * x;
            }
            dd_gamma_x *= rcp_denom;
        }
        for (int64_t c = 0; c < prb->c; ++c) {
            float gamma = prb->flags & USE_SCALESHIFT ? ((float *)ss)[c] : 1;
            auto off = n * prb->c + c;
            float ds = ((float *)d_dst)[off] * gamma;
            if (!(prb->flags & GLOB_STATS)) {
                const float x = ((float *)src)[off] - smean;
                ds -= (dd_gamma + x * dd_gamma_x * rcp_denom) / prb->c;
            }

            ((float *)d_src)[off] = rcp_denom * ds;
        }
    });
}

} // namespace lnorm
