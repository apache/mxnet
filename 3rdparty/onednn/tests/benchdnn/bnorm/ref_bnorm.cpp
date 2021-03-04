/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "bnorm/bnorm.hpp"

namespace bnorm {

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &mean, const dnn_mem_t &var, const dnn_mem_t &ss,
        dnn_mem_t &ws, dnn_mem_t &dst, dnn_mem_t &src_hat) {
    const int64_t MB = prb->mb;
    const int64_t C = prb->ic;
    const int64_t D = prb->id;
    const int64_t H = prb->ih;
    const int64_t W = prb->iw;
    const bool use_scale_shift = prb->flags & USE_SCALESHIFT;
    const bool fuse_relu = prb->flags & FUSE_NORM_RELU;
    const bool need_ws = prb->need_ws();
    const auto &attr = prb->attr;

    dnnl::impl::parallel_nd(C, [&](int64_t c) {
        float smean = mean.get_elem(c);
        float svar = var.get_elem(c);
        float sqrt_var = sqrtf(svar + prb->eps);
        float rcp_denom = 1.f / sqrt_var;
        float gamma = use_scale_shift ? ss.get_elem(c) : 1.f;
        float beta = use_scale_shift ? ss.get_elem(C + c) : 0;

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(prb, mb, c, d, h, w);
            float x_hat = (src.get_elem(off) - smean) * rcp_denom;
            float res = gamma * x_hat + beta;
            if (fuse_relu && res < 0) res = 0;
            if (need_ws) ws.set_elem(off, !!res);
            maybe_post_ops(attr, res);
            dst.set_elem(off, res);
            if (prb->dir & FLAG_BWD) src_hat.set_elem(off, x_hat);
        }
    });
}

void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &src_hat,
        const dnn_mem_t &var, const dnn_mem_t &d_dst, const dnn_mem_t &ss,
        const dnn_mem_t &ws, dnn_mem_t &d_src, dnn_mem_t &d_ss) {
    const int64_t MB = prb->mb;
    const int64_t C = prb->ic;
    const int64_t D = prb->id;
    const int64_t H = prb->ih;
    const int64_t W = prb->iw;
    const bool glob_stats = prb->flags & GLOB_STATS;
    const bool use_scale_shift = prb->flags & USE_SCALESHIFT;
    const bool fuse_relu = prb->flags & FUSE_NORM_RELU;

    const float MB_SP = MB * D * H * W;

    dnnl::impl::parallel_nd(C, [&](int64_t c) {
        float rcp_denom = 1.f / sqrtf(var.get_elem(c) + prb->eps);
        float gamma = use_scale_shift ? ss.get_elem(c) : 1.f;

        float d_gamma = 0;
        float d_beta = 0;

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(prb, mb, c, d, h, w);
            float dd = d_dst.get_elem(off);
            if (fuse_relu && ws.get_elem(off) == 0) dd = 0;
            d_gamma += dd * src_hat.get_elem(off);
            d_beta += dd;
        }

        if (use_scale_shift && (prb->dir & FLAG_WEI)) {
            d_ss.set_elem(c, d_gamma);
            d_ss.set_elem(C + c, d_beta);
        }

        for_(int64_t mb = 0; mb < MB; ++mb)
        for_(int64_t d = 0; d < D; ++d)
        for_(int64_t h = 0; h < H; ++h)
        for (int64_t w = 0; w < W; ++w) {
            auto off = data_off(prb, mb, c, d, h, w);
            float dd = d_dst.get_elem(off);
            if (fuse_relu && ws.get_elem(off) == 0) dd = 0;
            float ds = dd;

            if (!glob_stats)
                ds -= (d_beta + src_hat.get_elem(off) * d_gamma) / MB_SP;

            d_src.set_elem(off, rcp_denom * ds * gamma);
        }
    });
}

} // namespace bnorm
