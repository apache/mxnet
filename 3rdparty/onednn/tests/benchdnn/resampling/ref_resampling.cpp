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
#include <math.h>

#include "tests/test_thread.hpp"

#include "resampling/resampling.hpp"

namespace resampling {

float linear_map(const int64_t y, const int64_t y_max, const int64_t x_max) {
    const float s = (y + 0.5f) * x_max / y_max;
    return s - 0.5f;
}
int64_t left(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return MAX2((int64_t)floorf(linear_map(y, y_max, x_max)), (int64_t)0);
}
int64_t right(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return MIN2((int64_t)ceilf(linear_map(y, y_max, x_max)), x_max - 1);
}
int64_t near(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return roundf(linear_map(y, y_max, x_max));
}
float weight(const int64_t y, const int64_t y_max, const int64_t x_max) {
    return fabs(linear_map(y, y_max, x_max) - left(y, y_max, x_max));
}

void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src, dnn_mem_t &dst) {
    int64_t MB = prb->mb;
    int64_t IC = prb->ic;
    int64_t ID = prb->id;
    int64_t IH = prb->ih;
    int64_t IW = prb->iw;
    int64_t OD = prb->od;
    int64_t OH = prb->oh;
    int64_t OW = prb->ow;

    auto ker_nearest = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh,
                               int64_t ow) {
        const int64_t id = near(od, OD, ID);
        const int64_t ih = near(oh, OH, IH);
        const int64_t iw = near(ow, OW, IW);
        const auto dst_off = dst_off_f(prb, mb, ic, od, oh, ow);
        dst.set_elem(dst_off, src.get_elem(src_off_f(prb, mb, ic, id, ih, iw)));
    };
    auto ker_linear = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh,
                              int64_t ow) {
        const int64_t id[2] = {left(od, OD, ID), right(od, OD, ID)};
        const int64_t ih[2] = {left(oh, OH, IH), right(oh, OH, IH)};
        const int64_t iw[2] = {left(ow, OW, IW), right(ow, OW, IW)};
        const float wd[2] = {1.f - weight(od, OD, ID), weight(od, OD, ID)};
        const float wh[2] = {1.f - weight(oh, OH, IH), weight(oh, OH, IH)};
        const float ww[2] = {1.f - weight(ow, OW, IW), weight(ow, OW, IW)};

        float cd[2][2] = {{0}};
        for_(int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            cd[i][j] = src.get_elem(src_off_f(prb, mb, ic, id[0], ih[i], iw[j]))
                            * wd[0]
                    + src.get_elem(src_off_f(prb, mb, ic, id[1], ih[i], iw[j]))
                            * wd[1];

        float ch[2] = {0};
        for (int i = 0; i < 2; i++)
            ch[i] = cd[0][i] * wh[0] + cd[1][i] * wh[1];

        float cw = ch[0] * ww[0] + ch[1] * ww[1];

        const auto dst_off = dst_off_f(prb, mb, ic, od, oh, ow);
        dst.set_elem(dst_off, cw);
    };

    dnnl::impl::parallel_nd(MB, IC, OD, OH, OW,
            [&](int64_t mb, int64_t ic, int64_t od, int64_t oh, int64_t ow) {
                if (prb->alg == nearest) {
                    ker_nearest(mb, ic, od, oh, ow);
                } else {
                    ker_linear(mb, ic, od, oh, ow);
                }
            });
}

void compute_ref_bwd(
        const prb_t *prb, dnn_mem_t &diff_src, const dnn_mem_t &diff_dst) {
    int64_t MB = prb->mb;
    int64_t IC = prb->ic;
    int64_t ID = prb->id;
    int64_t IH = prb->ih;
    int64_t IW = prb->iw;
    int64_t OD = prb->od;
    int64_t OH = prb->oh;
    int64_t OW = prb->ow;

    auto ker_nearest = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh,
                               int64_t ow) {
        const auto diff_dst_off = dst_off_f(prb, mb, ic, od, oh, ow);
        float diff_dst_val = diff_dst.get_elem(diff_dst_off);
        const int64_t id = near(od, OD, ID);
        const int64_t ih = near(oh, OH, IH);
        const int64_t iw = near(ow, OW, IW);
        ((float *)diff_src)[src_off_f(prb, mb, ic, id, ih, iw)] += diff_dst_val;
    };
    auto ker_linear = [&](int64_t mb, int64_t ic, int64_t od, int64_t oh,
                              int64_t ow) {
        const auto diff_dst_off = dst_off_f(prb, mb, ic, od, oh, ow);
        float diff_dst_val = diff_dst.get_elem(diff_dst_off);
        const int64_t id[2] = {left(od, OD, ID), right(od, OD, ID)};
        const int64_t ih[2] = {left(oh, OH, IH), right(oh, OH, IH)};
        const int64_t iw[2] = {left(ow, OW, IW), right(ow, OW, IW)};
        const float wd[2] = {1.f - weight(od, OD, ID), weight(od, OD, ID)};
        const float wh[2] = {1.f - weight(oh, OH, IH), weight(oh, OH, IH)};
        const float ww[2] = {1.f - weight(ow, OW, IW), weight(ow, OW, IW)};
        for_(int i = 0; i < 2; i++)
        for_(int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) {
            ((float *)diff_src)[src_off_f(prb, mb, ic, id[i], ih[j], iw[k])]
                    += wd[i] * wh[j] * ww[k] * diff_dst_val;
        }
    };

    // zeroing diff_src for correct result
    dnnl::impl::parallel_nd(MB, IC, ID, IH, IW,
            [&](int64_t mb, int64_t ic, int64_t id, int64_t ih, int64_t iw) {
                diff_src.set_elem(src_off_f(prb, mb, ic, id, ih, iw), 0.);
            });

    dnnl::impl::parallel_nd(MB, IC, [&](int64_t mb, int64_t ic) {
        for_(int64_t od = 0; od < OD; ++od)
        for_(int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow)
            if (prb->alg == nearest) {
                ker_nearest(mb, ic, od, oh, ow);
            } else {
                ker_linear(mb, ic, od, oh, ow);
            }
    });
}

} // namespace resampling
