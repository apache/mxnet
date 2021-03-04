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

#include "gpu/ocl/ocl_types.h"

#undef SRC_OFF
#undef DST_OFF

#define SRC_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#define STAT_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(STAT, x0, x1, x2, x3, x4, x5)

#if IS_FWD
KERNEL_ATTR
__kernel void ref_lnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst,
        __global float *scaleshift, float eps) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

    float v_mean = CALCULATE_STATS ? 0 : mean[s_off];
    float v_variance = CALCULATE_STATS ? 0 : variance[s_off];

    if (CALCULATE_STATS) {
        for (int c = 0; c < C; ++c) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            v_mean += SRC_TO_REF(src[src_off]);
        }
        v_mean /= C;

        for (int c = 0; c < C; ++c) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            float m = SRC_TO_REF(src[src_off]) - v_mean;
            v_variance += m * m;
        }
        v_variance /= C;
    }

    float sqrt_variance = sqrt(v_variance + eps);
    for (int c = 0; c < C; ++c) {
        float sm = (USE_SCALESHIFT ? scaleshift[c] : 1.0f) / sqrt_variance;
        float sv = USE_SCALESHIFT ? scaleshift[C + c] : 0.0f;

        x[NDIMS - 1] = c;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        dst[dst_off] = TO_DST(sm * (SRC_TO_REF(src[src_off]) - v_mean) + sv);
    }

    if (CALCULATE_STATS) {
        if (SAVE_STATS) {
            mean[s_off] = v_mean;
            variance[s_off] = v_variance;
        }
    }
}
#endif

#if IS_BWD
#if USE_SCALESHIFT
NAMED_KERNEL_ATTR(SCALESHIFT)
__kernel void ref_lnorm_bwd_scaleshift(__global DATA_T *src,
        __global float *mean, __global float *variance,
        __global DATA_T *diff_dst, __global float *diff_scaleshift, float eps) {

    int c = GWS_GET_C();
    int x[6] = {0};

    float diff_gamma = 0;
    float diff_beta = 0;

    for (x[0] = 0; x[0] < max(1, STAT_D0); ++x[0]) {
        for (x[1] = 0; x[1] < max(1, STAT_D1); ++x[1]) {
            for (x[2] = 0; x[2] < max(1, STAT_D2); ++x[2]) {
                for (x[3] = 0; x[3] < max(1, STAT_D3); ++x[3]) {
                    x[NDIMS - 1] = 0;
                    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

                    x[NDIMS - 1] = c;
                    int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
                    int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

                    float inv_sqrt_variance
                            = 1.0f / sqrt(variance[s_off] + eps);
                    float dd = DST_TO_REF(diff_dst[dst_off]);

                    diff_gamma += (SRC_TO_REF(src[src_off]) - mean[s_off]) * dd
                            * inv_sqrt_variance;
                    diff_beta += dd;
                }
            }
        }
    }
    diff_scaleshift[c] = diff_gamma;
    diff_scaleshift[C + c] = diff_beta;
}
#endif

KERNEL_ATTR
__kernel void ref_lnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scaleshift, __global DATA_T *diff_src, float eps) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

    float inv_sqrt_variance = 1.0f / sqrt(variance[s_off] + eps);
    float dd_gamma = 0;
    float dd_gamma_x = 0;

    if (CALCULATE_STATS) {
        for (int c = 0; c < C; ++c) {
            float gamma = USE_SCALESHIFT ? scaleshift[c] : 1.0f;

            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            float dd = DST_TO_REF(diff_dst[dst_off]);
            dd_gamma += dd * gamma;
            dd_gamma_x += dd * gamma * (SRC_TO_REF(src[src_off]) - mean[s_off]);
        }
        dd_gamma_x *= inv_sqrt_variance;
    }

    for (int c = 0; c < C; ++c) {
        float gamma = USE_SCALESHIFT ? scaleshift[c] : 1.0f;

        x[NDIMS - 1] = c;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        float v_diff_src = DST_TO_REF(diff_dst[dst_off]) * gamma;
        if (CALCULATE_STATS) {
            v_diff_src -= dd_gamma / C
                    + (SRC_TO_REF(src[src_off]) - mean[s_off]) * dd_gamma_x
                            * inv_sqrt_variance / C;
        }
        v_diff_src *= inv_sqrt_variance;
        diff_src[src_off] = TO_SRC(v_diff_src);
    }
}

#endif
