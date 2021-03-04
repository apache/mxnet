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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#if IS_FWD == 1

KERNEL_ATTR
__kernel void ref_inner_product_fwd(__global SRC_DATA_T *src,
        __global WEI_DATA_T *wei, __global BIA_DATA_T *bias,
        __global DST_DATA_T *dst POST_OP_ARGS, float output_scale) {

    const int mb = GWS_GET_MB();
    const int oc = GWS_GET_OC();

    ACC_DATA_T d = 0;
#if HAS_SPATIAL == 1
    for (int ic = 0; ic < IC; ++ic)
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {
                    const uint src_off = SRC_OFF(mb, ic, kd, kh, kw);
                    const uint wei_off = WEI_OFF(0, oc, ic, kd, kh, kw);
#else
    for (int ic = 0; ic < IC_TOTAL; ++ic) {
        const uint src_off = mb * IC_TOTAL + ic;
        const uint wei_off = oc * IC_TOTAL + ic;
#endif
                    d += SRC_TO_REF(src[src_off]) * WEI_TO_REF(wei[wei_off]);
                }
    DATA_T tmp = d;
#if WITH_BIAS
    tmp += BIA_TO_REF(bias[oc]);
#endif

    tmp *= output_scale;

    float dest_data;
#if WITH_SUM
    dest_data = DST_TO_REF(dst[mb * OC + oc]);
#endif

    APPLY_POST_OPS(tmp, DATA_T, dest_data, float, mb, 1, oc, 1, 0, 1, 0, 1, 0,
            1, 0, 1);

    dst[mb * OC + oc] = TO_DST(tmp);
}
#endif

#if IS_BWD_D == 1
KERNEL_ATTR
__kernel void ref_inner_product_bwd_data(__global SRC_DATA_T *diff_src,
        __global WEI_DATA_T *wei, __global DST_DATA_T *diff_dst) {

    const int mb = GWS_GET_MB_IC() / IC;
    const int ic = GWS_GET_MB_IC() % IC;
    const int kd = GWS_GET_KD();
    const int kh = GWS_GET_KH();
    const int kw = GWS_GET_KW();

    float ds = 0.0f;
    for (int oc = 0; oc < OC; ++oc) {
        const uint diff_dst_off = DST_OFF(mb, oc, 0, 0, 0);
        const uint wei_off = WEI_OFF(0, oc, ic, kd, kh, kw);
        ds += DST_TO_REF(diff_dst[diff_dst_off]) * WEI_TO_REF(wei[wei_off]);
    }
    const uint diff_src_off = SRC_OFF(mb, ic, kd, kh, kw);
    diff_src[diff_src_off] = REF_TO_SRC(ds);
}
#endif

#if IS_BWD_W == 1
KERNEL_ATTR
__kernel void ref_inner_product_bwd_weights(__global SRC_DATA_T *src,
        __global WEI_DATA_T *diff_wei, __global BIA_DATA_T *diff_bias,
        __global DST_DATA_T *diff_dst) {

    const int oc = GWS_GET_OC();
    const int ic = GWS_GET_IC();
    const int kd = GWS_GET_KD();
    const int kh = GWS_GET_KH();
    const int kw = GWS_GET_KW();

    float ds = 0.0f;
    for (int mb = 0; mb < MB; ++mb) {
        const uint diff_dst_off = DST_OFF(mb, oc, 0, 0, 0);
        const uint src_off = SRC_OFF(mb, ic, kd, kh, kw);
        ds += DST_TO_REF(diff_dst[diff_dst_off]) * SRC_TO_REF(src[src_off]);
    }
    const uint diff_wei_off = WEI_OFF(0, oc, ic, kd, kh, kw);
    diff_wei[diff_wei_off] = REF_TO_WEI(ds);
#if WITH_BIAS == 1
    if (ic == 0) {
        float db = 0.0f;
        for (int mb = 0; mb < MB; ++mb) {
            const uint diff_dst_off = DST_OFF(mb, oc, 0, 0, 0);
            db += DST_TO_REF(diff_dst[diff_dst_off]);
        }
        diff_bias[oc] = REF_TO_BIA(db);
    }
#endif
}
#endif
