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

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_types.h"

#define BLOCK_READ_DST(data, idx) \
    data = AS_INT8_T( \
            intel_sub_group_block_read8((__global uint *)&current_dst[idx]));

#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_bwd_data_x8s8s32x(const __global uchar *src, const __global char *wei,
        const __global float *bias, __global DATA_T *dst) {

    const int mb_blocks = 2;

    const int group_ic = get_group_id(0) * IC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP / mb_blocks;
    const int group_sp = get_group_id(1) * SP_GROUP;

    const int sub_group_id = get_sub_group_id();
    const int mb = get_group_id(2) % mb_blocks;
    const int ic = (sub_group_id % IC_GROUP);
    const int sp = (sub_group_id / IC_GROUP);

    const int g = (group_ic + ic) / IC_NCHUNK;
    const int group_oc = OC_NCHUNK * g;

    const int gid = group_sp / (IW_PADDED * IH);
    const int gihw = group_sp % (IW_PADDED * IH);
    const int gih = gihw / IW_PADDED;
    const int giw = gihw % IW_PADDED;

    const int local_ih = sp / IW_PADDED;
    const int local_iw = sp % IW_PADDED;

    const int id = gid;
    const int iw = giw + local_iw;
    const int ih = gih + local_ih;

    if (iw >= IW) return;

    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * (group_ic + ic);
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK / 2 * mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw);

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * group_oc;
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK / 2 * mb;

    wei += OC_BLOCK * KD * KH * KW * IC_BLOCK * (group_ic + ic) * OC_NCHUNK;

    int8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    int8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;

    __attribute__((opencl_unroll_hint)) for (int oc_chunk = 0;
                                             oc_chunk < OC_NCHUNK; oc_chunk++) {
        INT8_T D0, D1;
        int8 W0, W1, W2, W3;
        for (int kd = 0; kd < KD; kd++) {
            if ((id + PD - kd * (1 + DD)) % SD != 0) {
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            const int od = (id + PD - kd * (1 + DD)) / SD;
            if (od < 0 || od >= OD) {
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            for (int kh = 0; kh < KH; kh++) {
                if ((ih + PH - kh * (1 + DH)) % SH != 0) {
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }
                const int oh = (ih + PH - kh * (1 + DH)) / SH;
                if (oh < 0 || oh >= OH) {
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }
                __attribute__((opencl_unroll_hint)) for (int kw = 0; kw < KW;
                                                         kw++) {
                    if ((iw + PW - kw * (1 + DW)) % SW == 0) {
                        const int ow = (iw + PW - kw * (1 + DW)) / SW;
                        if (ow >= 0 && ow < OW) {
                            __global DATA_T *current_dst = dst
                                    + OC_BLOCK * MB_BLOCK
                                            * (OW * OH * od + OW * oh + ow);
                            BLOCK_READ_DST(D0, 0);
#if MB > 8
                            BLOCK_READ_DST(D1, 8 * IC_BLOCK);
#endif // MB > 8
                            BLOCK_READ_WHT(W0, 0);
                            BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
                            BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
                            BLOCK_READ_WHT(W3, 24 * IC_BLOCK);
                            C00 = mmad8x8(D0, W0, C00);
                            C01 = mmad8x8(D0, W1, C01);
                            C02 = mmad8x8(D0, W2, C02);
                            C03 = mmad8x8(D0, W3, C03);
#if MB > 8
                            C10 = mmad8x8(D1, W0, C10);
                            C11 = mmad8x8(D1, W1, C11);
                            C12 = mmad8x8(D1, W2, C12);
                            C13 = mmad8x8(D1, W3, C13);
#endif // MB > 8
                        }
                    }
                    wei += IC_BLOCK * OC_BLOCK;
                }
            }
        }
        dst += OC_BLOCK * MB_BLOCK * OD * OH * OW;
    }

#if WITH_BIAS
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST) \
    TMP = (float)ACC + BIA; \
    RES = TO_SRC(TMP);
#else // WITH_BIAS
#define BIAS_SUM_RELU(RES, TMP, ACC, BIA, DST) RES = TO_SRC((float)ACC);
#endif // WITH_BIAS

#define PACK(idx) \
    BIAS_SUM_RELU(D00[0], T00, C00[idx], b0, S00[0]); \
    BIAS_SUM_RELU(D00[1], T01, C01[idx], b1, S00[1]); \
    BIAS_SUM_RELU(D00[2], T02, C02[idx], b2, S00[2]); \
    BIAS_SUM_RELU(D00[3], T03, C03[idx], b3, S00[3]); \
    T0[idx] = as_uint(D00); \
    BIAS_SUM_RELU(D01[0], T10, C10[idx], b0, S01[0]); \
    BIAS_SUM_RELU(D01[1], T11, C11[idx], b1, S01[1]); \
    BIAS_SUM_RELU(D01[2], T12, C12[idx], b2, S01[2]); \
    BIAS_SUM_RELU(D01[3], T13, C13[idx], b3, S01[3]); \
    T1[idx] = as_uint(D01);

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, (group_ic + ic) * IC_BLOCK);
    float b0 = bia[0];
    float b1 = bia[1];
    float b2 = bia[2];
    float b3 = bia[3];
#endif // WITH_BIAS

    uchar4 D00, D01;
    uint8 T0, T1;
    float T00, T01, T02, T03;
    float T10, T11, T12, T13;
    PACK(0);
    PACK(1);
    PACK(2);
    PACK(3);
    PACK(4);
    PACK(5);
    PACK(6);
    PACK(7);

    intel_sub_group_block_write_uc16(
            (__global uchar *)&src[0 * IC_BLOCK], as_uchar16(T0.s0123));
    intel_sub_group_block_write_uc16(
            (__global uchar *)&src[4 * IC_BLOCK], as_uchar16(T0.s4567));
#if MB > 8
    intel_sub_group_block_write_uc16(
            (__global uchar *)&src[8 * IC_BLOCK], as_uchar16(T1.s0123));
    intel_sub_group_block_write_uc16(
            (__global uchar *)&src[12 * IC_BLOCK], as_uchar16(T1.s4567));
#endif // MB > 8
}
