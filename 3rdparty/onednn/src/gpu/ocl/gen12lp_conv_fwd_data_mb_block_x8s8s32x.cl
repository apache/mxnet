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
#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"
#include "gpu/ocl/ocl_zero_points.h"

#define SRC_DATA_BLOCK_T MMAD_DATA8_T
#define AS_SRC_DATA_BLOCK_T AS_MMAD_DATA8_T

#define BLOCK_READ_SRC(data, idx) \
    data = AS_SRC_DATA_BLOCK_T( \
            intel_sub_group_block_read8((__global uint *)&src[idx]));

#define BLOCK_READ_WHT(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

#if OC % OC_BLOCK == 0
#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

#else
#define BLOCK_READ_BIA(data, idx) \
    data = (float4)0; \
    int i; \
    for (i = idx; i < idx + OC_BLOCK && i < OC - (OC % SUB_GROUP_SIZE); \
            i += SUB_GROUP_SIZE) { \
        data[(i - idx) / SUB_GROUP_SIZE] = as_float( \
                intel_sub_group_block_read((__global uint *)&bias[i])); \
    } \
    if ((get_sub_group_local_id() < OC % SUB_GROUP_SIZE) \
            && (i == OC - OC % SUB_GROUP_SIZE)) { \
        data[(i - idx) / SUB_GROUP_SIZE] \
                = as_float(bias[i + get_sub_group_local_id()]); \
    }

#endif

#define BLOCK_READ_SCALES(data, idx) \
    data = as_float4(intel_sub_group_block_read4( \
            (__global uint *)&scales_per_oc[idx]));

#if SCALES_PER_OC
#define SCALE scales
#elif SCALES_COMMON
#define SCALE scale
#else
#define SCALE 1
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_fwd_mb_block_x8s8s32x(const __global uchar *src, const __global char *wei,
        const __global float *bias, __global DST_DATA_T *dst POST_OP_ARGS,
        float scale, const __global float *scales_per_oc,
        const __global int *src_compensation, const __global int *src_zpoints,
        const __global int *dst_compensation) {
#ifdef MB_FULL_BLOCK
    const int mb_blocks = 1;
#else // MB_FULL_BLOCK
    const int mb_blocks = 2;
#endif // MB_FULL_BLOCK

    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP / mb_blocks;
    const int group_sp = get_group_id(1) * SP_GROUP;

    // XXX: Each work-group calculates 16 mb instead of 32
    const int mb = get_group_id(2) % mb_blocks;
    const int sub_group_id = get_sub_group_id();
    const int ocl_local_id = get_local_id(0);
    const int oc = (sub_group_id % OC_GROUP);
    const int sp = (sub_group_id / OC_GROUP);

    const int g = (group_oc + oc) / OC_NCHUNK;
    const int group_ic = IC_NCHUNK * g;

    const int god = group_sp / (OW_PADDED * OH);
    const int gohw = group_sp % (OW_PADDED * OH);
    const int goh = gohw / OW_PADDED;
    const int gow = gohw % OW_PADDED;

    const int gid = god * SD;
    const int gih = goh * SH;
    const int giw = gow * SW;

    const int local_oh = sp / OW_PADDED;
    const int local_ow = sp % OW_PADDED;
    const int local_ih = local_oh * SH;
    const int local_iw = local_ow * SW;

    const int od = god;
    const int ow = gow + local_ow;
    const int oh = goh + local_oh;
    const int id = gid - PD;
    const int iw = giw + local_iw - PW;
    const int ih = gih + local_ih - PH;

    if (ow >= OW) return;

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK / mb_blocks * mb; // XXX: Why this offset?
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);

    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * group_ic;
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK / mb_blocks * mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw);

    wei += IC_BLOCK * KD * KH * KW * OC_BLOCK * (group_oc + oc) * IC_NCHUNK;

    int8 C00 = 0, C01 = 0, C02 = 0, C03 = 0;
    int8 C10 = 0, C11 = 0, C12 = 0, C13 = 0;
    int8 C20 = 0, C21 = 0, C22 = 0, C23 = 0;
    int8 C30 = 0, C31 = 0, C32 = 0, C33 = 0;

    for (int ic_chunk = 0; ic_chunk < IC_NCHUNK; ic_chunk++) {

        SRC_DATA_BLOCK_T S0, S1, S2, S3;
        int8 W0, W1, W2, W3;
        for (int kd = 0; kd < KD; kd++) {
#if WITH_SRC_ZPOINTS
            const int is_pad_d
                    = kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID;
#else
            if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
                src += IC_BLOCK * MB_BLOCK * IH * IW * (1 + DD);
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
#endif // WITH_SRC_ZPOINTS
            for (int kh = 0; kh < KH; kh++) {
#if WITH_SRC_ZPOINTS
                const int is_pad_h
                        = kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH;
#else
                if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                    src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }
#endif // WITH_SRC_ZPOINTS
                for (int kw = 0; kw < KW; kw++) {
#if WITH_SRC_ZPOINTS
                    const int is_pad_w = kw * (1 + DW) + iw < 0
                            || kw * (1 + DW) + iw >= IW;
                    if (is_pad_w || is_pad_h || is_pad_d) {
#if WITH_SRC_ZPOINTS_PER_IC
                        const int4 z = read_src_zero_points_32c(
                                src_zpoints, (group_ic + ic_chunk) * IC_BLOCK);
#else
                        const int z = read_src_zero_point(src_zpoints);
#endif // WITH_SRC_ZPOINTS_PER_IC
                        BLOCK_READ_WHT(W0, 0);
                        BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
                        BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
                        BLOCK_READ_WHT(W3, 24 * IC_BLOCK);

                        int4 acc = 0;
#if WITH_SRC_ZPOINTS_PER_IC
                        acc.s0 += calc_src_compensation_x32(z, W0);
                        acc.s1 += calc_src_compensation_x32(z, W1);
                        acc.s2 += calc_src_compensation_x32(z, W2);
                        acc.s3 += calc_src_compensation_x32(z, W3);
#else
                        unroll_for(uint i = 0; i < 8; ++i) {
                            acc.s0 = idot4(0x01010101, W0[i], acc.s0);
                            acc.s1 = idot4(0x01010101, W1[i], acc.s1);
                            acc.s2 = idot4(0x01010101, W2[i], acc.s2);
                            acc.s3 = idot4(0x01010101, W3[i], acc.s3);
                        }
                        acc = z * acc;
#endif // WITH_SRC_ZPOINTS_PER_IC

                        C00 += acc.s0;
                        C01 += acc.s1;
                        C02 += acc.s2;
                        C03 += acc.s3;
#if MB > 8
                        C10 += acc.s0;
                        C11 += acc.s1;
                        C12 += acc.s2;
                        C13 += acc.s3;
#ifdef MB_FULL_BLOCK
                        C20 += acc.s0;
                        C21 += acc.s1;
                        C22 += acc.s2;
                        C23 += acc.s3;
                        C30 += acc.s0;
                        C31 += acc.s1;
                        C32 += acc.s2;
                        C33 += acc.s3;
#endif // MB_FULL_BLOCK
#endif // MB > 8
                    } else
#else
                    if (kw * (1 + DW) + iw >= 0 && kw * (1 + DW) + iw < IW)
#endif // WITH_SRC_ZPOINTS
                    {
                        BLOCK_READ_SRC(S0, 0);
#if MB > 8
                        BLOCK_READ_SRC(S1, 8 * IC_BLOCK);
#ifdef MB_FULL_BLOCK
                        BLOCK_READ_SRC(S2, 16 * IC_BLOCK);
                        BLOCK_READ_SRC(S3, 24 * IC_BLOCK);
#endif // MB_FULL_BLOCK
#endif // MB > 8
                        BLOCK_READ_WHT(W0, 0);
                        BLOCK_READ_WHT(W1, 8 * IC_BLOCK);
                        BLOCK_READ_WHT(W2, 16 * IC_BLOCK);
                        BLOCK_READ_WHT(W3, 24 * IC_BLOCK);
                        C00 = mmad8x8(S0, W0, C00);
                        C01 = mmad8x8(S0, W1, C01);
                        C02 = mmad8x8(S0, W2, C02);
                        C03 = mmad8x8(S0, W3, C03);
#if MB > 8
                        C10 = mmad8x8(S1, W0, C10);
                        C11 = mmad8x8(S1, W1, C11);
                        C12 = mmad8x8(S1, W2, C12);
                        C13 = mmad8x8(S1, W3, C13);
#ifdef MB_FULL_BLOCK
                        C20 = mmad8x8(S2, W0, C20);
                        C21 = mmad8x8(S2, W1, C21);
                        C22 = mmad8x8(S2, W2, C22);
                        C23 = mmad8x8(S2, W3, C23);
                        C30 = mmad8x8(S3, W0, C30);
                        C31 = mmad8x8(S3, W1, C31);
                        C32 = mmad8x8(S3, W2, C32);
                        C33 = mmad8x8(S3, W3, C33);
#endif // MB_FULL_BLOCK
#endif // MB > 8
                    }
                    src += IC_BLOCK * MB_BLOCK * (1 + DW);
                    wei += IC_BLOCK * OC_BLOCK;
                } // kw loop
                src += IC_BLOCK * MB_BLOCK * (IW * (1 + DH) - KW * (1 + DW));
            } // kh loop
            src += IC_BLOCK * MB_BLOCK * (IH * (1 + DD) - KH * (1 + DH)) * IW;
        } // kd loop
        src += IC_BLOCK * MB_BLOCK * (ID - KD * (1 + DD)) * IH * IW;
    } // IC_NCHUNK loop

#if WITH_SRC_ZPOINTS
    int4 src_comp = as_int4(intel_sub_group_block_read4(
            (__global uint *)(&src_compensation[(group_oc + oc) * OC_BLOCK])));

    C00 -= src_comp.s0;
    C01 -= src_comp.s1;
    C02 -= src_comp.s2;
    C03 -= src_comp.s3;
#if MB > 8
    C10 -= src_comp.s0;
    C11 -= src_comp.s1;
    C12 -= src_comp.s2;
    C13 -= src_comp.s3;
#ifdef MB_FULL_BLOCK
    C20 -= src_comp.s0;
    C21 -= src_comp.s1;
    C22 -= src_comp.s2;
    C23 -= src_comp.s3;
    C30 -= src_comp.s0;
    C31 -= src_comp.s1;
    C32 -= src_comp.s2;
    C33 -= src_comp.s3;
#endif // MB_FULL_BLOCK
#endif // MB > 8
#endif // WITH_SRC_ZPOINTS && !WITH_SRC_ZPOINTS_PER_IC

    float4 tmp;
    DST_DATA4_T dst_pack[8];
    DST_DATA4_T D0[8];
    DST_DATA4_T D1[8];
    DST_DATA4_T D2[8];
    DST_DATA4_T D3[8];

#if SCALES_PER_OC
    float4 scales;
    BLOCK_READ_SCALES(scales, (group_oc + oc) * OC_BLOCK);
#endif

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, (group_oc + oc) * OC_BLOCK);
    bia *= SCALE;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)SCALE, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= SCALE;
#endif

#if WITH_SUM
    *(DST_DATA16_T *)(D0 + 0) = BLOCK_READ_DST16(dst);
    *(DST_DATA16_T *)(D0 + 4) = BLOCK_READ_DST16(dst + 4 * OC_BLOCK);
#if MB > 8
    *(DST_DATA16_T *)(D1 + 0) = BLOCK_READ_DST16(dst + 8 * OC_BLOCK);
    *(DST_DATA16_T *)(D1 + 4) = BLOCK_READ_DST16(dst + 12 * OC_BLOCK);
#ifdef MB_FULL_BLOCK
    *(DST_DATA16_T *)(D2 + 0) = BLOCK_READ_DST16(dst + 16 * OC_BLOCK);
    *(DST_DATA16_T *)(D2 + 4) = BLOCK_READ_DST16(dst + 20 * OC_BLOCK);
    *(DST_DATA16_T *)(D3 + 0) = BLOCK_READ_DST16(dst + 24 * OC_BLOCK);
    *(DST_DATA16_T *)(D3 + 4) = BLOCK_READ_DST16(dst + 28 * OC_BLOCK);
#endif // MB_FULL_BLOCK
#endif // MB > 8
#endif // with_sum

#if WITH_DST_ZPOINTS
    int4 dst_zp = read_dst_zero_points_32c(
            dst_compensation, (group_oc + oc) * OC_BLOCK);
#define ADD_DST_COMPENSATION() tmp += convert_float4(dst_zp);
#else
#define ADD_DST_COMPENSATION()
#endif // WITH_DST_ZPOINTS

#if WITH_SRC_ZPOINTS
#define ZERO_PAD_DST() tmp = zero_pad_dst_32c(tmp, (group_oc + oc) * OC_BLOCK);
#else
#define ZERO_PAD_DST()
#endif // WITH_SRC_ZPOINTS

#define PACK(C0, C1, C2, C3, idx) \
    do { \
        tmp[0] = C0[idx]; \
        tmp[1] = C1[idx]; \
        tmp[2] = C2[idx]; \
        tmp[3] = C3[idx]; \
    } while (0)

#define CONVERT_PACK(idx) \
    do { \
        dst_pack[idx] = CONVERT_DST_DATA4_T(tmp); \
    } while (0)

#define STORE_DST(C0, C1, C2, C3, D, mb_stride) \
    do { \
        for (int n_i = 0; n_i < 8; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            for (int didx = 0; didx < 4; ++didx) { \
                float tmp_i = tmp[didx]; \
                float d_i = convert_float(AS_SUM_DATA_T(D[n_i][didx])); \
                const int po_mb \
                        = (group_mb * MB_BLOCK + mb * MB_BLOCK / mb_blocks \
                                  + mb_stride + n_i) \
                        % MB; \
                const int po_oc = (group_oc * OC_BLOCK + didx * SUB_GROUP_SIZE \
                                          + ocl_local_id) \
                        % (OC * G); \
                APPLY_POST_OPS(tmp_i, float, d_i, float, po_mb, 1, po_oc, 1, \
                        0, 1, 0, 1, 0, 1, 0, 1); \
                tmp[didx] = tmp_i; \
            } \
            ADD_DST_COMPENSATION(); \
            ZERO_PAD_DST(); \
            CONVERT_PACK(n_i); \
        } \
        BLOCK_WRITE_DST16( \
                &dst[mb_stride * OC_BLOCK], *(DST_DATA16_T *)dst_pack); \
        BLOCK_WRITE_DST16(&dst[mb_stride * OC_BLOCK + 16 * 8], \
                *(DST_DATA16_T *)(dst_pack + 4)); \
    } while (0)

    STORE_DST(C00, C01, C02, C03, D0, 0);
#if MB > 8
    STORE_DST(C10, C11, C12, C13, D1, 8);
#ifdef MB_FULL_BLOCK
    STORE_DST(C20, C21, C22, C23, D2, 16);
    STORE_DST(C30, C31, C32, C33, D3, 24);
#endif // MB_FULL_BLOCK
#endif // MB > 8
}
