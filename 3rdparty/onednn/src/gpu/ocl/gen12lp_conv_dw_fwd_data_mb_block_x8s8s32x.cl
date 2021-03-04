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

#define KDHW_SIZE (KD * KH * KW)

#if SCALES_PER_OC
#define SCALE scales
#define SCALE_VEC8 scales.s01010101
#elif SCALES_COMMON
#define SCALE scale
#define SCALE_VEC8 scale
#else
#define SCALE 1
#define SCALE_VEC8 1
#endif

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_dw_fwd_mb_block_x8s8s32x(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global DST_DATA_T *dst POST_OP_ARGS, float scale,
        const __global float *scales_per_oc,
        const __global int *src_compensation, const __global int *src_zpoints,
        const __global int *dst_compensation) {

    const int osp = get_global_id(1);
    const int ocl_local_id = get_local_id(0);
    const int od = osp / (OW * OH);
    const int ohw = osp % (OW * OH);
    const int ow = (ohw % OW);
    const int oh = ohw / OW;
    const int g = get_group_id(0) * OC_BLOCK;
    const int mb = (get_global_id(2) / 4) * MB_BLOCK;
    const int mb_half = (get_global_id(2) % 4) * MB_BLOCK / 4;
    const int id = od * SD - PD;
    const int ih = oh * SH - PH;
    const int iw = ow * SW - PW;

    dst += mb * G * OD * OH * OW + g * OD * OH * OW * MB_BLOCK
            + (od * OH * OW + oh * OW + ow) * MB_BLOCK * OC_BLOCK
            + mb_half * OC_BLOCK;
    src += mb * G * ID * IH * IW + g * ID * IH * IW * MB_BLOCK
            + (id * IH * IW + ih * IW + iw) * MB_BLOCK * IC_BLOCK
            + mb_half * IC_BLOCK;
    wei += g * KDHW_SIZE;

    int8 S00 = 0;
    int8 S01 = 0;
    uchar16 A00, A10, A20, A30;

#if WITH_SRC_ZPOINTS
#if WITH_SRC_ZPOINTS_PER_IC
    const int2 z = read_src_zero_points_32g(src_zpoints, g);
#else
    const int2 z = read_src_zero_point(src_zpoints);
#endif // WITH_SRC_ZPOINTS_PER_IC
#endif // WITH_SRC_ZPOINTS

    unroll_for(int sp = 0; sp < KDHW_SIZE - KDHW_SIZE % 4; sp += 4) {
        const int4 s = {sp, sp + 1, sp + 2, sp + 3};
        const int4 kd = s / (KH * KW);
        const int4 kh = (s % (KH * KW)) / KW;
        const int4 kw = (s % (KH * KW)) % KW;
        const int4 src_index
                = (kd * (1 + DD) * IH * IW + kh * (1 + DH) * IW + kw * (1 + DW))
                * MB_BLOCK * IC_BLOCK;
        const int4 index = id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID
                || ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH
                || iw + kw * (1 + DW) < 0 || iw + kw * (1 + DW) >= IW;
        if (index.s0) {
            A00 = 0;
        } else {
            A00 = (intel_sub_group_block_read_uc16(
                    (const __global uchar *)(&src[src_index.s0])));
        }
        if (index.s1) {
            A10 = 0;
        } else {
            A10 = (intel_sub_group_block_read_uc16(
                    (const __global uchar *)(&src[src_index.s1])));
        }
        if (index.s2) {
            A20 = 0;
        } else {
            A20 = (intel_sub_group_block_read_uc16(
                    (const __global uchar *)(&src[src_index.s2])));
        }
        if (index.s3) {
            A30 = 0;
        } else {
            A30 = (intel_sub_group_block_read_uc16(
                    (const __global uchar *)(&src[src_index.s3])));
        }
        char8 W = as_char8(
                intel_sub_group_block_read_uc8((const __global uchar *)(wei)));
        char4 W1 = W.s0246;
        char4 W2 = W.s1357;
#if WITH_SRC_ZPOINTS
        if (index.s0) {
            int2 src_comp;
            src_comp.s0 = z.s0 * W.s0;
            src_comp.s1 = z.s1 * W.s1;
            S00 += src_comp.s01010101;
            S01 += src_comp.s01010101;
        }
        if (index.s1) {
            int2 src_comp;
            src_comp.s0 = z.s0 * W.s2;
            src_comp.s1 = z.s1 * W.s3;
            S00 += src_comp.s01010101;
            S01 += src_comp.s01010101;
        }
        if (index.s2) {
            int2 src_comp;
            src_comp.s0 = z.s0 * W.s4;
            src_comp.s1 = z.s1 * W.s5;
            S00 += src_comp.s01010101;
            S01 += src_comp.s01010101;
        }
        if (index.s3) {
            int2 src_comp;
            src_comp.s0 = z.s0 * W.s6;
            src_comp.s1 = z.s1 * W.s7;
            S00 += src_comp.s01010101;
            S01 += src_comp.s01010101;
        }
#endif // WITH_SRC_ZPOINTS
        S00.s0 = idot4(
                (SRC_DATA4_T)(A00.s0, A10.s0, A20.s0, A30.s0), W1, S00.s0);
        S00.s1 = idot4(
                (SRC_DATA4_T)(A00.s1, A10.s1, A20.s1, A30.s1), W2, S00.s1);
        S00.s2 = idot4(
                (SRC_DATA4_T)(A00.s2, A10.s2, A20.s2, A30.s2), W1, S00.s2);
        S00.s3 = idot4(
                (SRC_DATA4_T)(A00.s3, A10.s3, A20.s3, A30.s3), W2, S00.s3);
        S00.s4 = idot4(
                (SRC_DATA4_T)(A00.s4, A10.s4, A20.s4, A30.s4), W1, S00.s4);
        S00.s5 = idot4(
                (SRC_DATA4_T)(A00.s5, A10.s5, A20.s5, A30.s5), W2, S00.s5);
        S00.s6 = idot4(
                (SRC_DATA4_T)(A00.s6, A10.s6, A20.s6, A30.s6), W1, S00.s6);
        S00.s7 = idot4(
                (SRC_DATA4_T)(A00.s7, A10.s7, A20.s7, A30.s7), W2, S00.s7);
        S01.s0 = idot4(
                (SRC_DATA4_T)(A00.s8, A10.s8, A20.s8, A30.s8), W1, S01.s0);
        S01.s1 = idot4(
                (SRC_DATA4_T)(A00.s9, A10.s9, A20.s9, A30.s9), W2, S01.s1);
        S01.s2 = idot4(
                (SRC_DATA4_T)(A00.sa, A10.sa, A20.sa, A30.sa), W1, S01.s2);
        S01.s3 = idot4(
                (SRC_DATA4_T)(A00.sb, A10.sb, A20.sb, A30.sb), W2, S01.s3);
        S01.s4 = idot4(
                (SRC_DATA4_T)(A00.sc, A10.sc, A20.sc, A30.sc), W1, S01.s4);
        S01.s5 = idot4(
                (SRC_DATA4_T)(A00.sd, A10.sd, A20.sd, A30.sd), W2, S01.s5);
        S01.s6 = idot4(
                (SRC_DATA4_T)(A00.se, A10.se, A20.se, A30.se), W1, S01.s6);
        S01.s7 = idot4(
                (SRC_DATA4_T)(A00.sf, A10.sf, A20.sf, A30.sf), W2, S01.s7);
        wei += 4 * OC_BLOCK;
    }

    unroll_for(int sp = KDHW_SIZE - KDHW_SIZE % 4; sp < KDHW_SIZE; sp++) {
        const int kd = sp / (KH * KW);
        const int kh = (sp % (KH * KW)) / KW;
        const int kw = (sp % (KH * KW)) % KW;
        const int src_index
                = (kd * (1 + DD) * IH * IW + kh * (1 + DH) * IW + kw * (1 + DW))
                * MB_BLOCK * IC_BLOCK;
        const int index = id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID
                || ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH
                || iw + kw * (1 + DW) < 0 || iw + kw * (1 + DW) >= IW;
        if (index) {
            A00 = 0;
        } else {
            A00 = (intel_sub_group_block_read_uc16(
                    (const __global uchar *)(&src[src_index])));
        }
        const char2 W = as_char2(
                intel_sub_group_block_read_uc2((const __global uchar *)(wei)));
        const char W1 = W.s0;
        const char W2 = W.s1;
#if WITH_SRC_ZPOINTS
        if (index) {
            int2 src_comp;
            src_comp.s0 = z.s0 * W.s0;
            src_comp.s1 = z.s1 * W.s1;

            S00 += src_comp.s01010101;
            S01 += src_comp.s01010101;
        }
#endif // WITH_SRC_ZPOINTS
        S00.s0 += ((SRC_DATA_T)A00.s0) * W1;
        S00.s1 += ((SRC_DATA_T)A00.s1) * W2;
        S00.s2 += ((SRC_DATA_T)A00.s2) * W1;
        S00.s3 += ((SRC_DATA_T)A00.s3) * W2;
        S00.s4 += ((SRC_DATA_T)A00.s4) * W1;
        S00.s5 += ((SRC_DATA_T)A00.s5) * W2;
        S00.s6 += ((SRC_DATA_T)A00.s6) * W1;
        S00.s7 += ((SRC_DATA_T)A00.s7) * W2;
        S01.s0 += ((SRC_DATA_T)A00.s8) * W1;
        S01.s1 += ((SRC_DATA_T)A00.s9) * W2;
        S01.s2 += ((SRC_DATA_T)A00.sa) * W1;
        S01.s3 += ((SRC_DATA_T)A00.sb) * W2;
        S01.s4 += ((SRC_DATA_T)A00.sc) * W1;
        S01.s5 += ((SRC_DATA_T)A00.sd) * W2;
        S01.s6 += ((SRC_DATA_T)A00.se) * W1;
        S01.s7 += ((SRC_DATA_T)A00.sf) * W2;

        wei += OC_BLOCK;
    }
#if WITH_SRC_ZPOINTS
    const int2 src_comp = as_int2(intel_sub_group_block_read2(
            (__global uint *)(&src_compensation[g])));
    S00 -= src_comp.s01010101;
    S01 -= src_comp.s01010101;
#endif // WITH_SRC_ZPOINTS

    float8 tmp00 = convert_float8(S00);
    float8 tmp01 = convert_float8(S01);

#if SCALES_PER_OC
    float2 scales = as_float2(intel_sub_group_block_read2(
            (const __global uint *)&scales_per_oc[g]));
#endif

#if WITH_BIAS
#if OC % OC_BLOCK == 0
    float2 B = as_float2(
            intel_sub_group_block_read2((const __global uint *)&bias[g]));
#else
    float2 B = 0;
    int i = 0;
    for (i = 0; i < (G - g) / SUB_GROUP_SIZE && i < 2; ++i) {
        B[i] = as_float(intel_sub_group_block_read(
                (const __global uint *)&bias[g + (i * SUB_GROUP_SIZE)]));
    }
    if (i < 2 && get_sub_group_local_id() < (G - g) % SUB_GROUP_SIZE) {
        B[i] = bias[g + (i * SUB_GROUP_SIZE) + get_sub_group_local_id()];
    }
#endif
    B *= SCALE;
    float8 B0123 = B.s01010101;
    float8 B4567 = B.s01010101;
    if (MB % MB_BLOCK != 0) {
        int8 mb_vec = mb + mb_half;
        int8 m0123 = (mb_vec + (int8)(0, 0, 1, 1, 2, 2, 3, 3)) < MB;
        int8 m4567 = (mb_vec + (int8)(4, 4, 5, 5, 6, 6, 7, 7)) < MB;
        B0123 = select((float8)0, B0123, m0123);
        B4567 = select((float8)0, B4567, m4567);
    }
    tmp00 = fma(tmp00, (float8)SCALE_VEC8, B0123);
    tmp01 = fma(tmp01, (float8)SCALE_VEC8, B4567);
#else
    tmp00 *= SCALE_VEC8;
    tmp01 *= SCALE_VEC8;
#endif

    SUM_DATA16_T D00;
#if WITH_SUM
    D00 = AS_SUM_DATA16_T(BLOCK_READ_DST16(dst));
#endif // WITH_SUM

    float16 tmp_x16 = (float16)(tmp00, tmp01);

    for (int didx = 0; didx < 16; ++didx) {
        float tmp_i = tmp_x16[didx];
        SUM_DATA_T d_i = D00[didx];
        const int po_mb = mb /* * MB_BLOCK */ + didx / 2 + mb_half;
        const int po_oc = g * OC + (didx % 2) * SUB_GROUP_SIZE + ocl_local_id;
        APPLY_POST_OPS(tmp_i, float, d_i, SUM_DATA_T, po_mb, 1, po_oc, 1, 0, 1,
                0, 1, 0, 1, 0, 1);
        tmp_x16[didx] = tmp_i;
    }

#if WITH_DST_ZPOINTS
    float2 dst_zp
            = convert_float2(read_dst_zero_points_32g(dst_compensation, g));
    float8 tmp_zp = dst_zp.s01010101;
    tmp_x16 += (float16)(tmp_zp, tmp_zp);
#endif // WITH_DST_ZPOINTS

    DST_DATA16_T R0 = CONVERT_DST_DATA16_T(tmp_x16);

    BLOCK_WRITE_DST16(dst, R0);
}
