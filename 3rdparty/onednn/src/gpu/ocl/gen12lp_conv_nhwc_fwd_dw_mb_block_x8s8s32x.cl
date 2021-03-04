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

#include "gpu/ocl/ocl_math_utils.h"
#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#define KDHW_SIZE KD *KH *KW

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

void block_read_src(uchar *s, __global uchar *src, const int g) {
    const int sglid = get_sub_group_local_id();
    for (int i = 0; i < SUB_GROUP_SIZE; i++) {
        if ((i % 2) * SUB_GROUP_SIZE + sglid
                < (g + IC_BLOCK > G ? G % IC_BLOCK : IC_BLOCK))
            s[i] = src[(i / 2) * (ID * IH * IW * G) + (i % 2) * SUB_GROUP_SIZE
                    + sglid];
        else
            s[i] = 0;
    }
}
void block_write_dst(
        const DST_DATA_T *d, __global DST_DATA_T *dst, const int g) {
    const int sglid = get_sub_group_local_id();
    for (int i = 0; i < SUB_GROUP_SIZE; i++) {
        if ((i % 2) * SUB_GROUP_SIZE + sglid
                < (g + OC_BLOCK > G ? G % OC_BLOCK : OC_BLOCK))
            dst[(i / 2) * (OD * OH * OW * G) + (i % 2) * SUB_GROUP_SIZE + sglid]
                    = d[i];
    }
}
void block_read_dst_as_sum(
        SUM_DATA_T *d, __global DST_DATA_T *dst, const int g) {
    const int sglid = get_sub_group_local_id();
    for (int i = 0; i < SUB_GROUP_SIZE; i++) {
        if ((i % 2) * SUB_GROUP_SIZE + sglid
                < (g + IC_BLOCK > G ? G % IC_BLOCK : IC_BLOCK))
            d[i] = AS_SUM_DATA_T(dst[(i / 2) * (OD * OH * OW * G)
                    + (i % 2) * SUB_GROUP_SIZE + sglid]);
        else
            d[i] = 0;
    }
}

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_nhwc_fwd_dw_mb_block_x8s8s32x(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global DST_DATA_T *dst POST_OP_ARGS, float scale,
        const __global float *scales_per_oc,
        const __global int *src_compensation, const __global int *src_zpoints,
        const __global int *dst_compensation) {

    const int osp = get_global_id(1);
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
    const int sglid = get_sub_group_local_id();

    src += (mb + mb_half) * ID * IH * IW * G + (id * IH * IW + ih * IW + iw) * G
            + g;
    dst += (mb + mb_half) * OD * OH * OW * G + (od * OH * OW + oh * OW + ow) * G
            + g;

    wei += g * KDHW_SIZE;

    int8 S00 = 0;
    int8 S01 = 0;
    uchar16 A00, A10, A20, A30;

    __attribute__((opencl_unroll_hint)) for (int sp = 0;
                                             sp < KDHW_SIZE - KDHW_SIZE % 4;
                                             sp += 4) {

        const int4 s = {sp, sp + 1, sp + 2, sp + 3};
        const int4 kd = s / (KH * KW);
        const int4 kh = (s % (KH * KW)) / KW;
        const int4 kw = (s % (KH * KW)) % KW;
        const int4 src_index
                = (kd * (1 + DD) * IH * IW + kh * (1 + DH) * IW + kw * (1 + DW))
                * G;
        const int4 index = id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID
                || ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH
                || iw + kw * (1 + DW) < 0 || iw + kw * (1 + DW) >= IW;
        if (index.s0) {
            A00 = 0;
        } else {
            block_read_src((uchar *)&A00, src + src_index.s0, g);
        }
        if (index.s1) {
            A10 = 0;
        } else {
            block_read_src((uchar *)&A10, src + src_index.s1, g);
        }
        if (index.s2) {
            A20 = 0;
        } else {
            block_read_src((uchar *)&A20, src + src_index.s2, g);
        }
        if (index.s3) {
            A30 = 0;
        } else {
            block_read_src((uchar *)&A30, src + src_index.s3, g);
        }

        char8 W = as_char8(
                intel_sub_group_block_read_uc8((const __global uchar *)(wei)));
        char4 W1 = W.s0246;
        char4 W2 = W.s1357;
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

    __attribute__((opencl_unroll_hint)) for (int sp = KDHW_SIZE - KDHW_SIZE % 4;
                                             sp < KDHW_SIZE; sp++) {
        const int kd = sp / (KH * KW);
        const int kh = (sp % (KH * KW)) / KW;
        const int kw = (sp % (KH * KW)) % KW;
        const int src_index
                = (kd * (1 + DD) * IH * IW + kh * (1 + DH) * IW + kw * (1 + DW))
                * G;

        const int index = id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID
                || ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH
                || iw + kw * (1 + DW) < 0 || iw + kw * (1 + DW) >= IW;
        if (index) {
            A00 = 0;
        } else {
            block_read_src((uchar *)&A00, src + src_index, g);
        }

        const char2 W = as_char2(
                intel_sub_group_block_read_uc2((const __global uchar *)(wei)));
        const char W1 = W.s0;
        const char W2 = W.s1;
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
    float8 tmp00 = convert_float8(S00);
    float8 tmp01 = convert_float8(S01);

#if SCALES_PER_OC
    float2 scales = as_float2(intel_sub_group_block_read2(
            (const __global uint *)&scales_per_oc[g]));
#endif

#if WITH_BIAS
    float2 B = as_float2(
            intel_sub_group_block_read2((const __global uint *)&bias[g]));
    B *= SCALE;
    tmp00 = fma(tmp00, (float8)SCALE_VEC8, B.s01010101);
    tmp01 = fma(tmp01, (float8)SCALE_VEC8, B.s01010101);
#else
    tmp00 *= SCALE_VEC8;
    tmp01 *= SCALE_VEC8;
#endif

    SUM_DATA16_T D00;
#if WITH_SUM
    block_read_dst_as_sum(&D00, dst, g);
#endif
    float16 tmp_x16 = (float16)(tmp00, tmp01);
    APPLY_POST_OPS(tmp_x16, float, D00, SUM_DATA_T, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1);
    DST_DATA16_T R0 = CONVERT_DST_DATA16_T(tmp_x16);
    block_write_dst((DST_DATA16_T *)&R0, dst, g);
}
