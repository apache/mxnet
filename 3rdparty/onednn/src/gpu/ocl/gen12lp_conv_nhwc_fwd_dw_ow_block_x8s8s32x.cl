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

#define PWX (PW > 1 ? PW : 1)
#if SCALES_PER_OC
#define SCALE scales.s0101010101010101
#elif SCALES_COMMON
#define SCALE scale
#else
#define SCALE 1
#endif

void block_read_dst(
        int n, DST_DATA_T *d, const __global DST_DATA_T *dst, const int g);
void block_write_dst(
        int n, const DST_DATA_T *d, __global DST_DATA_T *dst, const int g);
void block_read_src(int n, ushort *s, const __global ushort *src, const int g);

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_nhwc_fwd_dw_ow_block_x8s8s32x(const __global uchar *src,
        const __global char *wei, const __global float *bias,
        __global DST_DATA_T *dst POST_OP_ARGS, float scale,
        const __global float *scales_per_oc,
        const __global int *src_compensation, const __global int *src_zpoints,
        const __global int *dst_compensation) {
    const int osp = get_global_id(1);
    const int od = osp / (OWB * OH);
    const int ohw = osp % (OWB * OH);
    const int ow = (ohw % OWB) * OW_BLOCK;
    const int oh = ohw / OWB;
    const int g = get_group_id(0) * OC_BLOCK;
    const int mb = get_global_id(2) * MB_BLOCK;
    const int id = od * SD - PD;
    const int ih = oh * SH - PH;
    const int iw = ow * SW - PW;
    const int sglid = get_sub_group_local_id();

    int dst_off, src_off;
    dst_off = mb * OD * OH * OW * G + (od * OH * OW + oh * OW + ow) * G + g;
    src_off = mb * ID * IH * IW * G + (id * IH * IW + ih * IW + iw) * G + g;

    dst += dst_off;
    src += src_off;

    wei += g * KD * KH * KW;

    int16 S0 = 0;
    int16 S1 = 0;

#if SCALES_PER_OC
    float2 scales;
    scales.s0 = scales_per_oc[g + 2 * sglid];
    scales.s1 = scales_per_oc[g + 2 * sglid + 1];
#endif

#if WITH_BIAS
    float2 B;
    B.s0 = bias[g + 2 * sglid];
    B.s1 = bias[g + 2 * sglid + 1];
    S0 = convert_int16(B.s0101010101010101);
    S1 = convert_int16(B.s0101010101010101);
#endif

    for (int kd = 0; kd < KD; kd++) {
        if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
            src += G * IH * IW * (1 + DD);
            wei += OC_BLOCK * KH * KW;
            continue;
        }
        __attribute__((opencl_unroll_hint)) for (int kh = 0; kh < KH; kh++) {
            if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                src += G * MB_BLOCK * IW * (1 + DH);
                wei += OC_BLOCK * KW;
                continue;
            }

#if SW == 2
            ushort16 AAA = 0;
#else
            ushort AAA;
#endif
            ushort16 AA = 0;

#if OW % (SW * OW_BLOCK) + KW - 1 \
        > SW - 1 + PW //possible out of bounds read  if below check excluded
            /* get main block */
            if (iw + SW * (OW_BLOCK) + KW - 1 > IW) {
                if (iw >= 0) {
                    block_read_src(1, ((ushort *)&AA) + 0,
                            (const __global ushort *)(&src[0 * G]), g);
#if PW >= 2
                    block_read_src(1, ((ushort *)&AA) + 1,
                            (const __global ushort *)(&src[1 * G]), g);
#endif
#if PW >= 3
                    block_read_src(1, ((ushort *)&AA) + 2,
                            (const __global ushort *)(&src[2 * G]), g);
#endif
                }
#if IW_TAIL > 16
#if PW == 2
                block_read_src(8, ((ushort *)&AA) + 2,
                        (const __global ushort *)(&src[(PWX)*G]), g);
                block_read_src(4, ((ushort *)&AA) + 10,
                        (const __global ushort *)(&src[(PWX + 8) * G]), g);
                block_read_src(2, ((ushort *)&AA) + 14,
                        (const __global ushort *)(&src[(PWX + 12) * G]), g);
#elif PW == 3
                block_read_src(8, ((ushort *)&AA) + 3,
                        (const __global ushort *)(&src[(PWX)*G]), g);
                block_read_src(4, ((ushort *)&AA) + 11,
                        (const __global ushort *)(&src[(PWX + 8) * G]), g);
                block_read_src(1, ((ushort *)&AA) + 15,
                        (const __global ushort *)(&src[(PWX + 12) * G]), g);
#else
                block_read_src(8, ((ushort *)&AA) + 1,
                        (const __global ushort *)(&src[(PWX)*G]), g);
                block_read_src(4, ((ushort *)&AA) + 9,
                        (const __global ushort *)(&src[(PWX + 8) * G]), g);
                block_read_src(2, ((ushort *)&AA) + 13,
                        (const __global ushort *)(&src[(PWX + 12) * G]), g);
                block_read_src(1, ((ushort *)&AA) + 15,
                        (const __global ushort *)(&src[(PWX + 14) * G]), g);
#endif

#if ((IW_TAIL - 16) & 0b1000) == 0b1000
                block_read_src(8, ((ushort *)&AAA),
                        (const __global ushort *)(&src[(16) * G]), g);
#endif
#if ((IW_TAIL - 16) & 0b100) == 0b100
                block_read_src(4, ((ushort *)&AAA) + ((IW_TAIL - 16) & 0b1000),
                        (const __global ushort
                                        *)(&src[(16 + ((IW_TAIL - 16) & 0b1000))
                                * G]),
                        g);
#endif
#if ((IW_TAIL - 16) & 0b10) == 0b10
                block_read_src(2, ((ushort *)&AAA) + ((IW_TAIL - 16) & 0b1100),
                        (const __global ushort
                                        *)(&src[(16 + ((IW_TAIL - 16) & 0b1100))
                                * G]),
                        g);
#endif
#if ((IW_TAIL - 16) & 0b1) == 0b1
                block_read_src(1, ((ushort *)&AAA) + ((IW_TAIL - 16) & 0b1110),
                        (const __global ushort
                                        *)(&src[(16 + ((IW_TAIL - 16) & 0b1110))
                                * G]),
                        g);
#endif

#else

#if ((IW_TAIL - PWX) & 0b1000) == 0b1000
                block_read_src(8, ((ushort *)&AA) + PWX,
                        (const __global ushort *)(&src[(PWX)*G]), g);
#endif
#if ((IW_TAIL - PWX) & 0b100) == 0b100
                block_read_src(4,
                        ((ushort *)&AA) + PWX + ((IW_TAIL - PWX) & 0b1000),
                        (const __global ushort
                                        *)(&src[(PWX
                                                        + ((IW_TAIL - PWX)
                                                                & 0b1000))
                                * G]),
                        g);
#endif
#if ((IW_TAIL - PWX) & 0b10) == 0b10
                block_read_src(2,
                        ((ushort *)&AA) + PWX + ((IW_TAIL - PWX) & 0b1100),
                        (const __global ushort
                                        *)(&src[(PWX
                                                        + ((IW_TAIL - PWX)
                                                                & 0b1100))
                                * G]),
                        g);
#endif
#if ((IW_TAIL - PWX) & 0b1) == 0b1
                block_read_src(1,
                        ((ushort *)&AA) + PWX + ((IW_TAIL - PWX) & 0b1110),
                        (const __global ushort
                                        *)(&src[(PWX
                                                        + ((IW_TAIL - PWX)
                                                                & 0b1110))
                                * G]),
                        g);
#endif

#endif
            } else {
#endif
#if SW == 1
#define READ_BLOCK (OW_BLOCK + KW - 1 - PWX)

                if (iw >= 0) {
                    block_read_src(1, ((ushort *)&AA) + 0,
                            (const __global ushort *)(&src[0 * G]), g);
#if PW >= 2
                    block_read_src(1, ((ushort *)&AA) + 1,
                            (const __global ushort *)(&src[1 * G]), g);
#endif
#if PW >= 3
                    block_read_src(1, ((ushort *)&AA) + 2,
                            (const __global ushort *)(&src[2 * G]), g);
#endif
                }

#if (READ_BLOCK & 0b1000) == 0b1000
                block_read_src(8, ((ushort *)&AA) + PWX,
                        (const __global ushort *)(&src[(PWX)*G]), g);
#endif
#if (READ_BLOCK & 0b100) == 0b100
                block_read_src(4, ((ushort *)&AA) + PWX + (READ_BLOCK & 0b1000),
                        (const __global ushort
                                        *)(&src[(PWX + (READ_BLOCK & 0b1000))
                                * G]),
                        g);
#endif
#if (READ_BLOCK & 0b10) == 0b10
                block_read_src(2, ((ushort *)&AA) + PWX + (READ_BLOCK & 0b1100),
                        (const __global ushort
                                        *)(&src[(PWX + (READ_BLOCK & 0b1100))
                                * G]),
                        g);
#endif
#if (READ_BLOCK & 0b1) == 0b1
                block_read_src(1, ((ushort *)&AA) + PWX + (READ_BLOCK & 0b1110),
                        (const __global ushort
                                        *)(&src[(PWX + (READ_BLOCK & 0b1110))
                                * G]),
                        g);
#endif

#elif SW == 2

#if OW_BLOCK + KW - 1 >= 8
#define READ_BLOCK (2 * (OW_BLOCK) + KW - 1)
            if (iw >= 0) {
                block_read_src(16, (ushort *)&AA,
                        (const __global ushort *)(&src[0 * G]), g);
            } else {
#if PW == 0
                block_read_src(16, (ushort *)&AA,
                        (const __global ushort *)(&src[0 * G]), g);

#elif PW == 2
                block_read_src(8, ((ushort *)&AA) + 2,
                        (const __global ushort *)(&src[(PW)*G]), g);
                block_read_src(4, ((ushort *)&AA) + 10,
                        (const __global ushort *)(&src[(PW + 8) * G]), g);
                block_read_src(2, ((ushort *)&AA) + 14,
                        (const __global ushort *)(&src[(PW + 12) * G]), g);
#elif PW == 3
                block_read_src(8, ((ushort *)&AA) + 3,
                        (const __global ushort *)(&src[(PW)*G]), g);
                block_read_src(4, ((ushort *)&AA) + 11,
                        (const __global ushort *)(&src[(PW + 8) * G]), g);
                block_read_src(1, ((ushort *)&AA) + 15,
                        (const __global ushort *)(&src[(PW + 12) * G]), g);
#else
                block_read_src(8, ((ushort *)&AA) + 1,
                        (const __global ushort *)(&src[(PW)*G]), g);
                block_read_src(4, ((ushort *)&AA) + 9,
                        (const __global ushort *)(&src[(PW + 8) * G]), g);
                block_read_src(2, ((ushort *)&AA) + 13,
                        (const __global ushort *)(&src[(PW + 12) * G]), g);
                block_read_src(1, ((ushort *)&AA) + 15,
                        (const __global ushort *)(&src[(PW + 14) * G]), g);
#endif
            }

#if ((READ_BLOCK - 16) & 0b1000) == 0b1000
            block_read_src(8, ((ushort *)&AAA),
                    (const __global ushort *)(&src[(16) * G]), g);
#endif
#if ((READ_BLOCK - 16) & 0b100) == 0b100
            block_read_src(4, ((ushort *)&AAA) + ((READ_BLOCK - 16) & 0b1000),
                    (const __global ushort
                                    *)(&src[(16 + ((READ_BLOCK - 16) & 0b1000))
                            * G]),
                    g);
#endif
#if ((READ_BLOCK - 16) & 0b10) == 0b10
            block_read_src(2, ((ushort *)&AAA) + ((READ_BLOCK - 16) & 0b1100),
                    (const __global ushort
                                    *)(&src[(16 + ((READ_BLOCK - 16) & 0b1100))
                            * G]),
                    g);
#endif
#if ((READ_BLOCK - 16) & 0b1) == 0b1
            block_read_src(1, ((ushort *)&AAA) + ((READ_BLOCK - 16) & 0b1110),
                    (const __global ushort
                                    *)(&src[(16 + ((READ_BLOCK - 16) & 0b1110))
                            * G]),
                    g);
#endif

#else // OW_BLOCK >= 8

#define READ_BLOCK (2 * (OW_BLOCK) + KW - 1 - PWX)
            if (iw >= 0) {
                block_read_src(1, ((ushort *)&AA) + 0,
                        (const __global ushort *)(&src[0 * G]), g);
#if PW >= 2
                block_read_src(1, ((ushort *)&AA) + 1,
                        (const __global ushort *)(&src[1 * G]), g);
#endif
#if PW >= 3
                block_read_src(1, ((ushort *)&AA) + 2,
                        (const __global ushort *)(&src[2 * G]), g);
#endif
            }
#if (READ_BLOCK & 0b1000) == 0b1000
            block_read_src(8, ((ushort *)&AA) + PWX,
                    (const __global ushort *)(&src[(PWX)*G]), g);
#endif
#if (READ_BLOCK & 0b100) == 0b100
            block_read_src(4, ((ushort *)&AA) + PWX + (READ_BLOCK & 0b1000),
                    (const __global ushort
                                    *)(&src[(PWX + (READ_BLOCK & 0b1000)) * G]),
                    g);
#endif
#if (READ_BLOCK & 0b10) == 0b10
            block_read_src(2, ((ushort *)&AA) + PWX + (READ_BLOCK & 0b1100),
                    (const __global ushort
                                    *)(&src[(PWX + (READ_BLOCK & 0b1100)) * G]),
                    g);
#endif
#if (READ_BLOCK & 0b1) == 0b1
            block_read_src(2, ((ushort *)&AA) + PWX + (READ_BLOCK & 0b1110),
                    (const __global ushort
                                    *)(&src[(PWX + (READ_BLOCK & 0b1110)) * G]),
                    g);
#endif
#endif

#endif
#if OW % (SW * OW_BLOCK) + KW - 1 > SW - 1 + PW
            }
#endif
#if OW > OWX
            if (iw + READ_BLOCK > IW) {
                if (iw < IW) {
                    for (int i = IW - iw; i < READ_BLOCK; i++) {
                        if (i < 16) {
                            AA[i] = 0;
                        }
#if SW == 2
                        else {
                            AAA[i - 16] = 0;
                        }
#endif
                    }
                } else {
                    AA = 0;
#if SW == 2
                    AAA = 0;
#endif
                }
            }
#endif

            ushort4 WW = 0;
#if KW > 4
#error "Wrong KW"
#endif
#if KW == 4
            WW = intel_sub_group_block_read_us4((const __global ushort *)wei);
#endif
#if KW == 3
            WW.s01 = intel_sub_group_block_read_us2(
                    (const __global ushort *)wei);
            WW.s2 = intel_sub_group_block_read_us(
                    (const __global ushort *)wei + OC_BLOCK);
#endif
#if KW == 2
            WW.s01 = intel_sub_group_block_read_us2(
                    (const __global ushort *)wei);
#endif
#if KW == 1
            WW.s0 = intel_sub_group_block_read_us((const __global ushort *)wei);
#endif
            SRC_DATA16_T A0 = 0, A1 = 0;
            A0.s01234567 = AS_SRC_DATA16_T(AA.s01234567).s02468ace;
            A0.s89abcdef = AS_SRC_DATA16_T(AA.s89abcdef).s02468ace;
            A1.s01234567 = AS_SRC_DATA16_T(AA.s01234567).s13579bdf;
            A1.s89abcdef = AS_SRC_DATA16_T(AA.s89abcdef).s13579bdf;
#if SW == 2
            SRC_DATA16_T right0, right1;
            right0.s01234567 = AS_SRC_DATA16_T(AAA.s01234567).s02468ace;
            right0.s89abcdef = AS_SRC_DATA16_T(AAA.s89abcdef).s02468ace;
            right1.s01234567 = AS_SRC_DATA16_T(AAA.s01234567).s13579bdf;
            right1.s89abcdef = AS_SRC_DATA16_T(AAA.s89abcdef).s13579bdf;
#else
#if OW_BLOCK >= 14
            SRC_DATA_T right0, right1;
            right0 = AS_SRC_DATA2_T(AAA).s0;
            right1 = AS_SRC_DATA2_T(AAA).s1;
#endif
#endif
            char8 W = as_char8(WW);
#if SW == 1
            S0.s0 = idot4(A0.s0123, W.s0246, S0.s0);
#if OW_BLOCK >= 2
            S0.s2 = idot4(A0.s1234, W.s0246, S0.s2);
#endif
#if OW_BLOCK >= 3
            S0.s4 = idot4(A0.s2345, W.s0246, S0.s4);
#endif
#if OW_BLOCK >= 4
            S0.s6 = idot4(A0.s3456, W.s0246, S0.s6);
#endif
#if OW_BLOCK >= 5
            S0.s8 = idot4(A0.s4567, W.s0246, S0.s8);
#endif
#if OC_BLOCK >= 6
            S0.sa = idot4(A0.s5678, W.s0246, S0.sa);
#endif
#if OW_BLOCK >= 7
            S0.sc = idot4(A0.s6789, W.s0246, S0.sc);
#endif
#if OW_BLOCK >= 8
            S0.se = idot4(A0.s789a, W.s0246, S0.se);
#endif
#if OW_BLOCK >= 9
            S1.s0 = idot4(A0.s89ab, W.s0246, S1.s0);
#endif
#if OW_BLOCK >= 10
            S1.s2 = idot4(A0.s9abc, W.s0246, S1.s2);
#endif
#if OW_BLOCK >= 11
            S1.s4 = idot4(A0.sabcd, W.s0246, S1.s4);
#endif
#if OW_BLOCK >= 12
            S1.s6 = idot4(A0.sbcde, W.s0246, S1.s6);
#endif
#if OW_BLOCK >= 13
            S1.s8 = idot4(A0.scdef, W.s0246, S1.s8);
#endif
#if OW_BLOCK >= 14
            S1.sa = idot4((SRC_DATA4_T)(A0.sde, A0.sf, right0), W.s0246, S1.sa);
#endif
#if OW_BLOCK >= 15
            S1.sc = idot4(
                    (SRC_DATA4_T)(A0.sef, right0, right1), W.s0246, S1.sc);
#endif
            S0.s1 = idot4(A1.s0123, W.s1357, S0.s1);
#if OW_BLOCK >= 2
            S0.s3 = idot4(A1.s1234, W.s1357, S0.s3);
#endif
#if OW_BLOCK >= 3
            S0.s5 = idot4(A1.s2345, W.s1357, S0.s5);
#endif
#if OW_BLOCK >= 4
            S0.s7 = idot4(A1.s3456, W.s1357, S0.s7);
#endif
#if OW_BLOCK >= 5
            S0.s9 = idot4(A1.s4567, W.s1357, S0.s9);
#endif
#if OW_BLOCK >= 6
            S0.sb = idot4(A1.s5678, W.s1357, S0.sb);
#endif
#if OW_BLOCK >= 7
            S0.sd = idot4(A1.s6789, W.s1357, S0.sd);
#endif
#if OW_BLOCK >= 8
            S0.sf = idot4(A1.s789a, W.s1357, S0.sf);
#endif
#if OW_BLOCK >= 9
            S1.s1 = idot4(A1.s89ab, W.s1357, S1.s1);
#endif
#if OW_BLOCK >= 10
            S1.s3 = idot4(A1.s9abc, W.s1357, S1.s3);
#endif
#if OW_BLOCK >= 11
            S1.s5 = idot4(A1.sabcd, W.s1357, S1.s5);
#endif
#if OW_BLOCK >= 12
            S1.s7 = idot4(A1.sbcde, W.s1357, S1.s7);
#endif
#if OW_BLOCK >= 13
            S1.s9 = idot4(A1.scdef, W.s1357, S1.s9);
#endif
#if OW_BLOCK >= 14
            S1.sb = idot4((SRC_DATA4_T)(A1.sde, A1.sf, 0), W.s1357, S1.sb);
#endif
#if OW_BLOCK >= 15
            S1.sd = idot4((SRC_DATA4_T)(A1.sef, right1, 0), W.s1357, S1.sd);
#endif

#elif SW == 2
            S0.s0 = idot4(A0.s0123, W.s0246, S0.s0);
#if OW_BLOCK >= 2
            S0.s2 = idot4(A0.s2345, W.s0246, S0.s2);
#endif
#if OW_BLOCK >= 3
            S0.s4 = idot4(A0.s4567, W.s0246, S0.s4);
#endif
#if OW_BLOCK >= 4
            S0.s6 = idot4(A0.s6789, W.s0246, S0.s6);
#endif
#if OW_BLOCK >= 5
            S0.s8 = idot4(A0.s89ab, W.s0246, S0.s8);
#endif
#if OW_BLOCK >= 6
            S0.sa = idot4(A0.sabcd, W.s0246, S0.sa);
#endif
#if OW_BLOCK >= 7
            S0.sc = idot4(A0.scdef, W.s0246, S0.sc);
#endif
#if OW_BLOCK >= 8

            S0.se = idot4((SRC_DATA4_T)(A0.sef, right0.s0, right0.s1), W.s0246,
                    S0.se);
#endif
#if OW_BLOCK >= 9
            S1.s0 = idot4(right0.s0123, W.s0246, S1.s0);
#endif
#if OW_BLOCK >= 10
            S1.s2 = idot4(right0.s2345, W.s0246, S1.s2);
#endif
#if OW_BLOCK >= 11
            S1.s4 = idot4(right0.s4567, W.s0246, S1.s4);
#endif
#if OW_BLOCK >= 12
            S1.s6 = idot4(right0.s6789, W.s0246, S1.s6);
#endif
#if OW_BLOCK >= 13
            S1.s8 = idot4(right0.s89ab, W.s0246, S1.s8);
#endif
#if OW_BLOCK >= 14
            S1.sa = idot4(right0.sabcd, W.s0246, S1.sa);
#endif
#if OW_BLOCK >= 15
            S1.sc = idot4(right0.scdef, W.s0246, S1.sc);
#endif

            S0.s1 = idot4(A1.s0123, W.s1357, S0.s1);
#if OW_BLOCK >= 2
            S0.s3 = idot4(A1.s2345, W.s1357, S0.s3);
#endif
#if OW_BLOCK >= 3
            S0.s5 = idot4(A1.s4567, W.s1357, S0.s5);
#endif
#if OW_BLOCK >= 4
            S0.s7 = idot4(A1.s6789, W.s1357, S0.s7);
#endif
#if OW_BLOCK >= 5
            S0.s9 = idot4(A1.s89ab, W.s1357, S0.s9);
#endif
#if OW_BLOCK >= 6
            S0.sb = idot4(A1.sabcd, W.s1357, S0.sb);
#endif
#if OW_BLOCK >= 7
            S0.sd = idot4(A1.scdef, W.s1357, S0.sd);
#endif
#if OW_BLOCK >= 8

            S0.sf = idot4((SRC_DATA4_T)(A1.sef, right1.s0, right1.s1), W.s1357,
                    S0.sf);
#endif
#if OW_BLOCK >= 9
            S1.s1 = idot4(right1.s0123, W.s1357, S1.s1);
#endif
#if OW_BLOCK >= 10
            S1.s3 = idot4(right1.s2345, W.s1357, S1.s3);
#endif
#if OW_BLOCK >= 11
            S1.s5 = idot4(right1.s4567, W.s1357, S1.s5);
#endif
#if OW_BLOCK >= 12
            S1.s7 = idot4(right1.s6789, W.s1357, S1.s7);
#endif
#if OW_BLOCK >= 13
            S1.s9 = idot4(right1.s89ab, W.s1357, S1.s9);
#endif
#if OW_BLOCK >= 14
            S1.sb = idot4(right1.sabcd, W.s1357, S1.sb);
#endif
#if OW_BLOCK >= 15
            S1.sd = idot4(right1.scdef, W.s1357, S1.sd);
#endif
#else
#error // SW > 2
#endif

            src += G * IW * (1 + DH);
            wei += OC_BLOCK * KW;
        }
        src += G * IW * (IH * (1 + DD) - KH * (1 + DH));
    }

#if WITH_POST_OP && !SUM_SCALE1 || SCALES_PER_OC || SCALES_COMMON
    float16 tmp00 = convert_float16(S0) * SCALE;
    float16 tmp01 = convert_float16(S1) * SCALE;
#define ACC_DATA_TYPE float
#define ACC0 tmp00
#define ACC1 tmp01
#else
#define ACC_DATA_TYPE int
#define ACC0 S0
#define ACC1 S1
#endif

    DST_DATA16_T D0 = 0;
    DST_DATA16_T D1 = 0;

#if WITH_SUM
    if (OW_TAIL != 0 && ow + OW_BLOCK >= OW) {
        block_read_dst(min(8, OW_TAIL), &D0, dst, g);
        block_read_dst(OW_TAIL - 8, &D1, dst + 8 * G, g);
    } else {
        block_read_dst(min(8, OW_BLOCK), &D0, dst, g);
        block_read_dst(OW_BLOCK - 8, &D1, dst + 8 * G, g);
    }
#endif

    SUM_DATA16_T D0_sdt = AS_SUM_DATA16_T(D0);
    SUM_DATA16_T D1_sdt = AS_SUM_DATA16_T(D1);

#define APPLY_POST_OPS_COMMON(accumulator, sum, offset) \
    { \
        for (int didx = 0; didx < 16; ++didx) { \
            int po_mb = mb; \
            int po_oc = g * OC + 2 * get_sub_group_local_id() + (didx % 2); \
            ACC_DATA_TYPE accum = accumulator[didx]; \
            SUM_DATA_T sum_di = sum[didx]; \
            APPLY_POST_OPS(accum, ACC_DATA_TYPE, sum_di, SUM_DATA_T, po_mb, 1, \
                    po_oc, 1, 0, 1, 0, 1, 0, 1, 0, 1); \
            accumulator[didx] = accum; \
        } \
    }
    APPLY_POST_OPS_COMMON(ACC0, D0_sdt, 0);
    APPLY_POST_OPS_COMMON(ACC1, D1_sdt, 8);

    DST_DATA16_T R0 = CONVERT_DST_DATA16_T(ACC0);
    DST_DATA16_T R1 = CONVERT_DST_DATA16_T(ACC1);

    if (OW_TAIL != 0 && ow + OW_BLOCK > OW) {
        block_write_dst(min(8, OW_TAIL), &R0, dst, g);
        block_write_dst(OW_TAIL - 8, &R1, dst + 8 * G, g);
    } else {
        block_write_dst(min(8, OW_BLOCK), &R0, dst, g);
        block_write_dst(OW_BLOCK - 8, &R1, dst + 8 * G, g);
    }
}

void block_read_dst(
        int n, DST_DATA_T *d, const __global DST_DATA_T *dst, const int g) {
    int sglid = get_sub_group_local_id();
#if DST_DT_S8 || DST_DT_U8
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < n; i++) {
        const int gtail = (g + OC_BLOCK) <= G ? G : G % OC_BLOCK;
        if (2 * sglid < gtail) { d[i * 2] = dst[i * G + 2 * sglid]; }
        if (1 + 2 * sglid < gtail) {
            d[i * 2 + 1] = dst[i * G + 1 + 2 * sglid];
        }
    }
    return;
#elif DST_DT_S32 || DST_DT_F32
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < n; i++) {
        uint block0 = 0, block1 = 0;
        if ((g + OC_BLOCK) <= G || sglid < (G % OC_BLOCK)) {
            block0 = ((const __global uint *)(dst + i * G))[sglid];
        }
        if ((g + OC_BLOCK) <= G || sglid < (G % OC_BLOCK) - SUB_GROUP_SIZE) {
            block1 = ((const __global uint *)(dst + i * G
                    + SUB_GROUP_SIZE))[sglid];
        }

        int from00 = min(2 * sglid, SUB_GROUP_SIZE - 1);
        int from01 = max(2 * sglid - 16, 0);
        int from10 = min(2 * sglid + 1, SUB_GROUP_SIZE - 1);
        int from11 = max(2 * sglid + 1 - 16, 0);

        uint block00 = intel_sub_group_shuffle(block0, from00);
        uint block01 = intel_sub_group_shuffle(block1, from01);
        uint block10 = intel_sub_group_shuffle(block0, from10);
        uint block11 = intel_sub_group_shuffle(block1, from11);

        block0 = (2 * sglid < SUB_GROUP_SIZE) ? block00 : block01;
        block1 = (2 * sglid + 1 < SUB_GROUP_SIZE) ? block10 : block11;

        *(DST_DATA2_T *)(&d[i * 2]) = AS_DST_DATA2_T((uint2)(block0, block1));
    }
    return;
#else
#error "Not expected"
#endif
}

// Shuffled write.
void block_write_dst(
        int n, const DST_DATA_T *d, __global DST_DATA_T *dst, const int g) {
    int sglid = get_sub_group_local_id();
    __attribute__((opencl_unroll_hint)) // attr:no-format
    for (int i = 0; i < n; i++) {
        DST_DATA2_T block = AS_DST_DATA2_T(*(DST_DATA2_T *)(&d[i * 2]));
        const int gtail = (g + OC_BLOCK) <= G ? G : G % OC_BLOCK;
        if (2 * sglid < gtail) { dst[i * G + 2 * sglid] = block.S0; }
        if (1 + 2 * sglid < gtail) { dst[i * G + 1 + 2 * sglid] = block.S1; }
    }
    return;
}

void block_read_src(int n, ushort *s, const __global ushort *src, const int g) {
    int sglid = get_sub_group_local_id();
#if G % 2 == 0
    for (int i = 0; i < n; i++) {
        if (sglid < ((g + IC_BLOCK) <= G ? 16 : (G % IC_BLOCK) / 2)) {
            s[i] = src[i * G / 2 + sglid];
        } else
            s[i] = 0;
    }
#else
    uchar *sc = (uchar *)s;
    const __global uchar *src_c = (const __global uchar *)src;

    if ((g + IC_BLOCK) <= G || (g + IC_BLOCK > G && sglid < G % IC_BLOCK - 1)) {
        for (int i = 0; i < n; i++) {
            sc[i * 2] = src_c[i * G + sglid * 2];
            sc[i * 2 + 1] = src_c[i * G + sglid * 2 + 1];
        }
    } else if (sglid == G % IC_BLOCK - 1) {
        for (int i = 0; i < n; i++) {
            sc[i * 2] = src_c[i * G + sglid * 2];
            sc[i * 2 + 1] = 0;
        }
    } else {
        for (int i = 0; i < n; i++) {
            sc[i * 2] = 0;
            sc[i * 2 + 1] = 0;
        }
    }
#endif
}
