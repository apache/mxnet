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

#define KDHW_SIZE (KH * KW * KD)

#if (PW % 4 == 0) && (SRC_SLM_SIZE % 4 == 0)
#define WRITE_SLM_BLOCK(p, data) block_write(p, data)
#define WRITE_SLM_BLOCK_SHORT(p, data) block_write_us(p, data)
#else
#define WRITE_SLM_BLOCK(p, data) block_write_emu(p, data)
#define WRITE_SLM_BLOCK_SHORT(p, data) block_write_us_emu(p, data)
#endif

#define GET_INT_BLOCK(SRC_SLM, SLM_INDEX, SRC_GLOBAL, GLOBAL_INDEX) \
    uchar4 res = 0; \
    for (int j = 0; j < IC; j++) { \
        res[j] = SRC_GLOBAL[GLOBAL_INDEX + j * IH * IW * ID]; \
    } \
    SRC_SLM[SLM_INDEX] = as_int(res);

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_WHT(data, idx) \
    data = as_int(intel_sub_group_block_read((__global uint *)&wei[idx]));

#define BLOCK_READ_WHT8(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

#define BLOCK_READ_SCALES(data, idx) \
    data = as_float4(intel_sub_group_block_read4( \
            (__global uint *)&scales_per_oc[idx]));

#if SCALES_PER_OC
#define SCALE_VEC4 scales.s01230123
#define SCALE scales
#elif SCALES_COMMON
#define SCALE_VEC4 scale
#define SCALE scale
#else
#define SCALE_VEC4 1
#define SCALE 1
#endif
#define OC_PADD8 ((OC % 8) ? (OC / 8 + 1) * 8 : OC)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) __kernel void
conv_fwd_first_x8s8s32x(const __global uchar *src, const __global char *wei,
        const __global float *bias, __global DST_DATA_T *dst POST_OP_ARGS,
        float scale, const __global float *scales_per_oc,
        const __global int *src_compensation, const __global int *src_zpoints,
        const __global int *dst_compensation) {

    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;
    const int sub_group_id = get_sub_group_id();
    const int sub_local_id = get_sub_group_local_id();
    const int oc = (sub_group_id % OC_GROUP);
    const int sp = (sub_group_id / OC_GROUP);
    const int g = (group_oc + oc) / OC_NCHUNK;
    const int group_ic = IC_NCHUNK * g;
    const int god = group_sp / (OW_PADDED * OH);
    const int gohw = group_sp % (OW_PADDED * OH);
    const int goh = gohw / OW_PADDED;
    const int gow = OW_BLOCK * (gohw % OW_PADDED);
    const int gid = god * SD;
    const int gih = goh * SH;
    const int giw = gow * SW;
    const int local_ow = OW_BLOCK * sp;
    const int local_iw = local_ow * SW;
    const int od = god;
    const int ow = gow + local_ow;
    const int oh = goh;
    const int id = gid - PD;
    const int iw = giw + local_iw - PW;
    const int ih = gih - PH;

    __local uint S_slice[SRC_SLM_SIZE * KH * KD];
    __local uint *S_part = S_slice + (sp * SW * OW_BLOCK + PW);
    __local MMAD_DATA_T *S_work = S_slice + (sp * SW * OW_BLOCK);

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK
            * (group_mb / MB_BLOCK);
    dst += OC_BLOCK * (group_mb % MB_BLOCK);
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);

    src += IC_BLOCK * ID * IH * IW * G * group_mb;
    src += IC_BLOCK * (IW * IH * id + IW * ih + iw + PW);

    wei += 4 * KDHW_SIZE * OC_BLOCK * (group_oc + oc);

    /* WORK WITH SLM */
    const bool left_tail = iw < 0;
    const bool left_nozero_tail = sub_group_id == 0 && iw >= 0;
    const bool right_tail = (iw + PW + OW_SLM_TAIL >= IW) && (iw + PW < IW);
    const bool empty = (iw + PW >= IW);
    const bool right_nozero_tail
            = sp == (LWS_1 - 1) && (iw + PW + OW_SLM_TAIL < IW);

    barrier(CLK_LOCAL_MEM_FENCE);
    /* KD */
#if KD > 1
    for (int kd = 0; kd < KD; kd++) {
        if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
            S_part += SRC_SLM_SIZE * KH;
            src += IC_BLOCK * IW * IH * (1 + DD);
            continue;
        }
#endif
        /* KH */
#if KH > 1
        for (int kh = 0; kh < KH; kh++) {
            if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                S_part += SRC_SLM_SIZE;
                src += IC_BLOCK * IW * (1 + DH);
                continue;
            }
#endif
            /* KW */
            /* left tail */
#if PW > 0
            if (left_tail) {
                for (int i = -PW; i < 0; i++) {
                    S_part[i] = 0;
                }
            }
#endif
            /* right tail */
#if ZERO_TAIL > 0
            if (right_tail) {
                for (int i = OW_SLM_TAIL;
                        i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW; i++) {
                    S_part[i] = 0;
                }
            }
#if SLM_WORKING_GROUPS < OW_NCHUNK
            if (empty) {
                for (int i = 0; i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                        i++) {
                    WRITE_SLM_BLOCK(S_part + i * 8, 0);
                }
            }
#endif
#endif
#if SLM_WORKING_GROUPS < OW_NCHUNK
            if (iw + PW < IW) {
#endif
#if OW_NCHUNK > LWS_1
                /* Copy tails in case of multigroups */
                if (ow < OW) {
#if PW > 0
                    if (left_nozero_tail) {
                        for (int i = -PW; i < 0; i++) {
#if NCHW == 1
                            GET_INT_BLOCK(S_part, i, src, i);

#else
                            S_part[i] = ((__global uint *)src)[i];
#endif
                        }
                    }
#endif

                    if (right_nozero_tail) {
                        for (int i = SW * OW_BLOCK;
                                i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                                i++) {
#if NCHW == 1
                            GET_INT_BLOCK(S_part, i, src, i);
#else
                            S_part[i] = ((__global uint *)src)[i];
#endif
                        }
                    }
#endif

#if OW_SLM_TAIL != OW_BLOCK * SW
                    /* Copy last block to SLM */
                    if (right_tail) {
                        __attribute__((opencl_unroll_hint)) for (int i = 0; i
                                                                 < OW_SLM_TAIL;
                                                                 i++) {
#if NCHW == 1
                            GET_INT_BLOCK(S_part, i, src, i);
#else
                            S_part[i] = ((__global uint *)src)[i];
#endif
                        }
                    } else {
#endif
#if (SW * OW_BLOCK) % 8 == 0
                        /* Copy block to SLM */
                        __attribute__((
                                opencl_unroll_hint)) for (int i = 0;
                                                          i < SW * OW_BLOCK;
                                                          i += 8) {
#if NCHW == 1
                            uchar4 res = 0;
                            for (int j = 0; j < IC; j++) {
                                res[j] = intel_sub_group_block_read_uc(
                                        src + i + j * IH * IW * ID);
                            }
                            WRITE_SLM_BLOCK(S_part + i, as_int(res));
#else
                            WRITE_SLM_BLOCK(S_part + i,
                                    intel_sub_group_block_read((
                                            const __global uint
                                                    *)(&src[i * IC_BLOCK])));
#endif
                        }
#elif (SW * OW_BLOCK) % 4 == 0 && NCHW == 0
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < SW * OW_BLOCK;
                                             i += 4) {
        WRITE_SLM_BLOCK_SHORT(S_part + i,
                intel_sub_group_block_read_us(
                        (const __global ushort *)(&src[i * IC_BLOCK])));
    }
#else
    __attribute__((opencl_unroll_hint)) for (int i = 0; i < SW * OW_BLOCK;
                                             i++) {
        GET_INT_BLOCK(S_part, i, src, i);
    }
#endif

#if OW_SLM_TAIL != OW_BLOCK * SW
                    }
#endif

#if OW_NCHUNK > LWS_1
                }
#endif
#if SLM_WORKING_GROUPS < OW_NCHUNK
            }
#endif
#if KH > 1
            S_part += SRC_SLM_SIZE;
            src += IC_BLOCK * IW * (1 + DH);
        }
        S_part -= SRC_SLM_SIZE * KH;
        src -= IC_BLOCK * KH * IW * (1 + DH);
#endif
#if KD > 1
        S_part += SRC_SLM_SIZE * KH;
        src += IC_BLOCK * IW * IH * (1 + DD);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    MMAD_DATA8_T S;
    int8 W0 = 0, W1 = 0, W2 = 0, W3 = 0;
    int W00 = 0, W10 = 0, W20 = 0, W30 = 0;
    int8 C00 = 0;
    int8 C10 = 0;
    int8 C20 = 0;
    int8 C30 = 0;
#if OW_BLOCK == 12
    MMAD_DATA4_T SS;
    int4 C01 = 0;
    int4 C11 = 0;
    int4 C21 = 0;
    int4 C31 = 0;
#endif

#if OW_BLOCK == 16
    int8 C01 = 0;
    int8 C11 = 0;
    int8 C21 = 0;
    int8 C31 = 0;
#endif

#if WITH_SRC_ZPOINTS
#if WITH_SRC_ZPOINTS_PER_IC
    int4 z = 0;
    if (IC > 0) z.s0 = src_zpoints[0];
    if (IC > 1) z.s1 = src_zpoints[1];
    if (IC > 2) z.s2 = src_zpoints[2];
    if (IC > 3) z.s3 = src_zpoints[3];
#else
    int4 z = read_src_zero_point(src_zpoints);
#endif // WITH_SRC_ZPOINTS_PER_IC
#endif // WITH_SRC_ZPOINTS

#if !WITH_SRC_ZPOINTS
    for (int i = 0; i < KDHW_SIZE - KDHW_SIZE % 8; i += 8) {
        const int ihw = (i + sub_local_id) % (KW * KH);
        const int filter_iw = (ihw % KW) * (1 + DW);
        const int filter_ih = ihw / KW;
        const int filter_id = (i + sub_local_id) / (KH * KW);
        const int filter = (filter_ih * (1 + DH) + ih >= 0)
                && (filter_ih * (1 + DH) + ih < IH)
                && (filter_id * (1 + DD) + id >= 0
                        && filter_id * (1 + DD) + id < ID);

        BLOCK_READ_WHT8(W0, 0);
#if OC_PADD8 * 4 > OC_BLOCK
        BLOCK_READ_WHT8(W1, KDHW_SIZE * OC_BLOCK);
#endif
#if OC_PADD8 * 4 > OC_BLOCK * 2
        BLOCK_READ_WHT8(W2, 2 * KDHW_SIZE * OC_BLOCK);
#endif
#if OC_PADD8 * 4 > OC_BLOCK * 3
        BLOCK_READ_WHT8(W3, 3 * KDHW_SIZE * OC_BLOCK);
#endif
        if (filter) {
            S.s0 = S_work[SW * 0 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s1 = S_work[SW * 1 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s2 = S_work[SW * 2 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s3 = S_work[SW * 3 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s4 = S_work[SW * 4 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s5 = S_work[SW * 5 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s6 = S_work[SW * 6 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s7 = S_work[SW * 7 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
#if OW_BLOCK == 12
            SS.s0 = S_work[SW * 8 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s1 = S_work[SW * 9 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s2 = S_work[SW * 10 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s3 = S_work[SW * 11 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
#endif // OW_BLOCK == 12
        } else {
            S = 0;
#if OW_BLOCK == 12
            SS = 0;
#endif
        }

        C00 = mmad8x8(S, W0, C00);
        C10 = mmad8x8(S, W1, C10);
        C20 = mmad8x8(S, W2, C20);
        C30 = mmad8x8(S, W3, C30);
#if OW_BLOCK == 12
        C01 = mmad8x4(SS, W0, C01);
        C11 = mmad8x4(SS, W1, C11);
        C21 = mmad8x4(SS, W2, C21);
        C31 = mmad8x4(SS, W3, C31);
#endif // OW_BLOCK == 12

#if OW_BLOCK == 16
        if (filter) {
            S.s0 = S_work[SW * 8 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s1 = S_work[SW * 9 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s2 = S_work[SW * 10 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s3 = S_work[SW * 11 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s4 = S_work[SW * 12 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s5 = S_work[SW * 13 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s6 = S_work[SW * 14 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s7 = S_work[SW * 15 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
        } else {
            S = 0;
        }

        C01 = mmad8x8(S, W0, C01);
        C11 = mmad8x8(S, W1, C11);
        C21 = mmad8x8(S, W2, C21);
        C31 = mmad8x8(S, W3, C31);
#endif // OW_BLOCK == 16

        wei += OC_BLOCK * 8;
    }
#endif // !WITH_SRC_ZPOINTS

#if WITH_SRC_ZPOINTS
    for (int i = 0; i < KDHW_SIZE; i++)
#else
    for (int i = KDHW_SIZE - KDHW_SIZE % 8; i < KDHW_SIZE; i++)
#endif // WITH_SRC_ZPOINTS
    {
        const int ihw = (i) % (KW * KH);
        const int filter_iw = (ihw % KW) * (1 + DW);
        const int filter_ih = ihw / KW;
        const int filter_id = (i) / (KH * KW);
        const int filter = (filter_ih * (1 + DH) + ih >= 0)
                && (filter_ih * (1 + DH) + ih < IH)
                && (filter_id * (1 + DD) + id >= 0
                        && filter_id * (1 + DD) + id < ID);
        if (filter) {
            BLOCK_READ_WHT(W00, 0);
#if OC_PADD8 * 4 > OC_BLOCK
            BLOCK_READ_WHT(W10, KDHW_SIZE * OC_BLOCK);
#endif
#if OC_PADD8 * 4 > OC_BLOCK * 2
            BLOCK_READ_WHT(W20, 2 * KDHW_SIZE * OC_BLOCK);
#endif
#if OC_PADD8 * 4 > OC_BLOCK * 3
            BLOCK_READ_WHT(W30, 3 * KDHW_SIZE * OC_BLOCK);
#endif

#if WITH_SRC_ZPOINTS
            int4 src_comp;
            src_comp.s0 = calc_src_compensation_x4(z, W00);
            src_comp.s1 = calc_src_compensation_x4(z, W10);
            src_comp.s2 = calc_src_compensation_x4(z, W20);
            src_comp.s3 = calc_src_compensation_x4(z, W30);

            unroll_for(uint j = 0; j < 8; ++j) {
                if (filter_iw + iw + j * SW >= 0
                        && filter_iw + iw + j * SW < IW) {
                    C00[j] -= src_comp.s0;
                    C10[j] -= src_comp.s1;
                    C20[j] -= src_comp.s2;
                    C30[j] -= src_comp.s3;
                }
            }
#if (OW_BLOCK == 12) || (OW_BLOCK == 16)
            unroll_for(uint j = 8; j < OW_BLOCK; ++j) {
                if (filter_iw + iw + j * SW >= 0
                        && filter_iw + iw + j * SW < IW) {
                    C01[j - 8] -= src_comp.s0;
                    C11[j - 8] -= src_comp.s1;
                    C21[j - 8] -= src_comp.s2;
                    C31[j - 8] -= src_comp.s3;
                }
            }
#endif // (OW_BLOCK == 12) || (OW_BLOCK == 16)
#endif // WITH_SRC_ZPOINTS

            S.s0 = S_work[SW * 0 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s1 = S_work[SW * 1 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s2 = S_work[SW * 2 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s3 = S_work[SW * 3 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s4 = S_work[SW * 4 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s5 = S_work[SW * 5 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s6 = S_work[SW * 6 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s7 = S_work[SW * 7 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
#if OW_BLOCK == 12
            SS.s0 = S_work[SW * 8 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s1 = S_work[SW * 9 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s2 = S_work[SW * 10 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            SS.s3 = S_work[SW * 11 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
#endif // OW_BLOCK == 12
            C00.s0 = idot4(AS_SRC_DATA4_T(S.s0), as_char4(W00), C00.s0);
            C00.s1 = idot4(AS_SRC_DATA4_T(S.s1), as_char4(W00), C00.s1);
            C00.s2 = idot4(AS_SRC_DATA4_T(S.s2), as_char4(W00), C00.s2);
            C00.s3 = idot4(AS_SRC_DATA4_T(S.s3), as_char4(W00), C00.s3);
            C00.s4 = idot4(AS_SRC_DATA4_T(S.s4), as_char4(W00), C00.s4);
            C00.s5 = idot4(AS_SRC_DATA4_T(S.s5), as_char4(W00), C00.s5);
            C00.s6 = idot4(AS_SRC_DATA4_T(S.s6), as_char4(W00), C00.s6);
            C00.s7 = idot4(AS_SRC_DATA4_T(S.s7), as_char4(W00), C00.s7);

            C10.s0 = idot4(AS_SRC_DATA4_T(S.s0), as_char4(W10), C10.s0);
            C10.s1 = idot4(AS_SRC_DATA4_T(S.s1), as_char4(W10), C10.s1);
            C10.s2 = idot4(AS_SRC_DATA4_T(S.s2), as_char4(W10), C10.s2);
            C10.s3 = idot4(AS_SRC_DATA4_T(S.s3), as_char4(W10), C10.s3);
            C10.s4 = idot4(AS_SRC_DATA4_T(S.s4), as_char4(W10), C10.s4);
            C10.s5 = idot4(AS_SRC_DATA4_T(S.s5), as_char4(W10), C10.s5);
            C10.s6 = idot4(AS_SRC_DATA4_T(S.s6), as_char4(W10), C10.s6);
            C10.s7 = idot4(AS_SRC_DATA4_T(S.s7), as_char4(W10), C10.s7);

            C20.s0 = idot4(AS_SRC_DATA4_T(S.s0), as_char4(W20), C20.s0);
            C20.s1 = idot4(AS_SRC_DATA4_T(S.s1), as_char4(W20), C20.s1);
            C20.s2 = idot4(AS_SRC_DATA4_T(S.s2), as_char4(W20), C20.s2);
            C20.s3 = idot4(AS_SRC_DATA4_T(S.s3), as_char4(W20), C20.s3);
            C20.s4 = idot4(AS_SRC_DATA4_T(S.s4), as_char4(W20), C20.s4);
            C20.s5 = idot4(AS_SRC_DATA4_T(S.s5), as_char4(W20), C20.s5);
            C20.s6 = idot4(AS_SRC_DATA4_T(S.s6), as_char4(W20), C20.s6);
            C20.s7 = idot4(AS_SRC_DATA4_T(S.s7), as_char4(W20), C20.s7);

            C30.s0 = idot4(AS_SRC_DATA4_T(S.s0), as_char4(W30), C30.s0);
            C30.s1 = idot4(AS_SRC_DATA4_T(S.s1), as_char4(W30), C30.s1);
            C30.s2 = idot4(AS_SRC_DATA4_T(S.s2), as_char4(W30), C30.s2);
            C30.s3 = idot4(AS_SRC_DATA4_T(S.s3), as_char4(W30), C30.s3);
            C30.s4 = idot4(AS_SRC_DATA4_T(S.s4), as_char4(W30), C30.s4);
            C30.s5 = idot4(AS_SRC_DATA4_T(S.s5), as_char4(W30), C30.s5);
            C30.s6 = idot4(AS_SRC_DATA4_T(S.s6), as_char4(W30), C30.s6);
            C30.s7 = idot4(AS_SRC_DATA4_T(S.s7), as_char4(W30), C30.s7);
#if OW_BLOCK == 12
            C01.s0 = idot4(AS_SRC_DATA4_T(SS.s0), as_char4(W00), C01.s0);
            C01.s1 = idot4(AS_SRC_DATA4_T(SS.s1), as_char4(W00), C01.s1);
            C01.s2 = idot4(AS_SRC_DATA4_T(SS.s2), as_char4(W00), C01.s2);
            C01.s3 = idot4(AS_SRC_DATA4_T(SS.s3), as_char4(W00), C01.s3);

            C11.s0 = idot4(AS_SRC_DATA4_T(SS.s0), as_char4(W10), C11.s0);
            C11.s1 = idot4(AS_SRC_DATA4_T(SS.s1), as_char4(W10), C11.s1);
            C11.s2 = idot4(AS_SRC_DATA4_T(SS.s2), as_char4(W10), C11.s2);
            C11.s3 = idot4(AS_SRC_DATA4_T(SS.s3), as_char4(W10), C11.s3);

            C21.s0 = idot4(AS_SRC_DATA4_T(SS.s0), as_char4(W20), C21.s0);
            C21.s1 = idot4(AS_SRC_DATA4_T(SS.s1), as_char4(W20), C21.s1);
            C21.s2 = idot4(AS_SRC_DATA4_T(SS.s2), as_char4(W20), C21.s2);
            C21.s3 = idot4(AS_SRC_DATA4_T(SS.s3), as_char4(W20), C21.s3);

            C31.s0 = idot4(AS_SRC_DATA4_T(SS.s0), as_char4(W30), C31.s0);
            C31.s1 = idot4(AS_SRC_DATA4_T(SS.s1), as_char4(W30), C31.s1);
            C31.s2 = idot4(AS_SRC_DATA4_T(SS.s2), as_char4(W30), C31.s2);
            C31.s3 = idot4(AS_SRC_DATA4_T(SS.s3), as_char4(W30), C31.s3);
#endif // OW_BLOCK == 12

#if OW_BLOCK == 16
            S.s0 = S_work[SW * 8 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s1 = S_work[SW * 9 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s2 = S_work[SW * 10 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s3 = S_work[SW * 11 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s4 = S_work[SW * 12 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s5 = S_work[SW * 13 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s6 = S_work[SW * 14 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];
            S.s7 = S_work[SW * 15 + SRC_SLM_SIZE * KH * filter_id
                    + SRC_SLM_SIZE * filter_ih + filter_iw];

            C01.s0 = idot4(AS_SRC_DATA4_T(S.s0), as_char4(W00), C01.s0);
            C01.s1 = idot4(AS_SRC_DATA4_T(S.s1), as_char4(W00), C01.s1);
            C01.s2 = idot4(AS_SRC_DATA4_T(S.s2), as_char4(W00), C01.s2);
            C01.s3 = idot4(AS_SRC_DATA4_T(S.s3), as_char4(W00), C01.s3);
            C01.s4 = idot4(AS_SRC_DATA4_T(S.s4), as_char4(W00), C01.s4);
            C01.s5 = idot4(AS_SRC_DATA4_T(S.s5), as_char4(W00), C01.s5);
            C01.s6 = idot4(AS_SRC_DATA4_T(S.s6), as_char4(W00), C01.s6);
            C01.s7 = idot4(AS_SRC_DATA4_T(S.s7), as_char4(W00), C01.s7);

            C11.s0 = idot4(AS_SRC_DATA4_T(S.s0), as_char4(W10), C11.s0);
            C11.s1 = idot4(AS_SRC_DATA4_T(S.s1), as_char4(W10), C11.s1);
            C11.s2 = idot4(AS_SRC_DATA4_T(S.s2), as_char4(W10), C11.s2);
            C11.s3 = idot4(AS_SRC_DATA4_T(S.s3), as_char4(W10), C11.s3);
            C11.s4 = idot4(AS_SRC_DATA4_T(S.s4), as_char4(W10), C11.s4);
            C11.s5 = idot4(AS_SRC_DATA4_T(S.s5), as_char4(W10), C11.s5);
            C11.s6 = idot4(AS_SRC_DATA4_T(S.s6), as_char4(W10), C11.s6);
            C11.s7 = idot4(AS_SRC_DATA4_T(S.s7), as_char4(W10), C11.s7);

            C21.s0 = idot4(AS_SRC_DATA4_T(S.s0), as_char4(W20), C21.s0);
            C21.s1 = idot4(AS_SRC_DATA4_T(S.s1), as_char4(W20), C21.s1);
            C21.s2 = idot4(AS_SRC_DATA4_T(S.s2), as_char4(W20), C21.s2);
            C21.s3 = idot4(AS_SRC_DATA4_T(S.s3), as_char4(W20), C21.s3);
            C21.s4 = idot4(AS_SRC_DATA4_T(S.s4), as_char4(W20), C21.s4);
            C21.s5 = idot4(AS_SRC_DATA4_T(S.s5), as_char4(W20), C21.s5);
            C21.s6 = idot4(AS_SRC_DATA4_T(S.s6), as_char4(W20), C21.s6);
            C21.s7 = idot4(AS_SRC_DATA4_T(S.s7), as_char4(W20), C21.s7);

            C31.s0 = idot4(AS_SRC_DATA4_T(S.s0), as_char4(W30), C31.s0);
            C31.s1 = idot4(AS_SRC_DATA4_T(S.s1), as_char4(W30), C31.s1);
            C31.s2 = idot4(AS_SRC_DATA4_T(S.s2), as_char4(W30), C31.s2);
            C31.s3 = idot4(AS_SRC_DATA4_T(S.s3), as_char4(W30), C31.s3);
            C31.s4 = idot4(AS_SRC_DATA4_T(S.s4), as_char4(W30), C31.s4);
            C31.s5 = idot4(AS_SRC_DATA4_T(S.s5), as_char4(W30), C31.s5);
            C31.s6 = idot4(AS_SRC_DATA4_T(S.s6), as_char4(W30), C31.s6);
            C31.s7 = idot4(AS_SRC_DATA4_T(S.s7), as_char4(W30), C31.s7);
#endif // OW_BLOCK == 16
        }
        wei += OC_BLOCK;
    }
    DST_DATA16_T R1, R2, R3, R4;

#if SCALES_PER_OC
    float4 scales;
    BLOCK_READ_SCALES(scales, (group_oc + oc) * OC_BLOCK);
#endif

#if WITH_BIAS
    float4 bia;
    BLOCK_READ_BIA(bia, (group_oc + oc) * OC_BLOCK);
    bia *= SCALE;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)SCALE, bia);
#define QUANTIZE_ADD_BIAS_4() \
    tmp0 = fma(tmp0, (float8)SCALE_VEC4, bia.s01230123); \
    tmp1 = fma(tmp1, (float8)SCALE_VEC4, bia.s01230123);
#else
#define QUANTIZE_ADD_BIAS() tmp *= SCALE;
#define QUANTIZE_ADD_BIAS_4() \
    tmp0 *= SCALE_VEC4; \
    tmp1 *= SCALE_VEC4;
#endif

#if WITH_DST_ZPOINTS
    int4 dst_zp = read_dst_zero_points_32c(
            dst_compensation, (group_oc + oc) * OC_BLOCK);
#define ADD_DST_COMPENSATION() tmp += convert_float4(dst_zp);
#define ADD_DST_COMPENSATION_4() \
    tmp0 += convert_float8(dst_zp.s01230123); \
    tmp1 += convert_float8(dst_zp.s01230123);
#else
#define ADD_DST_COMPENSATION()
#define ADD_DST_COMPENSATION_4()
#endif // WITH_DST_ZPOINTS

#if WITH_SRC_ZPOINTS
#define ZERO_PAD_DST() tmp = zero_pad_dst_32c(tmp, (group_oc + oc) * OC_BLOCK);
#define ZERO_PAD_DST_4() \
    const int4 zp_mask = as_int4(zero_pad_dst_32c( \
            as_float4((int4)(~0u)), (group_oc + oc) * OC_BLOCK)); \
    tmp0 = as_float8(as_int8(tmp0) & zp_mask.s01230123); \
    tmp1 = as_float8(as_int8(tmp1) & zp_mask.s01230123);
#else
#define ZERO_PAD_DST()
#define ZERO_PAD_DST_4()
#endif // WITH_SRC_ZPOINTS

#if WITH_POST_OP

#define DO_POST_OP(i) \
    { \
        SUM_DATA4_T d; \
        if (WITH_SUM) d = AS_SUM_DATA4_T(BLOCK_READ_DST4(dst)); \
        for (int didx = 0; didx < 4; ++didx) { \
            float tmp_i = tmp[didx]; \
            SUM_DATA_T d_i = d[didx]; \
            const int po_mb = group_mb % MB; \
            const int po_oc \
                    = (oc * OC_BLOCK + ((didx * SUB_GROUP_SIZE) % OC_BLOCK) \
                              + sub_local_id) \
                    % (OC * G); \
            APPLY_POST_OPS(tmp_i, float, d_i, SUM_DATA_T, po_mb, 1, po_oc, 1, \
                    0, 1, 0, 1, 0, 1, 0, 1); \
            tmp[didx] = tmp_i; \
        } \
    }

#define DO_POST_OP_4(i) \
    { \
        SUM_DATA16_T d; \
        if (WITH_SUM) d = AS_SUM_DATA16_T(BLOCK_READ_DST16(dst)); \
        float16 tmp_x16 = (float16)(tmp0, tmp1); \
        for (int didx = 0; didx < 16; ++didx) { \
            float tmp_i = tmp_x16[didx]; \
            SUM_DATA_T d_i = d[didx]; \
            const int po_mb = group_mb % MB; \
            const int po_oc \
                    = (oc * OC_BLOCK + ((didx * SUB_GROUP_SIZE) % OC_BLOCK) \
                              + sub_local_id) \
                    % (OC * G); \
            APPLY_POST_OPS(tmp_i, float, d_i, SUM_DATA_T, po_mb, 1, po_oc, 1, \
                    0, 1, 0, 1, 0, 1, 0, 1); \
            tmp_x16[didx] = tmp_i; \
        } \
        tmp0 = tmp_x16.s01234567; \
        tmp1 = tmp_x16.s89abcdef; \
    }

#else

#define DO_POST_OP(i) ;
#define DO_POST_OP_4(i) ;

#endif // #if WITH_POST_OP

#define PACK(C0, C1, C2, C3, idx) \
    do { \
        tmp[0] = C0[idx]; \
        tmp[1] = C1[idx]; \
        tmp[2] = C2[idx]; \
        tmp[3] = C3[idx]; \
    } while (0)

#define PACK_4(C0, C1, C2, C3, idx) \
    do { \
        tmp0.s0 = C0[idx]; \
        tmp0.s1 = C1[idx]; \
        tmp0.s2 = C2[idx]; \
        tmp0.s3 = C3[idx]; \
\
        tmp0.s4 = C0[idx + 1]; \
        tmp0.s5 = C1[idx + 1]; \
        tmp0.s6 = C2[idx + 1]; \
        tmp0.s7 = C3[idx + 1]; \
\
        tmp1.s0 = C0[idx + 2]; \
        tmp1.s1 = C1[idx + 2]; \
        tmp1.s2 = C2[idx + 2]; \
        tmp1.s3 = C3[idx + 2]; \
\
        tmp1.s4 = C0[idx + 3]; \
        tmp1.s5 = C1[idx + 3]; \
        tmp1.s6 = C2[idx + 3]; \
        tmp1.s7 = C3[idx + 3]; \
    } while (0)

#define CONVERT_PACK() \
    do { \
        tmp_cvt = CONVERT_DST_DATA4_T(tmp); \
    } while (0)

#define CONVERT_PACK_4() \
    do { \
        R.s01234567 = CONVERT_DST_DATA8_T(tmp0); \
        R.s89abcdef = CONVERT_DST_DATA8_T(tmp1); \
    } while (0)

#define STORE_DST(C0, C1, C2, C3, i) \
    do { \
        PACK(C0, C1, C2, C3, i); \
        QUANTIZE_ADD_BIAS(); \
        DO_POST_OP(i); \
        ADD_DST_COMPENSATION(); \
        ZERO_PAD_DST(); \
        CONVERT_PACK(); \
        BLOCK_WRITE_DST4(dst, tmp_cvt); \
        dst += OC_BLOCK * MB_BLOCK; \
    } while (0)

#define STORE_DST_4(C0, C1, C2, C3, i) \
    do { \
        PACK_4(C0, C1, C2, C3, i); \
        QUANTIZE_ADD_BIAS_4(); \
        DO_POST_OP_4(i); \
        ADD_DST_COMPENSATION_4(); \
        ZERO_PAD_DST_4(); \
        CONVERT_PACK_4(); \
        BLOCK_WRITE_DST16(dst, R); \
        dst += 4 * OC_BLOCK; \
    } while (0)

    if (ow < OW) {
        float4 tmp;
        DST_DATA4_T tmp_cvt;
        float8 tmp0, tmp1;
        DST_DATA16_T R;

#if OW_TAIL
        if (ow + OW_BLOCK < OW) {
#endif
#if MB_BLOCK == 32
            STORE_DST(C00, C10, C20, C30, 0);
            STORE_DST(C00, C10, C20, C30, 1);
            STORE_DST(C00, C10, C20, C30, 2);
            STORE_DST(C00, C10, C20, C30, 3);

            STORE_DST(C00, C10, C20, C30, 4);
            STORE_DST(C00, C10, C20, C30, 5);
            STORE_DST(C00, C10, C20, C30, 6);
            STORE_DST(C00, C10, C20, C30, 7);
#if OW_BLOCK >= 12
            STORE_DST(C01, C11, C21, C31, 0);
            STORE_DST(C01, C11, C21, C31, 1);
            STORE_DST(C01, C11, C21, C31, 2);
            STORE_DST(C01, C11, C21, C31, 3);
#endif
#if OW_BLOCK == 16
            STORE_DST(C01, C11, C21, C31, 4);
            STORE_DST(C01, C11, C21, C31, 5);
            STORE_DST(C01, C11, C21, C31, 6);
            STORE_DST(C01, C11, C21, C31, 7);
#endif

#else
        STORE_DST_4(C00, C10, C20, C30, 0);
        STORE_DST_4(C00, C10, C20, C30, 4);
#if OW_BLOCK >= 12
        STORE_DST_4(C01, C11, C21, C31, 0);
#endif
#if OW_BLOCK >= 16
        STORE_DST_4(C01, C11, C21, C31, 4);
#endif
#endif
#if OW_TAIL
        } else {

#if OW_TAIL < 4
            for (int i = 0; i < OW_TAIL; i++) {
                STORE_DST(C00, C10, C20, C30, i);
            }
#else
#if MB_BLOCK == 32
            STORE_DST(C00, C10, C20, C30, 0);
            STORE_DST(C00, C10, C20, C30, 1);
            STORE_DST(C00, C10, C20, C30, 2);
            STORE_DST(C00, C10, C20, C30, 3);
#else
            STORE_DST_4(C00, C10, C20, C30, 0);
#endif
#endif
#if OW_TAIL > 4
#if OW_TAIL < 8
            for (int i = 4; i < OW_TAIL; i++) {
                STORE_DST(C00, C10, C20, C30, i);
            }
#else
#if MB_BLOCK == 32
            STORE_DST(C00, C10, C20, C30, 4);
            STORE_DST(C00, C10, C20, C30, 5);
            STORE_DST(C00, C10, C20, C30, 6);
            STORE_DST(C00, C10, C20, C30, 7);
#else
            STORE_DST_4(C00, C10, C20, C30, 4);
#endif
#endif
#if OW_TAIL > 8
#if OW_TAIL < 12
            for (int i = 8; i < OW_TAIL; i++) {
                STORE_DST(C01, C11, C21, C31, i);
            }
#else
#if MB_BLOCK == 32
            STORE_DST(C01, C11, C21, C31, 0);
            STORE_DST(C01, C11, C21, C31, 1);
            STORE_DST(C01, C11, C21, C31, 2);
            STORE_DST(C01, C11, C21, C31, 3);
#else
            STORE_DST_4(C01, C11, C21, C31, 0);
#endif
#endif
#if OW_TAIL > 12
#if OW_TAIL < 16
            for (int i = 12; i < OW_TAIL; i++) {
                STORE_DST(C01, C11, C21, C31, i);
            }
#else
#if MB_BLOCK == 32
            STORE_DST(C01, C11, C21, C31, 4);
            STORE_DST(C01, C11, C21, C31, 5);
            STORE_DST(C01, C11, C21, C31, 6);
            STORE_DST(C01, C11, C21, C31, 7);
#else
            STORE_DST_4(C01, C11, C21, C31, 4);
#endif
#endif
#endif
#endif
#endif
        }
#endif
    }
}
