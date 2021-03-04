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

#if IC % IC_BLOCK != 0
#define IC_NBLOCKS_TAIL ((IC - (IC & ~(IC_BLOCK - 1)) + 3) / 4)
#else
#define IC_NBLOCKS_TAIL 8
#endif

#if OW_BLOCK == 4
#define BLOCK 4
#define ACC_DATA_BLOCK int4
#define SRC_DATA_BLOCK_T MMAD_DATA4_T
#define READ_BLOCK intel_sub_group_block_read4
#define WRITE_LOCAL block_write4

DECLARE_MMAD(
        mmad_tail, IC_NBLOCKS_TAIL, 4, SRC_DATA_BLOCK_T, int8, ACC_DATA_BLOCK)

#define MMAD_FULL mmad8x4
#define MMAD_TAIL mmad_tail
#else
#define BLOCK 8
#define ACC_DATA_BLOCK int8
#define SRC_DATA_BLOCK_T MMAD_DATA8_T
#define READ_BLOCK intel_sub_group_block_read8
#define WRITE_LOCAL block_write8

DECLARE_MMAD(
        mmad_tail, IC_NBLOCKS_TAIL, 8, SRC_DATA_BLOCK_T, int8, ACC_DATA_BLOCK)

#define MMAD_FULL mmad8x8
#define MMAD_TAIL mmad_tail
#endif

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_WHT_1x32(data, idx) \
    data = as_int(intel_sub_group_block_read((__global uint *)&wei[idx]));

#define BLOCK_READ_WHT_8x32(data, idx) \
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
conv_fwd_ow_block_x8s8s32x(const __global SRC_DATA_T *src,
        const __global char *wei, const __global float *bias,
        __global DATA_T *dst POST_OP_ARGS, float scale,
        const __global float *scales_per_oc,
        const __global int *src_compensation, const __global int *src_zpoints,
        const __global int *dst_compensation) {
    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;
    const int sub_group_id = get_sub_group_id();
    const int ocl_local_id = get_local_id(0);
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

    __local uint S_slice[SRC_SLM_SIZE];
    __local uint *S_part = S_slice + IC_BLOCK / 4 * (sp * SW * OW_BLOCK + PW);
    __local uint *S_work = S_slice + IC_BLOCK / 4 * (sp * SW * OW_BLOCK);

    const bool left_tail = iw < 0;
    const bool left_nozero_tail = sub_group_id == 0 && iw >= 0;
    const bool right_tail = (iw + PW + OW_SLM_TAIL >= IW) && (iw + PW < IW);
    const bool empty = (iw + PW >= IW);
    const bool right_nozero_tail
            = sp == (LWS_1 - 1) && (iw + PW + OW_SLM_TAIL < IW);

    dst += OC_BLOCK * OD * OH * OW * MB_BLOCK * (group_oc + oc);
    dst += OC_BLOCK * OD * OH * OW * OC_NCHUNK * G * MB_BLOCK * group_mb;
    dst += OC_BLOCK * MB_BLOCK * (OW * OH * od + OW * oh + ow);
    src += IC_BLOCK * ID * IH * IW * MB_BLOCK * group_ic;
    src += IC_BLOCK * ID * IH * IW * IC_NCHUNK * G * MB_BLOCK * group_mb;
    src += IC_BLOCK * MB_BLOCK * (IW * IH * id + IW * ih + iw + PW);
    wei += IC_BLOCK * KD * KH * KW * OC_BLOCK * (group_oc + oc) * IC_NCHUNK;

    /* Prepare S_slice tails */
#if PW > 0
    if (left_tail) {
        for (int i = 0; i < PW; i++) {
            block_write(S_slice + i * 8, 0);
        }
    }
#endif

#if ZERO_TAIL > 0
    if (right_tail) {
        for (int i = OW_SLM_TAIL; i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                i++) {
            block_write(S_part + i * 8, 0);
        }
    }
#if SLM_WORKING_GROUPS < OW_NCHUNK
    if (empty) {
        for (int i = 0; i < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW; i++) {
            block_write(S_part + i * 8, 0);
        }
    }
#endif
#endif

    ACC_DATA_BLOCK C00 = 0, C01 = 0, C02 = 0, C03 = 0;

    for (int ic_chunk = 0; ic_chunk < IC_NCHUNK; ic_chunk++) {
        SRC_DATA_BLOCK_T S0;

        for (int kd = 0; kd < KD; kd++) {
            if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
                src += IC_BLOCK * MB_BLOCK * IH * IW * (1 + DD);
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            for (int kh = 0; kh < KH; kh++) {
                if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                    src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }

                barrier(CLK_LOCAL_MEM_FENCE);
#if SLM_WORKING_GROUPS < OW_NCHUNK
                if (iw + PW < IW) {
#endif
#if OW_NCHUNK > LWS_1
                    /* Copy tails in case of multigroups */
                    if (ow < OW) {
#if PW > 0
                        if (left_nozero_tail) {
                            for (int i = -PW; i < 0; i++) {
                                block_write(S_part + i * 8,
                                        intel_sub_group_block_read(
                                                (const __global uint *)(&src[i
                                                        * IC_BLOCK])));
                            }
                        }
#endif

                        if (right_nozero_tail) {
                            for (int i = SW * OW_BLOCK; i
                                    < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                                    i++) {
                                block_write(S_part + i * 8,
                                        intel_sub_group_block_read(
                                                (const __global uint *)(&src[i
                                                        * IC_BLOCK])));
                            }
                        }
#endif

#if OW_SLM_TAIL != OW_BLOCK * SW
                        /* Copy last block to SLM */
                        if (right_tail) {
                            __attribute__((
                                    opencl_unroll_hint)) for (int i = 0;
                                                              i < OW_SLM_TAIL;
                                                              i++) {
                                block_write(S_part + i * 8,
                                        intel_sub_group_block_read(
                                                (const __global uint *)(&src[i
                                                        * IC_BLOCK])));
                            }
                        } else {
#endif
                            /* Copy block to SLM */
                            __attribute__((
                                    opencl_unroll_hint)) for (int i = 0;
                                                              i < SW * OW_BLOCK;
                                                              i += OW_BLOCK) {
                                WRITE_LOCAL(S_part + i * 8,
                                        READ_BLOCK(
                                                (const __global uint *)(&src[i
                                                        * IC_BLOCK])));
                            }

#if OW_SLM_TAIL != OW_BLOCK * SW
                        }
#endif

#if OW_NCHUNK > LWS_1
                    }
#endif
#if SLM_WORKING_GROUPS < OW_NCHUNK
                }
#endif
                barrier(CLK_LOCAL_MEM_FENCE);

                for (int kw = 0; kw < KW; kw++) {
                    unroll_for(int i = 0; i < OW_BLOCK; i++) {
                        S0[i] = block_read(
                                S_work + (kw * (1 + DW) + SW * i) * 8);
                    }

                    int8 W0 = 0, W1 = 0, W2 = 0, W3 = 0;
#if IC % IC_BLOCK != 0
                    if (ic_chunk == IC_NCHUNK - 1) {
                        unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                                BLOCK_READ_WHT_1x32(W0[i], (i + 0) * IC_BLOCK);
                        if (OC > 8)
                            unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                                    BLOCK_READ_WHT_1x32(
                                            W1[i], (i + 8) * IC_BLOCK);
                        if (OC > 16)
                            unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                                    BLOCK_READ_WHT_1x32(
                                            W2[i], (i + 16) * IC_BLOCK);
                        if (OC > 24)
                            unroll_for(int i = 0; i < IC_NBLOCKS_TAIL; ++i)
                                    BLOCK_READ_WHT_1x32(
                                            W3[i], (i + 24) * IC_BLOCK);

                        C00 = MMAD_TAIL(S0, W0, C00);
                        if (OC > 8) C01 = MMAD_TAIL(S0, W1, C01);
                        if (OC > 16) C02 = MMAD_TAIL(S0, W2, C02);
                        if (OC > 24) C03 = MMAD_TAIL(S0, W3, C03);
                    } else
#endif // IC % IC_BLOCK != 0
                    {
                        BLOCK_READ_WHT_8x32(W0, 0);
                        if (OC > 8) BLOCK_READ_WHT_8x32(W1, 8 * IC_BLOCK);
                        if (OC > 16) BLOCK_READ_WHT_8x32(W2, 16 * IC_BLOCK);
                        if (OC > 24) BLOCK_READ_WHT_8x32(W3, 24 * IC_BLOCK);

                        C00 = MMAD_FULL(S0, W0, C00);
                        if (OC > 8) C01 = MMAD_FULL(S0, W1, C01);
                        if (OC > 16) C02 = MMAD_FULL(S0, W2, C02);
                        if (OC > 24) C03 = MMAD_FULL(S0, W3, C03);
                    }

                    wei += IC_BLOCK * OC_BLOCK;
                }
                src += IC_BLOCK * MB_BLOCK * IW * (1 + DH);
            }
            src += IC_BLOCK * MB_BLOCK * (IH * (1 + DD) - KH * (1 + DH)) * IW;
        }
        src += IC_BLOCK * MB_BLOCK * (ID - KD * (1 + DD)) * IH * IW;
    }

#if WITH_SRC_ZPOINTS
    const int has_pad_d = id < 0 || id + KD * (1 + DD) >= ID;
    const int has_pad_h = ih < 0 || ih + KH * (1 + DH) >= IH;
    const int has_pad_w = iw < 0 || iw + KW * (1 + DW) + OW_BLOCK * SW >= IW;

    if (has_pad_d || has_pad_h || has_pad_w) {
        wei -= IC_NCHUNK * KD * KH * KW * IC_BLOCK * OC_BLOCK;

        for (int ic_chunk = 0; ic_chunk < IC_NCHUNK; ic_chunk++) {
#if WITH_SRC_ZPOINTS_PER_IC
            const int4 z = read_src_zero_points_32c(
                    src_zpoints, (group_ic + ic_chunk) * IC_BLOCK);
#else
            const int z = read_src_zero_point(src_zpoints);
#endif // WITH_SRC_ZPOINTS_PER_IC
            for (int kd = 0; kd < KD; kd++) {
                for (int kh = 0; kh < KH; kh++) {
                    for (int kw = 0; kw < KW; kw++) {
                        int8 w0, w1, w2, w3;
                        BLOCK_READ_WHT_8x32(w0, 0);
                        BLOCK_READ_WHT_8x32(w1, 8 * IC_BLOCK);
                        BLOCK_READ_WHT_8x32(w2, 16 * IC_BLOCK);
                        BLOCK_READ_WHT_8x32(w3, 24 * IC_BLOCK);

                        int4 acc = 0;
#if WITH_SRC_ZPOINTS_PER_IC
                        acc.s0 += calc_src_compensation_x32(z, w0);
                        acc.s1 += calc_src_compensation_x32(z, w1);
                        acc.s2 += calc_src_compensation_x32(z, w2);
                        acc.s3 += calc_src_compensation_x32(z, w3);
#else
                        unroll_for(uint j = 0; j < 8; ++j) {
                            acc.s0 = idot4(0x01010101, w0[j], acc.s0);
                            acc.s1 = idot4(0x01010101, w1[j], acc.s1);
                            acc.s2 = idot4(0x01010101, w2[j], acc.s2);
                            acc.s3 = idot4(0x01010101, w3[j], acc.s3);
                        }
                        acc = z * acc;
#endif // WITH_SRC_ZPOINTS_PER_IC

                        for (int i = 0; i < OW_BLOCK; ++i) {
                            const int id0 = kd * (1 + DD) + id;
                            const int ih0 = kh * (1 + DH) + ih;
                            const int iw0 = kw * (1 + DW) + iw + i * SW;
                            const int is_pad_d = id0 < 0 || id0 >= ID;
                            const int is_pad_h = ih0 < 0 || ih0 >= IH;
                            const int is_pad_w = iw0 < 0 || iw0 >= IW;
                            if (is_pad_d || is_pad_h || is_pad_w) {
                                C00[i] += acc.s0;
                                C01[i] += acc.s1;
                                C02[i] += acc.s2;
                                C03[i] += acc.s3;
                            }
                        }

                        wei += IC_BLOCK * OC_BLOCK;
                    } // loop kw
                } // loop kh
            } // loop kd
        } // loop ic
    } // has_pad_d || has_pad_h || has_pad_w

    int4 src_comp = as_int4(intel_sub_group_block_read4(
            (__global uint *)(&src_compensation[(group_oc + oc) * OC_BLOCK])));

    C00 -= src_comp.s0;
    C01 -= src_comp.s1;
    C02 -= src_comp.s2;
    C03 -= src_comp.s3;
#endif // WITH_SRC_ZPOINTS

    if (ow < OW) {
        float4 tmp;

        DST_DATA4_T dst_pack[BLOCK];
        DST_DATA4_T D0[BLOCK];

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
#if OW_BLOCK == 4
        *(DST_DATA16_T *)D0 = BLOCK_READ_DST16(dst);
#endif
#if OW_BLOCK == 8
        *(DST_DATA16_T *)(D0 + 0) = BLOCK_READ_DST16(dst);
        *(DST_DATA16_T *)(D0 + 4) = BLOCK_READ_DST16(dst + 16 * 8);
#endif
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

#define PACK_DST(C0, C1, C2, C3, D) \
    do { \
        for (int n_i = 0; n_i < OW_BLOCK; n_i++) { \
            PACK(C0, C1, C2, C3, n_i); \
            QUANTIZE_ADD_BIAS(); \
            for (int didx = 0; didx < 4; ++didx) { \
                float tmp_i = tmp[didx]; \
                float d_i = convert_float(AS_SUM_DATA_T(D[n_i][didx])); \
                const int po_mb = group_mb * MB_BLOCK; \
                const int po_oc = (group_oc * OC_BLOCK + oc * OC_BLOCK \
                                          + ((didx * SUB_GROUP_SIZE) % OC) \
                                          + ocl_local_id % SUB_GROUP_SIZE) \
                        % (OC * G); \
                APPLY_POST_OPS(tmp_i, float, d_i, float, po_mb, 1, po_oc, 1, \
                        0, 1, 0, 1, 0, 1, 0, 1); \
                tmp[didx] = tmp_i; \
            } \
            ADD_DST_COMPENSATION(); \
            ZERO_PAD_DST(); \
            CONVERT_PACK(n_i); \
        } \
    } while (0)

        PACK_DST(C00, C01, C02, C03, D0);
#if OW_TAIL
        if (ow + OW_BLOCK > OW) {
            __attribute__((opencl_unroll_hint(OW_TAIL))) for (int i = 0;
                                                              i < OW_TAIL;
                                                              i++) {
                BLOCK_WRITE_DST4(&dst[i * 32], dst_pack[i]);
            }
        } else {
#endif
#if OW_BLOCK == 4
            BLOCK_WRITE_DST16(dst, *(DST_DATA16_T *)dst_pack);
#endif
#if OW_BLOCK == 8
            BLOCK_WRITE_DST16(dst, *(DST_DATA16_T *)dst_pack);
            BLOCK_WRITE_DST16(dst + 16 * 8, *(DST_DATA16_T *)(dst_pack + 4));
#endif
#if OW_TAIL
        }
#endif
    }
}
