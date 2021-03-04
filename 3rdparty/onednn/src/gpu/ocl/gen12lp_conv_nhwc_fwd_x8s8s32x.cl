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

#if IC % IC_BLOCK != 0
#define IC_NBLOCKS_TAIL ((IC % IC_BLOCK + 3) / 4)
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
#elif OW_BLOCK == 8
#define BLOCK 8
#define ACC_DATA_BLOCK int8
#define SRC_DATA_BLOCK_T MMAD_DATA8_T
#define READ_BLOCK intel_sub_group_block_read8
#define WRITE_LOCAL block_write8

DECLARE_MMAD(
        mmad_tail, IC_NBLOCKS_TAIL, 8, SRC_DATA_BLOCK_T, int8, ACC_DATA_BLOCK)

#define MMAD_FULL mmad8x8
#define MMAD_TAIL mmad_tail
#else
#error "Wrong OW_BLOCK"
#endif

#define BLOCK_READ_SRC(data, idx) \
    data = intel_sub_group_block_read8((__global uint *)&src[idx]);

#define BLOCK_READ_WHT_1x32(data, idx) \
    data = as_int(intel_sub_group_block_read((__global uint *)&wei[idx]));

#define BLOCK_READ_WHT_8x32(data, idx) \
    data = as_int8(intel_sub_group_block_read8((__global uint *)&wei[idx]));

#define BLOCK_READ_BIA(data, idx) \
    data = as_float4(intel_sub_group_block_read4((__global uint *)&bias[idx]));

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

#define BLOCK_READ_BOUND 1
#define BLOCK_WRITE_BOUND 4

inline float4 read_bias_scale_block4(const __global float *dst, int off) {
    const int local_id = get_sub_group_local_id();
#if OC % OC_BLOCK != 0
    int tail = OC - off;
    if (tail < OC_BLOCK) {
        return (float4)(local_id < tail - 8 * 0 ? dst[0 * 8 + local_id] : 0,
                local_id < tail - 8 * 1 ? dst[1 * 8 + local_id] : 0,
                local_id < tail - 8 * 2 ? dst[2 * 8 + local_id] : 0,
                local_id < tail - 8 * 3 ? dst[3 * 8 + local_id] : 0);
    }
#endif
#if OC % BLOCK_READ_BOUND != 0
    return (float4)(dst[0 * 8 + local_id], dst[1 * 8 + local_id],
            dst[2 * 8 + local_id], dst[3 * 8 + local_id]);
#else
    return as_float4(intel_sub_group_block_read4((__global uint *)dst));
#endif
}

inline DST_DATA4_T read_oc_block4(const __global DATA_T *dst, int off) {
    const int local_id = get_sub_group_local_id();
#if OC % OC_BLOCK != 0
    int tail = OC - off;
    if (tail < OC_BLOCK) {
        return (DST_DATA4_T)(
                local_id < tail - 8 * 0 ? dst[0 * 8 + local_id] : 0,
                local_id < tail - 8 * 1 ? dst[1 * 8 + local_id] : 0,
                local_id < tail - 8 * 2 ? dst[2 * 8 + local_id] : 0,
                local_id < tail - 8 * 3 ? dst[3 * 8 + local_id] : 0);
    }
#endif
#if OC % BLOCK_READ_BOUND != 0
    return (DST_DATA4_T)(dst[0 * 8 + local_id], dst[1 * 8 + local_id],
            dst[2 * 8 + local_id], dst[3 * 8 + local_id]);
#else
    return BLOCK_READ_DST4(dst);
#endif
}

inline void write_oc_block4(__global DATA_T *dst, int off, DATA4_T value) {
    const int local_id = get_sub_group_local_id();
#if OC % OC_BLOCK != 0
    int tail = OC - off;
    if (tail < OC_BLOCK) {
        if (local_id < tail) dst[0 * 8 + local_id] = value.s0;
        if (local_id < tail - 8 * 1) dst[1 * 8 + local_id] = value.s1;
        if (local_id < tail - 8 * 2) dst[2 * 8 + local_id] = value.s2;
        if (local_id < tail - 8 * 3) dst[3 * 8 + local_id] = value.s3;
        return;
    }
#endif
#if OC % BLOCK_WRITE_BOUND != 0
    dst[0 * 8 + local_id] = value.s0;
    dst[1 * 8 + local_id] = value.s1;
    dst[2 * 8 + local_id] = value.s2;
    dst[3 * 8 + local_id] = value.s3;
    return;
#else
    BLOCK_WRITE_DST4(dst, value);
    return;
#endif
}

inline void write_local_1(__local uint *S, __global SRC_DATA_T *src1) {
    const int local_id = get_sub_group_local_id();
#if IC % 4 != 0
    __local SRC_DATA_T *S1 = S + local_id;
    __global SRC_DATA_T *src2 = (const __global uint *)(src1) + local_id;
    S1[0] = src2[0];
    S1[1] = src2[1];
    S1[2] = src2[2];
    S1[3] = src2[3];
#else
    block_write(S, intel_sub_group_block_read((const __global uint *)(src1)));
#endif
    return;
}

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
__kernel void
conv_nhwc_fwd_x8s8s32x(const __global SRC_DATA_T *src, const __global char *wei,
        const __global float *bias, __global DATA_T *dst POST_OP_ARGS,
        float scale, const __global float *scales_per_oc,
        const __global int *src_compensation, const __global int *src_zpoints,
        const __global int *dst_compensation) {
    const int group_oc = get_group_id(0) * OC_GROUP;
    const int group_mb = get_group_id(2) * MB_GROUP;
    const int group_sp = get_group_id(1) * SP_GROUP;
    const int sub_group_id = get_sub_group_id();
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

    const int local_id = get_sub_group_local_id();

    __local uint S_slice[SRC_SLM_SIZE];
    __local uint *S_part = S_slice + IC_BLOCK / 4 * (sp * SW * OW_BLOCK + PW);
    __local uint *S_work = S_slice + IC_BLOCK / 4 * (sp * SW * OW_BLOCK);

    const bool left_tail = iw < 0;
    const bool left_nozero_tail = sub_group_id == 0 && iw >= 0;
    const bool right_tail = (iw + PW + OW_SLM_TAIL >= IW) && (iw + PW < IW);
    const bool empty = (iw + PW >= IW);
    const bool right_nozero_tail
            = sp == (LWS_1 - 1) && (iw + PW + OW_SLM_TAIL < IW);

    dst += group_mb * MB_BLOCK * OD * OH * OW * G * OC;
    dst += (OW * OH * od + OW * oh + ow) * G * OC;
    dst += OC_BLOCK * (group_oc + oc);

    src += group_mb * MB_BLOCK * ID * IH * IW * G * IC;
    src += (IW * IH * id + IW * ih + iw + PW) * G * IC;
    src += group_ic * IC_BLOCK;

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

    __attribute__((opencl_unroll_hint(1))) // attr:no-format
    for (int ic_chunk = 0; ic_chunk < IC_NCHUNK; ic_chunk++) {
        SRC_DATA_BLOCK_T S0;

        __attribute__((opencl_unroll_hint(1))) // attr:no-format
        for (int kd = 0; kd < KD; kd++) {
            if (kd * (1 + DD) + id < 0 || kd * (1 + DD) + id >= ID) {
                wei += IC_BLOCK * OC_BLOCK * KH * KW;
                continue;
            }
            __attribute__((opencl_unroll_hint(1))) // attr:no-format
            for (int kh = 0; kh < KH; kh++) {
                if (kh * (1 + DH) + ih < 0 || kh * (1 + DH) + ih >= IH) {
                    wei += IC_BLOCK * OC_BLOCK * KW;
                    continue;
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                const __global SRC_DATA_T *src1 = src
                        + kd * (1 + DD) * IH * IW * G * IC
                        + kh * (1 + DH) * IW * G * IC;

#if SLM_WORKING_GROUPS < OW_NCHUNK
                if (iw + PW < IW) {
#endif
#if OW_NCHUNK > LWS_1
                    /* Copy tails in case of multigroups */
                    if (ow < OW) {
#if PW > 0
                        if (left_nozero_tail) {
                            for (int i = -PW; i < 0; i++) {
                                write_local_1(
                                        S_part + i * 8, src1 + i * G * IC);
                            }
                        }
#endif

                        if (right_nozero_tail) {
                            for (int i = SW * OW_BLOCK; i
                                    < SW * OW_BLOCK + (KW - 1) * (1 + DW) - PW;
                                    i++) {
                                write_local_1(
                                        S_part + i * 8, src1 + i * G * IC);
                            }
                        }
#endif

#if OW_SLM_TAIL != OW_BLOCK * SW
                        /* Copy last block to SLM */
                        if (right_tail) {
                            __attribute__((
                                    opencl_unroll_hint)) // attr:no-format
                            for (int i = 0; i < OW_SLM_TAIL; i++) {
                                write_local_1(
                                        S_part + i * 8, src1 + i * G * IC);
                            }
                        } else {
#endif
                            __attribute__((
                                    opencl_unroll_hint)) // attr:no-format
                            for (int i = 0; i < SW * OW_BLOCK; i++) {
                                write_local_1(
                                        S_part + i * 8, src1 + i * G * IC);
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

                __attribute__((opencl_unroll_hint)) // attr:no-format
                for (int kw = 0; kw < KW; kw++) {
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
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
                } // kw loop
            } //kh loop
        } //kd loop
        src += IC_BLOCK;
    } // ic_chunk loop

    if (ow < OW) {
        float4 tmp = {0, 0, 0, 0};

        DST_DATA4_T dst_pack[BLOCK] = {0};
        DST_DATA4_T D0[BLOCK] = {0};

#if SCALES_PER_OC
        float4 scales = {0, 0, 0, 0};
        scales = read_bias_scale_block4(
                scales_per_oc + (group_oc + oc) * OC_BLOCK,
                (group_oc + oc) * OC_BLOCK);
#endif

#if WITH_BIAS
        float4 bia = {0, 0, 0, 0};
        bia = read_bias_scale_block4(
                bias + (group_oc + oc) * OC_BLOCK, (group_oc + oc) * OC_BLOCK);
        bia *= SCALE;
#define QUANTIZE_ADD_BIAS() tmp = fma(tmp, (float4)SCALE, bia);
#else
#define QUANTIZE_ADD_BIAS() tmp *= SCALE;
#endif

#if WITH_SUM
#if OW_BLOCK >= 4
        *(DST_DATA16_T *)D0 = (DST_DATA16_T)(
                read_oc_block4(dst + G * OC * 0, (group_oc + oc) * OC_BLOCK),
                read_oc_block4(dst + G * OC * 1, (group_oc + oc) * OC_BLOCK),
                read_oc_block4(dst + G * OC * 2, (group_oc + oc) * OC_BLOCK),
                read_oc_block4(dst + G * OC * 3, (group_oc + oc) * OC_BLOCK));
#endif
#if OW_BLOCK == 8
        *(DST_DATA16_T *)(D0 + 4) = (DST_DATA16_T)(
                read_oc_block4(dst + G * OC * 4, (group_oc + oc) * OC_BLOCK),
                read_oc_block4(dst + G * OC * 5, (group_oc + oc) * OC_BLOCK),
                read_oc_block4(dst + G * OC * 6, (group_oc + oc) * OC_BLOCK),
                read_oc_block4(dst + G * OC * 7, (group_oc + oc) * OC_BLOCK));
#endif
#endif // with_sum

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
                float dni_i = convert_float(AS_SUM_DATA_T(D[n_i][didx])); \
                int po_mb; \
                po_mb = group_mb % MB; \
                const int po_oc = ((group_oc + oc) * OC_BLOCK + local_id \
                                          + didx * SUB_GROUP_SIZE) \
                        % (OC * G); \
                APPLY_POST_OPS(tmp_i, float, dni_i, float, po_mb, 1, po_oc, 1, \
                        0, 1, 0, 1, 0, 1, 0, 1); \
                tmp[didx] = tmp_i; \
            } \
            CONVERT_PACK(n_i); \
        } \
    } while (0)

        PACK_DST(C00, C01, C02, C03, D0);
#if OW_TAIL
        if (ow + OW_BLOCK > OW) {
            __attribute__((opencl_unroll_hint(OW_TAIL))) // attr:no-format
            for (int i = 0; i < OW_TAIL; i++) {
                write_oc_block4(dst + i * G * OC, (group_oc + oc) * OC_BLOCK,
                        dst_pack[i]);
            }
        } else {
#endif

#if OW_BLOCK >= 4
            write_oc_block4(
                    dst + G * OC * 0, (group_oc + oc) * OC_BLOCK, dst_pack[0]);
            write_oc_block4(
                    dst + G * OC * 1, (group_oc + oc) * OC_BLOCK, dst_pack[1]);
            write_oc_block4(
                    dst + G * OC * 2, (group_oc + oc) * OC_BLOCK, dst_pack[2]);
            write_oc_block4(
                    dst + G * OC * 3, (group_oc + oc) * OC_BLOCK, dst_pack[3]);
#endif
#if OW_BLOCK == 8
            write_oc_block4(
                    dst + G * OC * 4, (group_oc + oc) * OC_BLOCK, dst_pack[4]);
            write_oc_block4(
                    dst + G * OC * 5, (group_oc + oc) * OC_BLOCK, dst_pack[5]);
            write_oc_block4(
                    dst + G * OC * 6, (group_oc + oc) * OC_BLOCK, dst_pack[6]);
            write_oc_block4(
                    dst + G * OC * 7, (group_oc + oc) * OC_BLOCK, dst_pack[7]);
#endif
#if OW_TAIL
        }
#endif
    }
}
