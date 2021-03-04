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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

// FWD fp32 anf fp16 Convolution Kernel with NHWC data layout support
// Features:
// - based on 8ow16c version of blocked implementation
// - weights are blocked and padded: OIhw16i16o
// - explicit tail processing for src and dst channels
// - due to 16-bytes alignment requred, intel_sub_groups_block_write usage
//   is limited

#define _BLOCK_READ8(ptr) \
    AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)(ptr)))
#define _BLOCK_READ4(ptr) \
    AS_DATA4_T(BLOCK_READ4((const __global BLOCK_DATA_T *)(ptr)))
#define _BLOCK_READ2(ptr) \
    AS_DATA2_T(BLOCK_READ2((const __global BLOCK_DATA_T *)(ptr)))
#define _BLOCK_READ(ptr) \
    AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)(ptr)))

#define _BLOCK_WRITE8(ptr, v) \
    BLOCK_WRITE8((__global BLOCK_DATA_T *)(ptr), AS_BLOCK_DATA8_T(v))
#define _BLOCK_WRITE4(ptr, v) \
    BLOCK_WRITE4((__global BLOCK_DATA_T *)(ptr), AS_BLOCK_DATA4_T(v))
#define _BLOCK_WRITE2(ptr, v) \
    BLOCK_WRITE2((__global BLOCK_DATA_T *)(ptr), AS_BLOCK_DATA2_T(v))
#define _BLOCK_WRITE(ptr, v) \
    BLOCK_WRITE((__global BLOCK_DATA_T *)(ptr), AS_BLOCK_DATA_T(v))
#define ENABLE_KW_BUF (KW >= 5)

#define IS_3D (OD > 1)
#define KDHW_SIZE (KD * KH * KW)
#define HAS_PAD_D (PD > 0 || (OD - 1) * SD - PD + (KD - 1) * (1 + DD) >= ID)
#define HAS_PAD_H (PH > 0 || (OH - 1) * SH - PH + (KH - 1) * (1 + DH) >= IH)
#define HAS_PAD_W (PW > 0 || (OW - 1) * SW - PW + (KW - 1) * (1 + DW) >= IW)
#define OC_PAD_BLOCK (OC % OC_BLOCK ? (OC / OC_BLOCK + 1) * OC_BLOCK : OC)

#if DT_F32
#define BLOCK_READ_BOUND 1
#define BLOCK_WRITE_BOUND 4
#elif DT_F16
#define BLOCK_READ_BOUND 2
#define BLOCK_WRITE_BOUND 8
#else
#error "Wrong Data Type"
#endif

inline DATA_T read_ic_block(const __global DATA_T *ptr, int off) {
    const int local_id = get_local_id(0);
#if IC == 3
    return (local_id < IC) ? *ptr : 0;
#else
#if (IS_DW ? G_WO_PADDING : IC_WO_PADDING) % IC_BLOCK != 0
    int tail = (IS_DW ? G_WO_PADDING : IC_WO_PADDING) - off;
    if (tail < IC_BLOCK) { return (local_id < tail) ? ptr[local_id] : 0; }
#endif
#if (IS_DW ? G_WO_PADDING : IC_WO_PADDING) % BLOCK_READ_BOUND != 0
    return ptr[local_id];
#else
    return _BLOCK_READ(ptr);
#endif
#endif
}

inline DATA_T read_oc_block(const __global DATA_T *ptr, int off) {
    const int local_id = get_local_id(0);
#if (IS_DW ? G_WO_PADDING : OC_WO_PADDING) % OC_BLOCK != 0
    int tail = (IS_DW ? G_WO_PADDING : OC_WO_PADDING) - off;
    if (tail < OC_BLOCK) { return (local_id < tail) ? ptr[local_id] : 0; }
#endif
#if (IS_DW ? G_WO_PADDING : OC_WO_PADDING) % BLOCK_READ_BOUND != 0
    return ptr[local_id];
#else
    return _BLOCK_READ(ptr);
#endif
}

inline void write_oc_block(__global DATA_T *ptr, int off, DATA_T value) {
    const int local_id = get_local_id(0);
#if (IS_DW ? G_WO_PADDING : OC_WO_PADDING) % OC_BLOCK != 0
    int tail = (IS_DW ? G_WO_PADDING : OC_WO_PADDING) - off;
    if (tail < OC_BLOCK) {
        if (local_id < tail) ptr[local_id] = value;
        return;
    }
#endif
#if (IS_DW ? G_WO_PADDING : OC_WO_PADDING) % BLOCK_WRITE_BOUND != 0
    ptr[local_id] = value;
    return;
#else
    return _BLOCK_WRITE(ptr, value);
#endif
}

void multiply_blocks_8x8_ic3(DATA_T *res, DATA_T blockA, const DATA_T *blockB) {
    *res = fma(blockB[0], intel_sub_group_shuffle(blockA, 0), *res);
    *res = fma(blockB[1], intel_sub_group_shuffle(blockA, 1), *res);
    *res = fma(blockB[2], intel_sub_group_shuffle(blockA, 2), *res);
}

void multiply_blocks_8x8(
        DATA_T *res, DATA_T blockA, DATA8_T blockB0, DATA8_T blockB1) {
    for (int i = 0; i < 8; i++) {
        *res = fma(blockB0[i], intel_sub_group_shuffle(blockA, i), *res);
    }
    for (int i = 0; i < 8; i++) {
        *res = fma(blockB1[i], intel_sub_group_shuffle(blockA, 8 + i), *res);
    }
}

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
gen9_conv_nhwc_fwd(const __global DATA_T *src, const __global DATA_T *wei,
        const __global DATA_T *bias, __global DATA_T *dst POST_OP_ARGS) {

    const int sp = get_group_id(1);
    const int local_id = get_sub_group_local_id();
    const int ocb_mb = get_group_id(2);
    const int ocb = ocb_mb / (MB);
    const int mb = ocb_mb % (MB);

#if IS_DW
    const int oc = get_group_id(0);
    const int g = 0;
    const int goc = oc;
#else
    const int oc = (ocb * OCB) / OC_BLOCK + get_group_id(0);
    const int g = oc / (OC_PAD_BLOCK / OC_BLOCK);
    const int goc = oc % (OC_PAD_BLOCK / OC_BLOCK);
#endif

    const int od = IS_3D ? sp / (OWB * OHB) : 0;
    const int ohw = IS_3D ? sp % (OWB * OHB) : sp;
    const int id = IS_3D ? od * SD - PD : 0;
    const int oh = (ohw / OWB) * OH_BLOCK;
    const int ow = (ohw % OWB) * OW_BLOCK;

    DATA_T blockC00[OW_BLOCK] = {0};
    if (WITH_BIAS) {
        const int bc_off = oc * OC_BLOCK + local_id;
        DATA_T b = (OC_WO_PADDING % OC_BLOCK == 0 || bc_off < OC_WO_PADDING)
                ? bias[bc_off]
                : DATA_ZERO;
        unroll_for(int i = 0; i < OW_BLOCK; i++) { blockC00[i] = b; }
    }

    int ih = oh * SH - PH;
    int iw = ow * SW - PW;

    src += mb * ID * IH * IW * G * IC_WO_PADDING;
    src += (id * IH * IW + ih * IW + iw) * G * IC_WO_PADDING;
    src += g * IC_WO_PADDING;
    src += (IS_DW ? oc * OC_BLOCK : 0);

    wei += goc * KDHW_SIZE * OC_BLOCK * IC + g * IC * OC_PAD_BLOCK * KDHW_SIZE;

#if (KD == 1 && KH == 1) && (HAS_PAD_D || HAS_PAD_H)
    const bool dh_out_of_range = (id < 0 || id >= ID || ih < 0 || ih >= IH);
#else
    const bool dh_out_of_range = false;
#endif

#if IS_DW
    const int icb_min = goc * OC_BLOCK;
    const int icb_max = icb_min + OC_BLOCK;
#else
    const int icb_min = 0;
    const int icb_max = dh_out_of_range ? 0 : (IC == 3 ? 1 : IC);
#endif

    for (int icb = icb_min; icb < icb_max; icb += IC_BLOCK) {
        __attribute__((opencl_unroll_hint(1))) // attr:no-format
        for (int kd = 0; kd < KD; ++kd) {
#if HAS_PAD_D
            if (id + kd * (1 + DD) < 0 || id + kd * (1 + DD) >= ID) continue;
#endif
            __attribute__((opencl_unroll_hint(1))) // attr:no-format
            for (int kh = 0; kh < KH; ++kh) {
#if HAS_PAD_H
                if (ih + kh * (1 + DH) < 0 || ih + kh * (1 + DH) >= IH)
                    continue;
#endif
                const __global DATA_T *src1 = src
                        + kd * (1 + DD) * IH * IW * G * IC_WO_PADDING
                        + kh * (1 + DH) * IW * G * IC_WO_PADDING;
                if (IC == 3) src1 += local_id;
#if ENABLE_KW_BUF
                DATA_T tempA[SW * OW_BLOCK + KW * (1 + DW)] = {0};
                __attribute__((opencl_unroll_hint(
                        SW * OW_BLOCK + KW * (1 + DW)))) // attr:no-format
                for (int i = 0; i < SW * OW_BLOCK + KW * (1 + DW); i++) {
                    if ((i + iw) >= 0 && (i + iw) < IW) {
                        tempA[i] = read_ic_block(
                                &src1[i * G * IC_WO_PADDING], icb);
                    }
                }
#endif
                __attribute__((opencl_unroll_hint(KW))) // attr:no-format
                for (int kw = 0; kw < KW; ++kw) {
#if IC == 3
                    const __global DATA_T *wei1 = wei
                            + (kd * KH * KW + kh * KW + kw) * IC * OC_BLOCK;
#elif IS_DW
                    const __global DATA_T *wei1
                            = wei + (kd * KH * KW + kh * KW + kw) * OC_BLOCK;
#else
                    const __global DATA_T *wei1 = wei
                            + (kd * KH * KW + kh * KW + kw) * IC_BLOCK
                                    * OC_BLOCK;
#endif

                    DATA_T blockA[OW_BLOCK] = {0};
#if ENABLE_KW_BUF
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockA[i] = tempA[i * SW + kw * (1 + DW)];
                    }
#else
                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        int iw_off = i * SW + kw * (1 + DW);
                        if (iw + iw_off >= 0 && iw + iw_off < IW) {
                            blockA[i] = read_ic_block(
                                    &src1[iw_off * G * IC_WO_PADDING], icb);
                        }
                    }
#endif

#if IC == 3
                    DATA_T blockB[IC];
                    __attribute__((opencl_unroll_hint(IC))) // attr:no-format
                    for (int i = 0; i < IC; i++) {
                        blockB[i] = _BLOCK_READ(wei1 + i * OC_BLOCK);
                    }

                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        multiply_blocks_8x8_ic3(
                                &blockC00[i], blockA[i], blockB);
                    }
#elif IS_DW
                    DATA_T blockB = _BLOCK_READ(wei1);
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockC00[i] = fma(blockA[i], blockB, blockC00[i]);
                    }
#else
                    DATA8_T blockB00 = _BLOCK_READ8(wei1);
                    DATA8_T blockB01 = _BLOCK_READ8(wei1 + 8 * OC_BLOCK);

                    __attribute__((
                            opencl_unroll_hint(OW_BLOCK))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        multiply_blocks_8x8(
                                &blockC00[i], blockA[i], blockB00, blockB01);
                    }
#endif
                }
            }
        }
        src += IC_BLOCK;
        wei += KDHW_SIZE * IC_BLOCK * OC_BLOCK;
    }

    __global DATA_T *dst_write0 = dst + mb * OD * OH * OW * G * OC_WO_PADDING;
    dst_write0 += (od * OH * OW + oh * OW + ow) * G * OC_WO_PADDING;
    dst_write0 += g * OC_WO_PADDING + goc * OC_BLOCK;

    // Apply postops
    DATA_T blockS00[OW_BLOCK];
#if WITH_SUM
    for (int i = 0; i < min(OW_BLOCK, OW - ow); i++) {
        blockS00[i] = read_oc_block(
                &dst_write0[i * G * OC_WO_PADDING], goc * OC_BLOCK);
    }

#endif // WITH_SUM

    for (int didx = 0; didx < OW_BLOCK; ++didx) {
        DATA_T accum = blockC00[didx];
        DATA_T sum = blockS00[didx];
        const int po_mb = (mb) % MB;
        const int po_oc = (oc * OC_BLOCK + local_id) % (OC * G);
        APPLY_POST_OPS(accum, DATA_T, sum, DATA_T, po_mb, 1, po_oc, 1, od, 1,
                oh, 1, ow, 1, 0, 1);
        blockC00[didx] = accum;
    }

    // Save
    for (int i = 0; i < min(OW_BLOCK, OW - ow); i++) {
        write_oc_block(&dst_write0[i * G * OC_WO_PADDING], goc * OC_BLOCK,
                blockC00[i]);
    }
}
