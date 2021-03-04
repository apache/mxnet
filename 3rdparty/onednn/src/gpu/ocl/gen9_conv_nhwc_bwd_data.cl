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

#include "gpu/ocl/ocl_types.h"

#if ID > 1
#define CASE_3D 1
#else
#define CASE_3D 0
#endif

#define TRANSPOSE_1(_block, _col) \
    (DATA_T)(intel_sub_group_shuffle(_block, _col))

#define FMA1(a, b, c) fma((DATA_T)(a), (DATA_T)b, (DATA_T)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, _blockB1) \
    { \
        _result = FMA1(_blockB.s0, TRANSPOSE_1(_blockA, 0), _result); \
        _result = FMA1(_blockB.s1, TRANSPOSE_1(_blockA, 1), _result); \
        _result = FMA1(_blockB.s2, TRANSPOSE_1(_blockA, 2), _result); \
        _result = FMA1(_blockB.s3, TRANSPOSE_1(_blockA, 3), _result); \
        _result = FMA1(_blockB.s4, TRANSPOSE_1(_blockA, 4), _result); \
        _result = FMA1(_blockB.s5, TRANSPOSE_1(_blockA, 5), _result); \
        _result = FMA1(_blockB.s6, TRANSPOSE_1(_blockA, 6), _result); \
        _result = FMA1(_blockB.s7, TRANSPOSE_1(_blockA, 7), _result); \
        _result = FMA1(_blockB1.s0, TRANSPOSE_1(_blockA, 8), _result); \
        _result = FMA1(_blockB1.s1, TRANSPOSE_1(_blockA, 9), _result); \
        _result = FMA1(_blockB1.s2, TRANSPOSE_1(_blockA, 10), _result); \
        _result = FMA1(_blockB1.s3, TRANSPOSE_1(_blockA, 11), _result); \
        _result = FMA1(_blockB1.s4, TRANSPOSE_1(_blockA, 12), _result); \
        _result = FMA1(_blockB1.s5, TRANSPOSE_1(_blockA, 13), _result); \
        _result = FMA1(_blockB1.s6, TRANSPOSE_1(_blockA, 14), _result); \
        _result = FMA1(_blockB1.s7, TRANSPOSE_1(_blockA, 15), _result); \
    }

inline DATA_T read_oc_block(const __global DATA_T *ptr, int off) {
    const int local_id = get_sub_group_local_id();
#if OC_WO_PADDING % OC_BLOCK != 0
    int tail = OC_WO_PADDING - off;
    if (tail < OC_BLOCK) {
        return (local_id < tail) ? AS_DATA_T(ptr[local_id]) : DATA_ZERO;
    }
#endif
    if ((OC_WO_PADDING * sizeof(DATA_T)) % 4 != 0)
        return ptr[local_id];
    else
        return AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)(ptr)));
}

inline void write_ic_block(__global DATA_T *ptr, int off, DATA_T value) {
    const int local_id = get_sub_group_local_id();
#if IC_WO_PADDING % IC_BLOCK != 0
    int tail = IC_WO_PADDING - off;
    if (tail < IC_BLOCK) {
        if (local_id < tail) ptr[local_id] = value;
        return;
    }
#endif
    if ((IC_WO_PADDING * sizeof(DATA_T)) % 16 != 0)
        ptr[local_id] = value;
    else
        BLOCK_WRITE((__global BLOCK_DATA_T *)ptr, AS_BLOCK_DATA_T(value));
    return;
}

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
__kernel void
gen9_conv_nhwc_bwd_data(__global DATA_T *diff_src, __global DATA_T *wei,
        __global DATA_T *diff_dst, __global DATA_T *bias) {
    const int sp = get_group_id(1);
    const int local_id = get_sub_group_local_id();
    const int icb_mb = get_group_id(2);
    const int mb = icb_mb / (G * IC_PADDED / ICB);
    const int icb = icb_mb % (G * IC_PADDED / ICB);
    const int ic = (icb * ICB) / IC_BLOCK + get_group_id(0);

    const int g = ic / (IC_PADDED / IC_BLOCK);
    const int gic = ic % (IC_PADDED / IC_BLOCK);

#if CASE_3D
    const int id = sp / (IWB * IH);
    const int ihw = sp % (IWB * IH);
#else
    const int id = 0;
    const int ihw = sp;
#endif
    const int ih = ihw / IWB;
    const int iw = (ihw % IWB) * IW_BLOCK;

    diff_dst += mb * OC_WO_PADDING * G * OD * OH * OW + g * OC_WO_PADDING;
    DATA_T blockC00[IW_BLOCK] = {DATA_ZERO};

    if (WITH_BIAS) {
        const int bg_off = g * IC;
        const int bc_off = gic * IC_BLOCK + local_id;
        DATA_T b = (IC_WO_PADDING % IC_BLOCK == 0 || bc_off < IC_WO_PADDING)
                ? bias[bg_off + bc_off]
                : DATA_ZERO;
        unroll_for(int i = 0; i < IW_BLOCK; ++i) { blockC00[i] = b; }
    }
    wei += gic * KD * KH * KW * OC_BLOCK * IC_BLOCK
            + g * IC_PADDED * OC_PADDED * KD * KH * KW;
#if KH != 1 || KW != 1 || KD != 1
    for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw) {

                if (ih + PH < kh * (1 + DH)) continue;
#if CASE_3D
                if (id + PD < kd * (1 + DD)) continue;
                int od = id - kd * (1 + DD) + PD;
                if (od % SD != 0) continue;
                od /= SD;
                if (od >= OD) continue;
#endif

                int oh = ih - kh * (1 + DH) + PH;
                if (oh % SH != 0) continue;
                oh /= SH;
                if (oh >= OH) continue;

                const __global DATA_T *diff_dst1
                        = diff_dst + oh * OW * G * OC_WO_PADDING;
#if CASE_3D
                diff_dst1 += od * OH * OW * G * OC_WO_PADDING;
#endif
                const __global DATA_T *wei1 = wei
#if CASE_3D
                        + kd * KH * KW * OC_BLOCK * IC_BLOCK
#endif
                        + kh * KW * OC_BLOCK * IC_BLOCK
                        + kw * OC_BLOCK * IC_BLOCK;
#else
    int oh = (ih + PH);
#if CASE_3D
    int od = (id + PD);
#endif
    bool do_ker = true;
#if SW != 1 || SH != 1 || SD != 1
    do_ker = oh % SH == 0;
    oh /= SH;
#if CASE_3D
    do_ker = do_ker && od % SD == 0;
    od /= SD;
#endif
#endif
#if PH != 0 || PW != 0 || PD != 0
    do_ker = do_ker && (oh < OH);
#if CASE_3D
    do_ker = do_ker && (od < OD);
#endif
#endif
#if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
    if (do_ker) {
#endif
        const __global DATA_T *diff_dst1
                = diff_dst + oh * OW * G * OC_WO_PADDING;
#if CASE_3D
        diff_dst1 += od * OH * OW * G * OC_WO_PADDING;
#endif
        const __global DATA_T *wei1 = wei;
#endif
                int ocb = 0;
                do {
                    DATA8_T blockB00 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)wei1));
                    DATA8_T blockB01 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)(wei1
                                    + 8 * IC_BLOCK)));
                    DATA_T blockA[IW_BLOCK];
                    __attribute__((
                            opencl_unroll_hint(IW_BLOCK))) // attr:no-format
                    for (int i = 0; i < IW_BLOCK; i++) {
#if KW != 1
                        if (iw + i + PW < kw * (1 + DW)) {
                            blockA[i] = 0.0;
                            continue;
                        }
                        int ow = iw + i - kw * (1 + DW) + PW;
#else
                int ow = iw + i + PW;
#endif
#if SW != 1
                        if (ow % SW != 0) {
                            blockA[i] = 0.0;
                            continue;
                        }
                        ow /= SW;
#endif
                        if (ow >= OW) {
                            blockA[i] = 0.0;
                            continue;
                        }
                        blockA[i] = read_oc_block(
                                &diff_dst1[ow * G * OC_WO_PADDING], ocb);
                    }
                    __attribute__((
                            opencl_unroll_hint(IW_BLOCK))) // attr:no-format
                    for (int i = 0; i < IW_BLOCK; i++) {
                        MULTIPLY_BLOCKS_8x8(
                                blockC00[i], blockA[i], blockB00, blockB01);
                    }
                    diff_dst1 += OC_BLOCK;
                    wei1 += KD * KH * KW * OC_BLOCK * IC_PADDED;
                    ocb += OC_BLOCK;
                } while (ocb < OC);

#if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
            }
#else
#if KH != 1 || KW != 1 || KD != 1
    }
#endif
#endif

    __global DATA_T *src_write0 = diff_src
            + mb * IC_WO_PADDING * G * ID * IH * IW
            + id * IH * IW * G * IC_WO_PADDING + ih * IW * G * IC_WO_PADDING
            + iw * G * IC_WO_PADDING + g * IC_WO_PADDING + gic * IC_BLOCK;

    for (int i = 0; i < min(IW_BLOCK, IW - iw); i++) {
        write_ic_block(&src_write0[i * G * IC_WO_PADDING], gic * IC_BLOCK,
                blockC00[i]);
    }
}
#undef TRANSPOSE_BLOCK_8
#undef MULTIPLY_BLOCKS_8x8
