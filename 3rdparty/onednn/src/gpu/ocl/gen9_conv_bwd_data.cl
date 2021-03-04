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

#include "gpu/ocl/ocl_types.h"

#if ID > 1
#define CASE_3D 1
#else
#define CASE_3D 0
#endif

#if BWD_DATA == 1

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if VER_16MB16C == 1 || VER_8OW16C == 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
#endif
__kernel void
gen9_conv_bwd_data(__global DATA_T *diff_src, __global DATA_T *wei,
        __global DATA_T *diff_dst, __global DATA_T *bias) {

#if VER_16MB16C == 1
    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);

    const int icb_mb = get_group_id(2);
    const int mb = icb_mb / (G * IC / ICB) * MB_BLOCK;
    const int icb = icb_mb % (G * IC / ICB);
    const int ic = (icb * ICB) / IC_BLOCK + get_group_id(0);

    const int g = ic / (IC / IC_BLOCK);
    const int gic = ic % (IC / IC_BLOCK);

#if CASE_3D
    const int id = sp / (IW * IH);
    const int ihw = sp % (IW * IH);
#else
    const int id = 0;
    const int ihw = sp;
#endif
    const int ih = ihw / IW;
    const int iw = ihw % IW;

    diff_dst += mb * OC * G * OD * OH * OW + g * OC * OD * OH * OW * MB_BLOCK;

    DATA8_T blockC00 = (DATA8_T)DATA_ZERO;
    DATA8_T blockC01 = (DATA8_T)DATA_ZERO;

    if (WITH_BIAS) {
#if IS_DW
        const int bg_off = g * IC + local_id;
        const int bc_off = gic * IC_BLOCK;
        DATA_T b = (G_WO_PADDING % IC_BLOCK == 0 || bg_off < G_WO_PADDING)
                ? bias[bg_off + bc_off]
                : DATA_ZERO;
#else
        const int bg_off = g * IC;
        const int bc_off = gic * IC_BLOCK + local_id;
        DATA_T b = (IC_WO_PADDING % IC_BLOCK == 0 || bc_off < IC_WO_PADDING)
                ? bias[bg_off + bc_off]
                : DATA_ZERO;
#endif
        unroll_for(int i = 0; i < 8; ++i) {
            blockC00[i] = b;
            blockC01[i] = b;
        }
    }

    wei += gic * KD * KH * KW * OC_BLOCK * IC_BLOCK
            + g * IC * OC * KD * KH * KW;
    int ocb = 0;
    do {
#if KH != 1 || KW != 1 || KD != 1
        for (int kd = 0; kd < KD; ++kd)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {

                    if (iw + PW < kw * (1 + DW) || ih + PH < kh * (1 + DH))
                        continue;
#if CASE_3D
                    if (id + PD < kd * (1 + DD)) continue;
                    int od = id - kd * (1 + DD) + PD;
                    if (od % SD != 0) continue;
                    od /= SD;
                    if (od >= OD) continue;
#endif

                    int ow = iw - kw * (1 + DW) + PW;
                    int oh = ih - kh * (1 + DH) + PH;
#if SW != 1 || SH != 1
                    if (ow % SW != 0 || oh % SH != 0) continue;
                    ow /= SW;
                    oh /= SH;
#endif
                    if (oh >= OH || ow >= OW) continue;

                    const __global DATA_T *diff_dst1 = diff_dst
                            + ow * OC_BLOCK * MB_BLOCK
                            + oh * OW * OC_BLOCK * MB_BLOCK;
#if CASE_3D
                    diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
#endif
                    const __global DATA_T *wei1 = wei
#if CASE_3D
                            + kd * KH * KW * OC_BLOCK * IC_BLOCK
#endif
                            + kh * KW * OC_BLOCK * IC_BLOCK
                            + kw * OC_BLOCK * IC_BLOCK;
#else
        int ow = (iw + PW);
        int oh = (ih + PH);
#if CASE_3D
        int od = (id + PD);
#endif
        bool do_ker = true;
#if SW != 1 || SH != 1 || SD != 1
        do_ker = ow % SW == 0 && oh % SH == 0;
        ow /= SW;
        oh /= SH;
#if CASE_3D
        do_ker = do_ker && od % SD == 0;
        od /= SD;
#endif
#endif
#if PH != 0 || PW != 0 || PD != 0
        do_ker = do_ker && (oh < OH && ow < OW);
#if CASE_3D
        do_ker = do_ker && (od < OD);
#endif
#endif
#if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
        if (do_ker) {
#endif
            const __global DATA_T *diff_dst1 = diff_dst
                    + ow * OC_BLOCK * MB_BLOCK + oh * OW * OC_BLOCK * MB_BLOCK;
#if CASE_3D
            diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
#endif
            const __global DATA_T *wei1 = wei;
#endif

#define LOAD_DIFF_DST(_block, _diff_dst, mb_chunk) \
    { \
        (_block) = AS_DATA8_T( \
                BLOCK_READ8((const __global BLOCK_DATA_T *)((_diff_dst) \
                        + (mb_chunk)*OC_BLOCK))); \
    }

#define SAVE_SRC_DIFF(_block, _diff_src, mb_chunk) \
    { \
        BLOCK_WRITE8((const __global BLOCK_DATA_T *)(&( \
                             _diff_src)[(mb_chunk)*IC_BLOCK]), \
                AS_BLOCK_DATA8_T((_block))); \
    }

#if DT_F32
#define TRANSPOSE_8(_block, _col) \
    (DATA8_T)(intel_sub_group_shuffle(_block, _col))
#else
#define TRANSPOSE_8(_block, _col) \
    (DATA8_T)(intel_sub_group_shuffle(_block[0], _col), \
            intel_sub_group_shuffle(_block[1], _col), \
            intel_sub_group_shuffle(_block[2], _col), \
            intel_sub_group_shuffle(_block[3], _col), \
            intel_sub_group_shuffle(_block[4], _col), \
            intel_sub_group_shuffle(_block[5], _col), \
            intel_sub_group_shuffle(_block[6], _col), \
            intel_sub_group_shuffle(_block[7], _col))
#endif

#define FMA8(a, b, c) fma((DATA8_T)(a), (DATA8_T)b, (DATA8_T)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, _blockB1) \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_8(_blockA, 0), _result); \
        _result = FMA8(_blockB.s1, TRANSPOSE_8(_blockA, 1), _result); \
        _result = FMA8(_blockB.s2, TRANSPOSE_8(_blockA, 2), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_8(_blockA, 3), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_8(_blockA, 4), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_8(_blockA, 5), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_8(_blockA, 6), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_8(_blockA, 7), _result); \
        _result = FMA8(_blockB1.s0, TRANSPOSE_8(_blockA, 8), _result); \
        _result = FMA8(_blockB1.s1, TRANSPOSE_8(_blockA, 9), _result); \
        _result = FMA8(_blockB1.s2, TRANSPOSE_8(_blockA, 10), _result); \
        _result = FMA8(_blockB1.s3, TRANSPOSE_8(_blockA, 11), _result); \
        _result = FMA8(_blockB1.s4, TRANSPOSE_8(_blockA, 12), _result); \
        _result = FMA8(_blockB1.s5, TRANSPOSE_8(_blockA, 13), _result); \
        _result = FMA8(_blockB1.s6, TRANSPOSE_8(_blockA, 14), _result); \
        _result = FMA8(_blockB1.s7, TRANSPOSE_8(_blockA, 15), _result); \
    }

                    DATA8_T blockA0, blockA1;
                    LOAD_DIFF_DST(blockA0, diff_dst1, 0);
                    LOAD_DIFF_DST(blockA1, diff_dst1, 8);
                    DATA8_T blockB00 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)wei1));
                    DATA8_T blockB01 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)(wei1
                                    + 8 * IC_BLOCK)));
                    MULTIPLY_BLOCKS_8x8(blockC00, blockA0, blockB00, blockB01);
                    MULTIPLY_BLOCKS_8x8(blockC01, blockA1, blockB00, blockB01);

#undef TRANSPOSE_BLOCK_8
#undef MULTIPLY_BLOCKS_8x8
#if KH != 1 || KW != 1 || KD != 1
                }
#else
#if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
        }
#endif
#endif
        diff_dst += OC_BLOCK * OD * OH * OW * MB_BLOCK;
        wei += IC * KD * KH * KW * OC_BLOCK;
        ocb += OC_BLOCK;
    } while (ocb < OC);

    __global DATA_T *src_write0 = diff_src + mb * IC * G * ID * IH * IW
            + gic * ID * IH * IW * IC_BLOCK * MB_BLOCK
            + g * IC * ID * IH * IW * MB_BLOCK
            + id * IH * IW * IC_BLOCK * MB_BLOCK + ih * IW * IC_BLOCK * MB_BLOCK
            + iw * IC_BLOCK * MB_BLOCK;

    SAVE_SRC_DIFF(blockC00, src_write0, 0);
    SAVE_SRC_DIFF(blockC01, src_write0, 8);

#endif
#if VER_8OW16C == 1

    const int sp = get_group_id(1);
    const int local_id = get_local_id(0);
    const int icb_mb = get_group_id(2);
    const int mb = icb_mb / (G * IC / ICB);
    const int icb = icb_mb % (G * IC / ICB);
    const int ic = (icb * ICB) / IC_BLOCK + get_group_id(0);

    const int g = ic / (IC / IC_BLOCK);
    const int gic = ic % (IC / IC_BLOCK);

#if CASE_3D
    const int id = sp / (IWB * IH);
    const int ihw = sp % (IWB * IH);
#else
    const int id = 0;
    const int ihw = sp;
#endif
    const int ih = ihw / IWB;
    const int iw = (ihw % IWB) * IW_BLOCK;

    diff_dst += mb * OC * G * OD * OH * OW + g * OC * OD * OH * OW * MB_BLOCK;
    DATA_T blockC00[IW_BLOCK] = {DATA_ZERO};

    if (WITH_BIAS) {
#if IS_DW
        const int bg_off = g * IC + local_id;
        const int bc_off = gic * IC_BLOCK;
        DATA_T b = (G_WO_PADDING % IC_BLOCK == 0 || bg_off < G_WO_PADDING)
                ? bias[bg_off + bc_off]
                : DATA_ZERO;
#else
        const int bg_off = g * IC;
        const int bc_off = gic * IC_BLOCK + local_id;
        DATA_T b = (IC_WO_PADDING % IC_BLOCK == 0 || bc_off < IC_WO_PADDING)
                ? bias[bg_off + bc_off]
                : DATA_ZERO;
#endif
        unroll_for(int i = 0; i < IW_BLOCK; ++i) { blockC00[i] = b; }
    }

    wei += gic * KD * KH * KW * OC_BLOCK * IC_BLOCK
            + g * IC * OC * KD * KH * KW;

    for (int ocb = 0; ocb < OC; ocb += OC_BLOCK) {
        const __global DATA_T *diff_dst1;
        const __global DATA_T *wei1;

#if KH != 1 || KW != 1 || KD != 1
        for (int kd = 0; kd < KD; ++kd) {
#if CASE_3D
            if (id + PD < kd * (1 + DD)) continue;
            int od = id - kd * (1 + DD) + PD;
            if (od % SD != 0) continue;
            od /= SD;
            if (od >= OD) continue;
#endif
            for (int kh = 0; kh < KH; ++kh) {
                if (ih + PH < kh * (1 + DH)) continue;
                int oh = ih - kh * (1 + DH) + PH;
                if (oh % SH != 0) continue;
                oh /= SH;
                if (oh >= OH) continue;

                __attribute__((opencl_unroll_hint(KW))) // attr:no-format
                for (int kw = 0; kw < KW; ++kw) {

                    diff_dst1 = diff_dst + oh * OW * OC_BLOCK * MB_BLOCK;
#if CASE_3D
                    diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
#endif
                    wei1 = wei
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
            diff_dst1 = diff_dst + oh * OW * OC_BLOCK * MB_BLOCK;
#if CASE_3D
            diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
#endif
            wei1 = wei;
#endif

#define TRANSPOSE_1(_block, _col) \
    (DATA_T)(intel_sub_group_shuffle(_block, _col))

#define FMA1(a, b, c) fma((DATA_T)(a), (DATA_T)b, (DATA_T)c)
#define HAS_PAD_W (PW > 0 || (OW - 1) * SW - PW + (KW - 1) * (1 + DW) >= IW)

#define _BLOCK_READ8(ptr) \
    AS_DATA8_T(BLOCK_READ8((const __global BLOCK_DATA_T *)(ptr)))
#define _BLOCK_READ4(ptr) \
    AS_DATA4_T(BLOCK_READ4((const __global BLOCK_DATA_T *)(ptr)))
#define _BLOCK_READ2(ptr) \
    AS_DATA2_T(BLOCK_READ2((const __global BLOCK_DATA_T *)(ptr)))
#define _BLOCK_READ(ptr) \
    AS_DATA_T(BLOCK_READ((const __global BLOCK_DATA_T *)(ptr)))

// block[i].j = ptr[i * 16 + j], 0 <= i < n, 0 <= j < 16
#define unrolled_read(n, block, ptr) \
    do { \
        if ((n)&16) { \
            *((DATA8_T *)(block)) = _BLOCK_READ8((ptr)); \
            *((DATA8_T *)((block) + 8)) = _BLOCK_READ8((ptr) + 8 * 16); \
        } \
        if ((n)&8) \
            *((DATA8_T *)((block) + ((n) & ~15))) \
                    = _BLOCK_READ8((ptr) + ((n) & ~15) * 16); \
        if ((n)&4) \
            *((DATA4_T *)((block) + ((n) & ~7))) \
                    = _BLOCK_READ4((ptr) + ((n) & ~7) * 16); \
        if ((n)&2) \
            *((DATA2_T *)((block) + ((n) & ~3))) \
                    = _BLOCK_READ2((ptr) + ((n) & ~3) * 16); \
        if ((n)&1) \
            *((block) + ((n) & ~1)) = _BLOCK_READ((ptr) + ((n) & ~1) * 16); \
    } while (0)

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
                    DATA8_T blockB00 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)wei1));
                    DATA8_T blockB01 = AS_DATA8_T(
                            BLOCK_READ8((const __global BLOCK_DATA_T *)(wei1
                                    + 8 * IC_BLOCK)));
                    DATA_T blockA[IW_BLOCK] = {0};
#if KW == 1 && !HAS_PAD_W && SW == 1
                    for (int i = 0; i < IW_BLOCK; i += 8) {
                        int iw_bound = min(8, IW_BLOCK - i);
                        int ow = iw + i;
                        unrolled_read(iw_bound, &blockA[i],
                                &(diff_dst1)[ow * OC_BLOCK]);
                    }
                    __attribute__((
                            opencl_unroll_hint(IW_BLOCK))) // attr:no-format
                    for (int i = 0; i < IW_BLOCK; i++) {
                        MULTIPLY_BLOCKS_8x8(
                                blockC00[i], blockA[i], blockB00, blockB01);
                    }
#else
            __attribute__((opencl_unroll_hint(IW_BLOCK))) // attr:no-format
            for (int i = 0; i < IW_BLOCK; i++) {
#if KW != 1
                if (iw + i + PW < kw * (1 + DW)) continue;
                int ow = iw + i - kw * (1 + DW) + PW;
#else
            int ow = iw + i + PW;
#endif
#if SW != 1
                if (ow % SW != 0) continue;
                ow /= SW;
#endif
                if (ow >= OW) continue;
                blockA[i] = AS_DATA_T(
                        BLOCK_READ((const __global BLOCK_DATA_T *)(&(
                                diff_dst1)[ow * OC_BLOCK])));
            }
            __attribute__((opencl_unroll_hint(IW_BLOCK))) // attr:no-format
            for (int i = 0; i < IW_BLOCK; i++) {
                MULTIPLY_BLOCKS_8x8(blockC00[i], blockA[i], blockB00, blockB01);
            }

#endif
#undef TRANSPOSE_BLOCK_8
#undef MULTIPLY_BLOCKS_8x8
#if KH != 1 || KW != 1 || KD != 1
                }
            }
        }

#else
#if SW != 1 || SH != 1 || SD != 1 || PH != 0 || PW != 0 || PD != 0
        }
#endif
#endif
        diff_dst += OC_BLOCK * OD * OH * OW * MB_BLOCK;
        wei += IC * KD * KH * KW * OC_BLOCK;
    }
    __global DATA_T *src_write0 = diff_src + mb * IC * G * ID * IH * IW
            + gic * ID * IH * IW * IC_BLOCK * MB_BLOCK
            + g * IC * ID * IH * IW * MB_BLOCK
            + id * IH * IW * IC_BLOCK * MB_BLOCK + ih * IW * IC_BLOCK * MB_BLOCK
            + iw * IC_BLOCK * MB_BLOCK;

    for (int i = 0; i < IW_BLOCK; i++) {
        if (iw + i >= IW) continue;
        BLOCK_WRITE((__global BLOCK_DATA_T *)(&(src_write0)[i * IC_BLOCK]),
                AS_BLOCK_DATA_T(blockC00[i]));
    }
#endif
}
#endif
