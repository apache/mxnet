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

#if BWD_DATA == 1

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
__kernel void
gen9_conv_dw_bwd_data(__global DATA_T *diff_src, __global DATA_T *wei,
        __global DATA_T *diff_dst, __global DATA_T *bias) {

#if VER_16MB16C == 1
    const int mb_unroll = 16;

    const int ic = get_group_id(1);
    const int sp = get_group_id(0);
    const int local_id = get_local_id(1);
    int mb = get_group_id(2) * mb_unroll;

    const int g = ic * IC_BLOCK;
    const int gic = 0;

    const int id = sp / (IW * IH);
    const int ihw = sp % (IW * IH);
    const int ih = ihw / IW;
    const int iw = ihw % IW;

    diff_dst += mb * OC * G * OD * OH * OW + g * OC * OD * OH * OW * MB_BLOCK;

    DATA8_T blockC00 = (DATA8_T)DATA_ZERO;
    DATA8_T blockC01 = (DATA8_T)DATA_ZERO;

    if (WITH_BIAS) {
        const int bg_off = g * IC + gic * IC_BLOCK + local_id;
        DATA_T b = (G_WO_PADDING % IC_BLOCK == 0 || bg_off < G_WO_PADDING)
                ? bias[bg_off]
                : DATA_ZERO;
        unroll_for(int i = 0; i < 8; ++i) {
            blockC00[i] = b;
            blockC01[i] = b;
        }
    }

    wei += gic * KD * KH * KW * OC_BLOCK * IC_BLOCK
            + g * IC * OC * KD * KH * KW;
#if KH != 1 || KW != 1 || KD != 1
    for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw) {
                if (id + PD < kd * (1 + DD) || iw + PW < kw * (1 + DW)
                        || ih + PH < kh * (1 + DH))
                    continue;
                int od = id - kd * (1 + DD) + PD;
                int ow = iw - kw * (1 + DW) + PW;
                int oh = ih - kh * (1 + DH) + PH;
                if (od % SD != 0 || ow % SW != 0 || oh % SH != 0) continue;
                od /= SD;
                ow /= SW;
                oh /= SH;
                if (od >= OD || oh >= OH || ow >= OW) continue;

                const __global DATA_T *diff_dst1 = diff_dst
                        + ow * OC_BLOCK * MB_BLOCK
                        + oh * OW * OC_BLOCK * MB_BLOCK;
                diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
                const __global DATA_T *wei1 = wei + kd * KH * KW * OC_BLOCK
                        + kh * KW * OC_BLOCK + kw * OC_BLOCK;
#else
    int ow = (iw + PW);
    int oh = (ih + PH);
    int od = (id + PD);
    bool do_ker = ow % SW == 0 && oh % SH == 0 && od % SD == 0;
    ow /= SW;
    oh /= SH;
    od /= SD;
#if PH != 0 || PW != 0 || PD != 0
    do_ker = do_ker && (od < OD && oh < OH && ow < OW);
#endif
    if (do_ker) {
        const __global DATA_T *diff_dst1 = diff_dst + ow * OC_BLOCK * MB_BLOCK
                + oh * OW * OC_BLOCK * MB_BLOCK;
        diff_dst1 += od * OH * OW * OC_BLOCK * MB_BLOCK;
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

                DATA8_T blockA0, blockA1;
                LOAD_DIFF_DST(blockA0, diff_dst1, 0);
                LOAD_DIFF_DST(blockA1, diff_dst1, 8);
                DATA_T blockB00 = AS_DATA_T(
                        BLOCK_READ((const __global BLOCK_DATA_T *)wei1));
                blockC00 = fma(blockA0, (DATA8_T)blockB00, blockC00);
                blockC01 = fma(blockA1, (DATA8_T)blockB00, blockC01);

#if KH != 1 || KW != 1 || KD != 1
            }
#else
    }
#endif

    __global DATA_T *src_write0 = diff_src + mb * IC * G * ID * IH * IW
            + gic * ID * IH * IW * IC_BLOCK * MB_BLOCK
            + g * IC * ID * IH * IW * MB_BLOCK
            + id * IH * IW * IC_BLOCK * MB_BLOCK + ih * IW * IC_BLOCK * MB_BLOCK
            + iw * IC_BLOCK * MB_BLOCK;

    SAVE_SRC_DIFF(blockC00, src_write0, 0);
    SAVE_SRC_DIFF(blockC01, src_write0, 8);

#endif
#if VER_8OW16C == 1
    const int ic = get_group_id(1);
    const int sp = get_group_id(0);
    const int local_id = get_local_id(1);
    const int mb = get_group_id(2);

    const int g = ic * IC_BLOCK;
    const int gic = 0;

    const int id = sp / (IWB * IH);
    const int ihw = sp % (IWB * IH);
    const int ih = ihw / IWB;
    const int iw = (ihw % IWB) * IW_BLOCK;

    diff_dst += mb * OC * G * OD * OH * OW + g * OC * OD * OH * OW * MB_BLOCK;

    DATA_T blockC00[IW_BLOCK] = {DATA_ZERO};

    if (WITH_BIAS) {
        const int bg_off = g * IC + gic * IC_BLOCK + local_id;
        DATA_T b = (G_WO_PADDING % IC_BLOCK == 0 || bg_off < G_WO_PADDING)
                ? bias[bg_off]
                : DATA_ZERO;
        unroll_for(int i = 0; i < IW_BLOCK; ++i) { blockC00[i] = b; }
    }

    wei += gic * KD * KH * KW * OC_BLOCK * IC_BLOCK
            + g * IC * OC * KD * KH * KW;

#if KH != 1 || KW != 1 || KD != 1
    for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw) {

                if (id + PD < kd * (1 + DD)) continue;
                if (ih + PH < kh * (1 + DH)) continue;
                int od = id - kd * (1 + DD) + PD;
                int oh = ih - kh * (1 + DH) + PH;
                if (od % SD != 0 || oh % SH != 0) continue;
                od /= SD;
                oh /= SH;
                if (od >= OD || oh >= OH) continue;

                const __global DATA_T *diff_dst1 = diff_dst
                        + oh * OW * OC_BLOCK * MB_BLOCK
                        + od * OH * OW * OC_BLOCK * MB_BLOCK;
                const __global DATA_T *wei1 = wei + kd * KH * KW * OC_BLOCK
                        + kh * KW * OC_BLOCK + kw * OC_BLOCK;
#else
    int oh = (ih + PH);
    int od = (id + PD);
    bool do_ker = od % SD == 0 && oh % SH == 0;
    oh /= SH;
    od /= SD;
#if PH != 0 || PW != 0 || PD != 0
    do_ker = do_ker && (oh < OH && od < OD);
#endif
    if (do_ker) {
        const __global DATA_T *diff_dst1 = diff_dst
                + oh * OW * OC_BLOCK * MB_BLOCK
                + od * OH * OW * OC_BLOCK * MB_BLOCK;
        const __global DATA_T *wei1 = wei;
#endif

                DATA_T blockB00 = AS_DATA_T(
                        BLOCK_READ((const __global BLOCK_DATA_T *)wei1));
                DATA_T blockA[IW_BLOCK];

                __attribute__((opencl_unroll_hint(IW_BLOCK))) // attr:no-format
                for (int i = 0; i < IW_BLOCK; i++) {
                    if (iw + i + PW < kw * (1 + DW)) {
                        blockA[i] = 0.0;
                        continue;
                    }
                    int ow = iw + i - kw * (1 + DW) + PW;
                    if (ow % SW != 0) {
                        blockA[i] = 0.0;
                        continue;
                    }
                    ow /= SW;
                    if (ow >= OW) {
                        blockA[i] = 0.0;
                        continue;
                    }
                    blockA[i] = AS_DATA_T(
                            BLOCK_READ((const __global BLOCK_DATA_T *)(&(
                                    diff_dst1)[ow * OC_BLOCK])));
                }

                __attribute__((opencl_unroll_hint(IW_BLOCK))) // attr:no-format
                for (int i = 0; i < IW_BLOCK; i++) {
                    blockC00[i] = fma(blockA[i], (DATA_T)blockB00, blockC00[i]);
                }

                diff_dst1 += OC_BLOCK * OD * OH * OW * MB_BLOCK;
                wei1 += IC * KD * KH * KW * OC_BLOCK;

#if KH != 1 || KW != 1 || KD != 1
            }
#else
    }
#endif

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
