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

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#if OD > 1
#define CASE_3D 1
#else
#define CASE_3D 0
#endif

#define HAS_PAD_D (PD != 0 || PD_R != 0)
#define HAS_PAD_H (PH != 0 || PH_R != 0)
#define HAS_PAD_W (PW != 0 || PW_R != 0)

inline void atomic_add_global(
        volatile __global atomic_float *source, float operand) {
    float old_val = atomic_load_explicit(
            source, memory_order_relaxed, memory_scope_device);
    if (isnan(operand)) return;
    bool success = false;
    do {
        float new_val = old_val + operand;
        success = atomic_compare_exchange_strong_explicit(source, &old_val,
                new_val, memory_order_acq_rel, memory_order_relaxed,
                memory_scope_device);
    } while (!success);
}

#if BWD_WEIGHTS == 1

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
#if VER_16MB16C == 1 || VER_8OW16C == 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
#endif
__kernel void
gen9_conv_bwd_weights(__global float *src,
        volatile __global atomic_float *diff_wei,
        volatile __global atomic_float *diff_bias, __global float *diff_dst) {

#if VER_16MB16C == 1

    const uint ksp = get_global_id(1);
#if CASE_3D
    const uint kd = ksp / (KW * KH);
    const uint khw = ksp % (KW * KH);
#else
    const uint khw = ksp;
    const uint kd = 0;
#endif
    const uint kh = khw / KW;
    const uint kw = khw % KW;
    const uint local_x = get_local_id(0);

    const uint chunk = get_global_id(2) / ((IC / ICB) * (OC / OCB));
    const uint icb_ocb = get_global_id(2) % ((IC / ICB) * (OC / OCB));
    const uint icb = icb_ocb % (IC / ICB);
    const uint ocb = icb_ocb / (IC / ICB);

#if IS_DW
    const uint g = 0;
    const uint oc = get_group_id(0);
    const uint ic = oc;
#else
    const uint g_ic_oc = get_global_id(0);
    const uint g = g_ic_oc / (OCB * (ICB / IC_BLOCK));
    const uint io = g_ic_oc % (OCB * (ICB / IC_BLOCK));

    const uint oc = (io % OCB) / OC_BLOCK + ocb * (OCB / OC_BLOCK);
    const uint ic = io / OCB + icb * (ICB / IC_BLOCK);
#endif

    const uint sp_chunk = chunk % OSP_CHUNK;
    const uint mb_chunk = chunk / OSP_CHUNK;

    const uint oh_nb = (OH + OHB - 1) / OHB;
    const uint ow_nb = (OW + OWB - 1) / OWB;

    const uint od_beg = (sp_chunk / ow_nb) / oh_nb * ODB;
    const uint oh_beg = (sp_chunk / ow_nb) % oh_nb * OHB;
    const uint ow_beg = (sp_chunk % ow_nb) * OWB;

    const uint mb = mb_chunk * (MB_CHUNK_SIZE);
    const uint mb_end = min((mb_chunk + 1) * (MB_CHUNK_SIZE), (uint)MB);

    const bool do_bias = (ic == 0 || IS_DW) && kh == 0 && kw == 0 && kd == 0;

    src += ic * ID * IH * IW * IC_BLOCK * MB_BLOCK + mb * IC * G * ID * IH * IW
            + g * IC * ID * IH * IW * MB_BLOCK;
    diff_dst += oc * OD * OH * OW * OC_BLOCK * MB_BLOCK
            + g * OC * OD * OH * OW * MB_BLOCK;

#if WITH_BIAS == 1
    diff_bias += g * OC + oc * OC_BLOCK + local_x;
    float bias_loc = 0.0f;
#endif

#if IS_DW
    float blockC00 = 0.0f;
#else
    float8 blockC00 = 0.0f;
    float8 blockC01 = 0.0f;
#endif

#if MB != (MB_CHUNK * MB_BLOCK)
    uint omb = mb;
    do {
        const __global float *diff_dst1_
                = diff_dst + omb * OC * G * OD * OH * OW;
#else
    const __global float *diff_dst1_ = diff_dst + mb * OC * G * OD * OH * OW;
#endif
        for (uint od = od_beg; od < min(od_beg + ODB, (uint)OD); od++) {
            for (uint oh = oh_beg; oh < min(oh_beg + OHB, (uint)OH); oh++) {
                for (uint ow = ow_beg; ow < min(ow_beg + OWB, (uint)OW); ow++) {
                    const __global float *diff_dst1 = diff_dst1_
                            + od * OH * OW * OC_BLOCK * MB_BLOCK
                            + oh * OW * OC_BLOCK * MB_BLOCK
                            + ow * OC_BLOCK * MB_BLOCK;
                    const uint ih = oh * SH - PH + kh * (1 + DH);
                    const uint iw = ow * SW - PW + kw * (1 + DW);
#if CASE_3D
                    const uint id = od * SD - PD + kd * (1 + DD);
#endif

                    if (iw < 0 || ih < 0 || iw >= IW || ih >= IH
#if CASE_3D
                            || id < 0 || id >= ID
#endif
                    ) {
#if WITH_BIAS == 1
                        if (do_bias) {
                            float8 blockB
                                    = as_float8(intel_sub_group_block_read8((
                                            const __global uint *)(diff_dst1)));
                            for (int i = 0; i < 8; i++)
                                bias_loc += blockB[i];
                            blockB = as_float8(intel_sub_group_block_read8(
                                    (const __global uint *)(diff_dst1
                                            + 8 * OC_BLOCK)));
                            for (int i = 0; i < 8; i++)
                                bias_loc += blockB[i];
                        }
#endif
                        continue;
                    }

                    const __global float *src1 = src
                            + ih * IW * IC_BLOCK * MB_BLOCK
                            + iw * IC_BLOCK * MB_BLOCK;
#if CASE_3D
                    src1 += id * IH * IW * IC_BLOCK * MB_BLOCK;
#endif
#define TRANSPOSE_8(_block, _row, _col) \
    (float8)(intel_sub_group_shuffle(_block[_row], 0 + _col), \
            intel_sub_group_shuffle(_block[_row], 1 + _col), \
            intel_sub_group_shuffle(_block[_row], 2 + _col), \
            intel_sub_group_shuffle(_block[_row], 3 + _col), \
            intel_sub_group_shuffle(_block[_row], 4 + _col), \
            intel_sub_group_shuffle(_block[_row], 5 + _col), \
            intel_sub_group_shuffle(_block[_row], 6 + _col), \
            intel_sub_group_shuffle(_block[_row], 7 + _col))

#define FMA8(a, b, c) fma((float8)(a), (float8)b, (float8)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, col) \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_8(_blockA, 0, col), _result); \
        _result = FMA8(_blockB.s1, TRANSPOSE_8(_blockA, 1, col), _result); \
        _result = FMA8(_blockB.s2, TRANSPOSE_8(_blockA, 2, col), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_8(_blockA, 3, col), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_8(_blockA, 4, col), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_8(_blockA, 5, col), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_8(_blockA, 6, col), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_8(_blockA, 7, col), _result); \
    }

#if IS_DW
                    float8 blockA = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(src1)));
                    float8 blockA1 = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(src1 + 8 * IC_BLOCK)));

                    float8 blockB = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(diff_dst1)));
                    float8 blockB1 = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(diff_dst1 + 8 * OC_BLOCK)));

                    for (int i = 0; i < 8; i++) {
                        blockC00 = fma(blockA[i], blockB[i], blockC00);
                    }

#if WITH_BIAS == 1
                    for (int i = 0; i < 8; i++)
                        bias_loc += blockB[i];
#endif

                    for (int i = 0; i < 8; i++) {
                        blockC00 = fma(blockA1[i], blockB1[i], blockC00);
                    }

#if WITH_BIAS == 1
                    for (int i = 0; i < 8; i++)
                        bias_loc += blockB1[i];
#endif
#else
                float8 blockA = as_float8(intel_sub_group_block_read8(
                        (const __global uint *)(src1)));
                float8 blockB = as_float8(intel_sub_group_block_read8(
                        (const __global uint *)(diff_dst1)));

                MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB, 0);
                MULTIPLY_BLOCKS_8x8(blockC01, blockA, blockB, 8);

#if WITH_BIAS == 1
                for (int i = 0; i < 8; i++)
                    bias_loc += blockB[i];
#endif
                blockA = as_float8(intel_sub_group_block_read8(
                        (const __global uint *)(src1 + 8 * IC_BLOCK)));
                blockB = as_float8(intel_sub_group_block_read8(
                        (const __global uint *)(diff_dst1 + 8 * OC_BLOCK)));
                MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB, 0);
                MULTIPLY_BLOCKS_8x8(blockC01, blockA, blockB, 8);
#if WITH_BIAS == 1
                for (int i = 0; i < 8; i++)
                    bias_loc += blockB[i];
#endif
#endif
                }
            }
        }
#if MB != (MB_CHUNK * MB_BLOCK)
        omb += MB_BLOCK;
        src += IC * G * ID * IH * IW * MB_BLOCK;
    } while (omb < mb_end);
#endif

#if WITH_BIAS == 1
    if (do_bias
            && oc * OC_BLOCK + local_x < (IS_DW ? G_WO_PADDING : OC_WO_PADDING))
        atomic_add_global(diff_bias, bias_loc);

#endif

#if IS_DW
    diff_wei += oc * KD * KH * KW * OC_BLOCK + kd * KH * KW * OC_BLOCK
            + kh * KW * OC_BLOCK + kw * OC_BLOCK;
    atomic_add_global(diff_wei + local_x, blockC00);
#else
    diff_wei += ic * OC * KD * KH * KW * IC_BLOCK
            + oc * KD * KH * KW * IC_BLOCK * OC_BLOCK
            + kd * KH * KW * IC_BLOCK * OC_BLOCK + kh * KW * IC_BLOCK * OC_BLOCK
            + kw * IC_BLOCK * OC_BLOCK + g * OC * IC * KD * KH * KW;
#if NCHUNK == 1
    intel_sub_group_block_write8((__global uint *)diff_wei, as_uint8(blockC00));
    intel_sub_group_block_write8(
            (__global uint *)(diff_wei + 8 * OC_BLOCK), as_uint8(blockC01));
#else
    for (int i = 0; i < 8; i++)
        atomic_add_global(diff_wei + i * OC_BLOCK + local_x, blockC00[i]);

    for (int i = 0; i < 8; i++)
        atomic_add_global(diff_wei + (8 + i) * OC_BLOCK + local_x, blockC01[i]);
#endif
#endif

#endif
#if VER_8OW16C == 1
#define HAS_PAD_W (PW > 0 || (OW - 1) * SW - PW + (KW - 1) * (1 + DW) >= IW)
    const int ksp = get_global_id(1);
#if CASE_3D
    const int kd = ksp / (KW * KH);
    const int khw = ksp % (KW * KH);
#else
    const int khw = ksp;
    const int kd = 0;
#endif
    const int kh = khw / KW;
    const int kw = khw % KW;
    const int local_x = get_local_id(0);

    const int chunk = get_global_id(2) % NCHUNK;
    const int icb_ocb = get_global_id(2) / NCHUNK;
    const int icb = icb_ocb % (IC / ICB);
    const int ocb = icb_ocb / (IC / ICB);

#if IS_DW
    const int g = 0;
    const int oc = get_group_id(0);
    const int ic = oc;
#else
    const int g_ic_oc = get_global_id(0);
    const int g = g_ic_oc / (OC * (IC / IC_BLOCK));
    const int io = g_ic_oc % (OC * (IC / IC_BLOCK));
    const int oc = (io % OCB) / OC_BLOCK + ocb * (OCB / OC_BLOCK);
    const int ic = (IC == 3) ? 0 : (io / OCB + icb * (ICB / IC_BLOCK));
#endif

    const int sp_chunk = chunk % OSP_CHUNK;
    const int mb_chunk = chunk / OSP_CHUNK;

    const int ow_nb = (OW + OWB - 1) / OWB;
    const int oh_nb = (OH + OHB - 1) / OHB;

    const int od_beg = ((sp_chunk / ow_nb) / oh_nb) * ODB;
    const int oh_beg = ((sp_chunk / ow_nb) % oh_nb) * OHB;
    const int ow_beg = (sp_chunk % ow_nb) * OWB;

    const int mb = mb_chunk * MB_CHUNK_SIZE;
    const int mb_end = min((mb_chunk + 1) * MB_CHUNK_SIZE, MB);

    const bool do_bias = (ic == 0 || IS_DW) && kh == 0 && kw == 0 && kd == 0;
    const int OW_LOOP_BLOCK = 8;

    src += ic * ID * IH * IW * IC_BLOCK * MB_BLOCK + mb * IC * G * ID * IH * IW
            + g * IC * ID * IH * IW * MB_BLOCK;
    diff_dst += oc * OD * OH * OW * OC_BLOCK * MB_BLOCK
            + g * OC * OD * OH * OW * MB_BLOCK;

#if WITH_BIAS == 1
    diff_bias += g * OC + oc * OC_BLOCK + local_x;
    float bias_loc = 0.0f;
#endif

#if IC == 3
    float8 blockC00 = 0.0f;
#elif IS_DW
    float blockC00 = 0.0f;
#else
    float8 blockC00 = 0.0f;
    float8 blockC01 = 0.0f;
#endif

    for (int omb = mb; omb < mb_end; omb++) {
        const __global float *diff_dst1_
                = diff_dst + omb * OC * G * OD * OH * OW;

        for (int od = od_beg; od < min(od_beg + ODB, OD); od++)
            for (int oh = oh_beg; oh < min(oh_beg + OHB, OH); oh++) {
                const __global float *diff_dst1 = diff_dst1_
                        + od * OH * OW * OC_BLOCK + oh * OW * OC_BLOCK;
                if (oh * SH + kh * (1 + DH) < PH
                        || oh * SH + kh * (1 + DH) >= IH + PH
#if CASE_3D
                        || od * SD + kd * (1 + DD) < PD
                        || od * SD + kd * (1 + DD) >= ID + PD
#endif
                ) {
#if WITH_BIAS == 1
                    if (do_bias) {
                        for (int ow = ow_beg; ow < ow_beg + OWB;
                                ow += OW_LOOP_BLOCK) {
                            float8 blockB;
#if OW % OWB != 0
                            for (int i = 0; i < OW_LOOP_BLOCK; i++) {
                                if (ow + i >= OW) {
                                    blockB[i] = 0.0;
                                } else {
                                    blockB[i] = as_float(
                                            intel_sub_group_block_read((
                                                    const __global uint
                                                            *)(&diff_dst1[(ow + i)
                                                    * OC_BLOCK])));
                                }
                            }
#else
                            blockB = as_float8(intel_sub_group_block_read8(
                                    (const __global uint *)(diff_dst1
                                            + ow * OC_BLOCK)));
#endif

                            for (int i = 0; i < OW_LOOP_BLOCK; i++)
                                bias_loc += blockB[i];
                        }
                    }
#endif
                    continue;
                }

                for (int ow = ow_beg;
                        ow < min(ow_beg + OWB, (OW / OW_BLOCK) * OW_BLOCK);
                        ow += OW_BLOCK) {

                    const int id = od * SD - PD + kd * (1 + DD);
                    const int ih = oh * SH - PH + kh * (1 + DH);
                    const int iw = ow * SW - PW + kw * (1 + DW);
                    __global float *src1;

                    src1 = src + id * IH * IW * IC_BLOCK + ih * IW * IC_BLOCK
                            + iw * IC_BLOCK;

#define TRANSPOSE_8(_block, _row, _col) \
    { \
        (float8)(intel_sub_group_shuffle(_block[_row], 0 + _col), \
                intel_sub_group_shuffle(_block[_row], 1 + _col), \
                intel_sub_group_shuffle(_block[_row], 2 + _col), \
                intel_sub_group_shuffle(_block[_row], 3 + _col), \
                intel_sub_group_shuffle(_block[_row], 4 + _col), \
                intel_sub_group_shuffle(_block[_row], 5 + _col), \
                intel_sub_group_shuffle(_block[_row], 6 + _col), \
                intel_sub_group_shuffle(_block[_row], 7 + _col)) \
    }

#define FMA8(a, b, c) fma((float8)(a), (float8)b, (float8)c)

#define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB, col) \
    { \
        _result = FMA8(_blockB.s0, TRANSPOSE_8(_blockA, 0, col), _result); \
        _result = FMA8(_blockB.s1, TRANSPOSE_8(_blockA, 1, col), _result); \
        _result = FMA8(_blockB.s2, TRANSPOSE_8(_blockA, 2, col), _result); \
        _result = FMA8(_blockB.s3, TRANSPOSE_8(_blockA, 3, col), _result); \
        _result = FMA8(_blockB.s4, TRANSPOSE_8(_blockA, 4, col), _result); \
        _result = FMA8(_blockB.s5, TRANSPOSE_8(_blockA, 5, col), _result); \
        _result = FMA8(_blockB.s6, TRANSPOSE_8(_blockA, 6, col), _result); \
        _result = FMA8(_blockB.s7, TRANSPOSE_8(_blockA, 7, col), _result); \
    }

                    float8 blockA, blockB, blockC;
#if IC == 3
                    for (int i = 0; i < 3; i++) {
                        if (HAS_PAD_W
                                && (iw + (local_x)*SW < 0
                                        || iw + (local_x)*SW >= IW))
                            blockA[i] = 0;
                        else
                            blockA[i] = src1[i * ID * IH * IW + (local_x)*SW];
                    }
#else
                    for (int i = 0; i < OW_BLOCK; i++) {
                        if (HAS_PAD_W && (iw + i < 0 || iw + i * SW >= IW)) {
                            blockA[i] = 0;
                        } else {
                            blockA[i] = as_float(intel_sub_group_block_read((
                                    const __global uint
                                            *)(&src1[i * IC_BLOCK * SW])));
                        }
                    }
#endif

                    blockB = as_float8(intel_sub_group_block_read8((
                            const __global uint *)(diff_dst1 + ow * OC_BLOCK)));

#if IC == 3
                    blockC = as_float8(intel_sub_group_block_read8(
                            (const __global uint *)(diff_dst1
                                    + (ow + 8) * OC_BLOCK)));
                    for (int i = 0; i < OW_LOOP_BLOCK; i++) {
                        for (int j = 0; j < 3; j++)
                            blockC00[j] = fma(blockB[i],
                                    intel_sub_group_shuffle(blockA[j], i),
                                    blockC00[j]);
                        for (int j = 0; j < 3; j++)
                            blockC00[j] = fma(blockC[i],
                                    intel_sub_group_shuffle(blockA[j], i + 8),
                                    blockC00[j]);
                    }

#elif IS_DW
                    for (int i = 0; i < OW_LOOP_BLOCK; i++) {
                        blockC00 = fma(blockA[i], blockB[i], blockC00);
                    }
#else
                    MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB, 0);
                    MULTIPLY_BLOCKS_8x8(blockC01, blockA, blockB, 8);

#endif
#if WITH_BIAS == 1
                    for (int i = 0; i < OW_LOOP_BLOCK; i++) {
#if IC == 3
                        bias_loc += blockC[i];
#endif
                        bias_loc += blockB[i];
                    }
#endif
                }

                for (int ow = (OW / OW_BLOCK) * OW_BLOCK;
                        ow < min(ow_beg + OWB, OW); ow += OW_LOOP_BLOCK) {
                    const int id = od * SD - PD + kd * (1 + DD);
                    const int ih = oh * SH - PH + kh * (1 + DH);
                    const int iw = ow * SW - PW + kw * (1 + DW);
                    __global float *src1;
                    float8 blockA, blockB;

                    src1 = src + id * IH * IW * IC_BLOCK + ih * IW * IC_BLOCK
                            + iw * IC_BLOCK;
#if IC == 3
                    if (local_x < IC) {
                        for (int i = 0; i < min(OW_LOOP_BLOCK, OW - ow); i++) {
                            if (HAS_PAD_W
                                    && (iw + i * SW < 0 || iw + i * SW >= IW))
                                blockA[i] = 0;
                            else
                                blockA[i]
                                        = src1[local_x * ID * IH * IW + i * SW];
                        }
                    } else {
                        blockA = 0.0f;
                    }
#else

                    for (int i = 0; i < min(OW_LOOP_BLOCK, OW - ow); i++) {
                        if (HAS_PAD_W && (iw + i < 0 || iw + i * SW >= IW)) {
                            blockA[i] = 0;
                        } else {
                            blockA[i] = as_float(intel_sub_group_block_read((
                                    const __global uint
                                            *)(&src1[i * IC_BLOCK * SW])));
                        }
                    }
#endif

                    for (int i = 0; i < min(OW_LOOP_BLOCK, OW - ow); i++) {
                        blockB[i] = as_float(intel_sub_group_block_read((
                                const __global uint
                                        *)(&diff_dst1[(ow + i) * OC_BLOCK])));
                    }
#if IC == 3
                    for (int i = 0; i < min(OW_LOOP_BLOCK, OW - ow); i++) {
                        blockC00 = FMA8(
                                blockB[i], TRANSPOSE_8(blockA, i, 0), blockC00);
                    }

#elif IS_DW
                    for (int i = 0; i < min(OW_LOOP_BLOCK, OW - ow); i++) {
                        blockC00 = fma(blockA[i], blockB[i], blockC00);
                    }
#else

                    for (int i = 0; i < min(OW_LOOP_BLOCK, OW - ow); i++) {
                        blockC00 = FMA8(
                                blockB[i], TRANSPOSE_8(blockA, i, 0), blockC00);
                        blockC01 = FMA8(
                                blockB[i], TRANSPOSE_8(blockA, i, 8), blockC01);
                    }
#endif
#if WITH_BIAS == 1
                    for (int i = 0; i < min(OW_LOOP_BLOCK, OW - ow); i++)
                        bias_loc += blockB[i];
#endif
                }
            }
        src += G * IC * ID * IH * IW * MB_BLOCK;
    }

#if WITH_BIAS == 1
    if (do_bias
            && oc * OC_BLOCK + local_x < (IS_DW ? G_WO_PADDING : OC_WO_PADDING))
        atomic_add_global(diff_bias, bias_loc);
#endif

#if IC == 3
    diff_wei += oc * KD * KH * KW * IC * OC_BLOCK + g * OC * IC * KD * KH * KW
            + kd * KH * KW * IC * OC_BLOCK + kh * KW * IC * OC_BLOCK
            + kw * IC * OC_BLOCK;
    for (int i = 0; i < 3; i++)
        atomic_add_global(diff_wei + i * OC_BLOCK + local_x, blockC00[i]);
#elif IS_DW
    diff_wei += oc * KD * KH * KW * OC_BLOCK + kd * KH * KW * OC_BLOCK
            + kh * KW * OC_BLOCK + kw * OC_BLOCK;
    atomic_add_global(diff_wei + local_x, blockC00);
#else
    diff_wei += ic * OC * KD * KH * KW * IC_BLOCK
            + oc * KD * KH * KW * IC_BLOCK * OC_BLOCK
            + kd * KH * KW * IC_BLOCK * OC_BLOCK + kh * KW * IC_BLOCK * OC_BLOCK
            + kw * IC_BLOCK * OC_BLOCK + g * OC * IC * KD * KH * KW;
#if NCHUNK == 1
    intel_sub_group_block_write8((__global uint *)diff_wei, as_uint8(blockC00));
    intel_sub_group_block_write8(
            (__global uint *)(diff_wei + 8 * OC_BLOCK), as_uint8(blockC01));
#else
    for (int i = 0; i < 8; i++)
        atomic_add_global(diff_wei + i * OC_BLOCK + local_x, blockC00[i]);

    for (int i = 0; i < 8; i++)
        atomic_add_global(diff_wei + (8 + i) * OC_BLOCK + local_x, blockC01[i]);
#endif
#endif
#endif
}
#endif
