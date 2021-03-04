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

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#if ID > 1
#define CASE_3D 1
#else
#define CASE_3D 0
#endif

#define DIV_UP(a, b) (((a) + (b)-1) / (b))
#define RND_UP(a, b) (DIV_UP(a, b) * (b))

#if BWD_WEIGHTS == 1

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

inline float read_ic_block(const __global float *ptr, int off) {
#if (IS_DW ? G : IC) % IC_BLOCK != 0
    int tail = (IS_DW ? G : IC) - off;
    if (tail < IC_BLOCK) {
        const int local_x = get_local_id(0);
        return (local_x < tail) ? ptr[local_x] : 0.0f;
    }
#endif
    return as_float(intel_sub_group_block_read((const __global uint *)ptr));
}

inline float read_oc_block(const __global float *ptr, int off) {
#if (IS_DW ? G : OC_WO_PADDING) % OC_BLOCK != 0
    int tail = (IS_DW ? G : OC_WO_PADDING) - off;
    if (tail < OC_BLOCK) {
        const int local_x = get_local_id(0);
        return (local_x < tail) ? ptr[local_x] : 0.0f;
    }
#endif
    return as_float(intel_sub_group_block_read((const __global uint *)ptr));
}

__attribute__((reqd_work_group_size(LWS_0, LWS_1, LWS_2))) // attr:no-format
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) // attr:no-format
__kernel void
gen9_conv_nhwc_bwd_weights(__global float *src,
        volatile __global atomic_float *diff_wei,
        volatile __global atomic_float *diff_bias, __global float *diff_dst) {
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
    const int icb = icb_ocb % DIV_UP(IC, ICB);
    const int ocb = icb_ocb / DIV_UP(IC, ICB);

    const int ic_padded = RND_UP(IC, IC_BLOCK);
    const int oc_padded = RND_UP(OC, OC_BLOCK);

#if IS_DW
    const int g = 0;
    const int oc = get_group_id(0);
    const int ic = oc;
#else
    const int g_ic_oc = get_global_id(0);
    const int g = g_ic_oc / (oc_padded * DIV_UP(IC, IC_BLOCK));
    const int io = g_ic_oc % (oc_padded * DIV_UP(IC, IC_BLOCK));
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

    src += mb * ID * IH * IW * G * IC;
    src += g * IC + ic * IC_BLOCK;

    diff_dst += g * OC_WO_PADDING + oc * OC_BLOCK;

#if WITH_BIAS == 1
    diff_bias += g * OC_WO_PADDING + oc * OC_BLOCK + local_x;
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
                = diff_dst + omb * OD * OH * OW * G * OC_WO_PADDING;

        for (int od = od_beg; od < min(od_beg + ODB, OD); od++)
            for (int oh = oh_beg; oh < min(oh_beg + OHB, OH); oh++) {
                const __global float *diff_dst1 = diff_dst1_
                        + (od * OH * OW + oh * OW) * G * OC_WO_PADDING;
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
                                ow += OW_BLOCK) {
                            float8 blockB;
                            for (int i = 0; i < OW_BLOCK; i++) {
                                if (ow + i >= OW) {
                                    blockB[i] = 0.0;
                                } else {
                                    blockB[i] = read_oc_block(
                                            &diff_dst1[(ow + i) * G
                                                    * OC_WO_PADDING],
                                            oc * OC_BLOCK);
                                }
                            }

                            for (int i = 0; i < OW_BLOCK; i++)
                                bias_loc += blockB[i];
                        }
                    }
#endif
                    continue;
                }

                for (int ow = ow_beg; ow < ow_beg + OWB; ow += OW_BLOCK) {
                    const int id = od * SD - PD + kd * (1 + DD);
                    const int ih = oh * SH - PH + kh * (1 + DH);
                    const int iw = ow * SW - PW + kw * (1 + DW);
                    __global float *src1
                            = src + (id * IH * IW + ih * IW + iw) * G * IC;

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

                    float8 blockA, blockB;
#if IC == 3
                    if (local_x < IC) {
                        for (int i = 0; i < OW_BLOCK; i++) {
                            if (iw + i * SW < 0 || iw + i * SW >= IW) {
                                blockA[i] = 0;
                            } else {
                                blockA[i] = src1[i * SW * G * IC + local_x];
                            }
                        }
                    } else {
                        blockA = 0.0f;
                    }
#else
                    __attribute__((opencl_unroll_hint(8))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        if (iw + i < 0 || iw + i * SW >= IW) {
                            blockA[i] = 0;
                        } else {
                            blockA[i] = read_ic_block(
                                    &src1[i * SW * G * IC], ic * IC_BLOCK);
                        }
                    }
#endif
                    __attribute__((opencl_unroll_hint(8))) // attr:no-format
                    for (int i = 0; i < OW_BLOCK; i++) {
                        if (ow + i >= OW) {
                            blockB[i] = 0.0;
                        } else {
                            blockB[i] = read_oc_block(
                                    &diff_dst1[(ow + i) * G * OC_WO_PADDING],
                                    oc * OC_BLOCK);
                        }
                    }

#if IC == 3
                    MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB, 0);
#elif IS_DW
                    for (int i = 0; i < OW_BLOCK; i++) {
                        blockC00 = fma(blockA[i], blockB[i], blockC00);
                    }
#else
                    MULTIPLY_BLOCKS_8x8(blockC00, blockA, blockB, 0);
                    MULTIPLY_BLOCKS_8x8(blockC01, blockA, blockB, 8);
#endif
#if WITH_BIAS == 1
                    for (int i = 0; i < 8; i++)
                        bias_loc += blockB[i];
#endif
                }
            }
        src += ID * IH * IW * G * IC;
    }

#if WITH_BIAS == 1
    if (do_bias && oc * OC_BLOCK + local_x < (IS_DW ? G : OC_WO_PADDING))
        atomic_add_global(diff_bias, bias_loc);
#endif

#if IC == 3
    diff_wei += g * oc_padded * ic_padded * KD * KH * KW;
    diff_wei += oc * KD * KH * KW * ic_padded * OC_BLOCK;
    diff_wei += (kd * KH * KW + kh * KW + kw) * ic_padded * OC_BLOCK;
    for (int i = 0; i < 3; i++)
        atomic_add_global(diff_wei + i * OC_BLOCK + local_x, blockC00[i]);
#elif IS_DW
    diff_wei += oc * KD * KH * KW * OC_BLOCK;
    diff_wei += (kd * KH * KW + kh * KW + kw) * OC_BLOCK;
    atomic_add_global(diff_wei + local_x, blockC00);
#else
    diff_wei += g * ic_padded * oc_padded * KD * KH * KW;
    diff_wei += ic * oc_padded * KD * KH * KW * IC_BLOCK;
    diff_wei += oc * KD * KH * KW * IC_BLOCK * OC_BLOCK;
    diff_wei += (kd * KH * KW + kh * KW + kw) * IC_BLOCK * OC_BLOCK;

    for (int i = 0; i < 8; i++)
        atomic_add_global(diff_wei + i * OC_BLOCK + local_x, blockC00[i]);

    for (int i = 0; i < 8; i++)
        atomic_add_global(diff_wei + (8 + i) * OC_BLOCK + local_x, blockC01[i]);
#endif
}
#endif
