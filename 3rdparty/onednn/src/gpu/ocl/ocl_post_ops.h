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

#ifndef GPU_OCL_OCL_POST_OPS_H
#define GPU_OCL_OCL_POST_OPS_H

#ifndef SUB_GROUP_SIZE
#define SUB_GROUP_SIZE get_sub_group_size()
#endif

#if WITH_POST_OP

#if !WITH_ELTWISE
#undef WITH_ELTWISE
#define WITH_ELTWISE 1
#endif

#include "gpu/ocl/ocl_eltwise.h"
#include "gpu/ocl/ocl_types.h"

float fwd_Xnary(
        int algorithm, float x, float y, float alpha, float beta, float scale) {
    switch (algorithm) {
        // binary
        case BINARY_ADD: return x + y; break;
        case BINARY_MUL: return x * y; break;
        case BINARY_MIN: return x > y ? y : x; break;
        case BINARY_MAX: return x > y ? x : y; break;
        case BINARY_DIV: return x / y; break;

        // unary
        default:
            return fwd_eltwise_common(algorithm, x, alpha, beta, scale);
            break;
    }
}

#define CONV_BIN_ARG_TO_FLOAT(idx, bin_arg_val) \
    ({ \
        float ret_val; \
        if (CONCAT3(PO_, idx, _BIN_ARG_DT_IS_BF16)) \
            ret_val = cvt_bf16_to_f32(bin_arg_val); \
        else \
            ret_val = convert_float(bin_arg_val); \
\
        ret_val; \
    })

#define FWD_XNARY_GENERIC_DT(algorithm, result, result_elem_dt, arg0, \
        arg0_elem_dt, arg1, arg1_elem_dt, alpha, beta, scale) \
    { \
        const unsigned arg0_len = sizeof(arg0) / sizeof(arg0_elem_dt); \
        const unsigned arg1_len = sizeof(arg1) / sizeof(arg1_elem_dt); \
        const unsigned out_len = max(arg0_len, arg1_len); \
        result_elem_dt *res_ptr = (result_elem_dt *)(&result); \
        arg0_elem_dt *arg0_ptr = (arg0_elem_dt *)(&arg0); \
        arg1_elem_dt *arg1_ptr = (arg1_elem_dt *)(&arg1); \
        for (unsigned idx = 0; idx < out_len; ++idx) { \
            if (arg0_len == 1 && arg1_len == 1) { \
                *res_ptr = fwd_Xnary(algorithm, convert_float(*arg0_ptr), \
                        convert_float(*arg1_ptr), alpha, beta, scale); \
            } else if (arg0_len == 1) { \
                res_ptr[idx] = fwd_Xnary(algorithm, convert_float(*arg0_ptr), \
                        convert_float(arg1_ptr[idx]), alpha, beta, scale); \
            } else if (arg1_len == 1) { \
                res_ptr[idx] \
                        = fwd_Xnary(algorithm, convert_float(arg0_ptr[idx]), \
                                convert_float(*arg1_ptr), alpha, beta, scale); \
            } else { \
                res_ptr[idx] = fwd_Xnary(algorithm, \
                        convert_float(arg0_ptr[idx]), \
                        convert_float(arg1_ptr[idx]), alpha, beta, scale); \
            } \
        } \
    }

#define FMA_BLOCK( \
        block_size, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b) \
    unroll_for(; nof_elems >= block_size; acc_ptr += block_size, \
               a_ptr += block_size, nof_elems -= block_size) { \
        CONCAT2(acc_elem_dt, block_size) \
        a_conv = CONCAT3(convert_, acc_elem_dt, block_size)( \
                *((CONCAT2(a_elem_dt, block_size) *)a_ptr)); \
        *((CONCAT2(acc_elem_dt, block_size) *)acc_ptr) = fma( \
                a_conv, b, *((CONCAT2(acc_elem_dt, block_size) *)acc_ptr)); \
    }

#define FMA_MIXED(acc_nof_elems, a, a_elem_dt, b, acc, acc_elem_dt) \
    { \
        unsigned nof_elems = acc_nof_elems; \
        a_elem_dt *a_ptr = (a_elem_dt *)(&a); \
        acc_elem_dt *acc_ptr = (acc_elem_dt *)(&acc); \
        FMA_BLOCK(8, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b); \
        FMA_BLOCK(4, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b); \
        FMA_BLOCK(2, nof_elems, acc_ptr, acc_elem_dt, a_ptr, a_elem_dt, b); \
        if (nof_elems == 1) { *acc_ptr += (*a_ptr) * b; } \
    }

#define X_NELEMS(x) ({ x / SUB_GROUP_SIZE; })
#define REPLICATE_DATA( \
        dest_ptr, dest_size, x0_s, x1_s, x2_s, x3_s, x4_s, x5_s) \
    { \
        const unsigned copy_size \
                = x0_s * X_NELEMS(x1_s) * x2_s * x3_s * x4_s * x5_s; \
        unroll_for(unsigned fid = copy_size; fid < dest_size; ++fid) { \
            *(dest_ptr + fid) = *(dest_ptr + (fid % copy_size)); \
        } \
    }

#define APPLY_PO_BINARY(idx, accumulator, acc_elem_dt, x0, x0_s, x1, x1_s, x2, \
        x2_s, x3, x3_s, x4, x4_s, x5, x5_s) \
    { \
        const unsigned bin_arg_size \
                = sizeof(accumulator) / sizeof(acc_elem_dt); \
        float bin_arg[bin_arg_size]; \
        unroll_for(unsigned x0_idx = x0, bin_arg_offset = 0; \
                   x0_idx < x0 + x0_s; ++x0_idx) { \
            unroll_for(unsigned x1_idx = x1; x1_idx < x1 + x1_s; ++x1_idx) { \
                unroll_for(unsigned x2_idx = x2; x2_idx < x2 + x2_s; \
                           ++x2_idx) { \
                    unroll_for(unsigned x3_idx = x3; x3_idx < x3 + x3_s; \
                               ++x3_idx) { \
                        unroll_for(unsigned x4_idx = x4; x4_idx < x4 + x4_s; \
                                   ++x4_idx) { \
                            unroll_for(unsigned x5_idx = x5; \
                                       x5_idx < x5 + x5_s; \
                                       ++x5_idx, ++bin_arg_offset) { \
                                const int bin_arg_glob_off \
                                        = OFF_MD(CONCAT3(PO_, idx, _BIN_ARG), \
                                                x0_idx \
                                                        % CONCAT3(PO_, idx, \
                                                                _BIN_ARG_D0), \
                                                x1_idx \
                                                        % CONCAT3(PO_, idx, \
                                                                _BIN_ARG_D1), \
                                                x2_idx \
                                                        % CONCAT3(PO_, idx, \
                                                                _BIN_ARG_D2), \
                                                x3_idx \
                                                        % CONCAT3(PO_, idx, \
                                                                _BIN_ARG_D3), \
                                                x4_idx \
                                                        % CONCAT3(PO_, idx, \
                                                                _BIN_ARG_D4), \
                                                x5_idx \
                                                        % CONCAT3(PO_, idx, \
                                                                _BIN_ARG_D5)); \
                                bin_arg[bin_arg_offset] \
                                        = CONV_BIN_ARG_TO_FLOAT(idx, \
                                                CONCAT3(po_, idx, _binary_arg) \
                                                        [bin_arg_glob_off]); \
                            } \
                        } \
                    } \
                } \
            } \
        } \
        float *bin_arg_ptr = &bin_arg[0]; \
        REPLICATE_DATA(bin_arg_ptr, bin_arg_size, x0_s, x1_s, x2_s, x3_s, \
                x4_s, x5_s); \
        FWD_XNARY_GENERIC_DT(CONCAT3(PO_, idx, _ALG), accumulator, \
                acc_elem_dt, accumulator, acc_elem_dt, bin_arg, float, 0.0f, \
                0.0f, 0.0f); \
    }

#define APPLY_PO_SUM(idx, accumulator, acc_elem_dt, sum_src, sum_elem_dt) \
    { \
        unsigned acc_size = sizeof(accumulator) / sizeof(acc_elem_dt); \
        FMA_MIXED(acc_size, sum_src, sum_elem_dt, \
                CONCAT3(po_, idx, _sum_scale), accumulator, acc_elem_dt); \
    }

#define APPLY_PO_ELTWISE(idx, accumulator, acc_elem_dt) \
    { \
        FWD_XNARY_GENERIC_DT(CONCAT3(PO_, idx, _ALG), accumulator, \
                acc_elem_dt, accumulator, acc_elem_dt, accumulator, \
                acc_elem_dt, CONCAT3(po_, idx, _eltwise_alpha), \
                CONCAT3(po_, idx, _eltwise_beta), \
                CONCAT3(po_, idx, _eltwise_scale)); \
    }

#define APPLY_PO_STAGE(idx, accumulator, acc_elem_dt, sum_src, sum_elem_dt, \
        x0, x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s) \
    switch (CONCAT3(PO_, idx, _KIND)) { \
        case PO_BINARY: \
            APPLY_PO_BINARY(idx, accumulator, acc_elem_dt, x0, x0_s, x1, x1_s, \
                    x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
            break; \
        case PO_ELTWISE: APPLY_PO_ELTWISE(idx, accumulator, acc_elem_dt); \
                break; \
        case PO_SUM: \
            APPLY_PO_SUM(idx, accumulator, acc_elem_dt, sum_src, sum_elem_dt); \
            break; \
    }

#define APPLY_POST_OPS(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s) \
    { \
        APPLY_PO_STAGE(0, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        APPLY_PO_STAGE(1, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        APPLY_PO_STAGE(2, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        APPLY_PO_STAGE(3, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        APPLY_PO_STAGE(4, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        APPLY_PO_STAGE(5, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        APPLY_PO_STAGE(6, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        APPLY_PO_STAGE(7, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        APPLY_PO_STAGE(8, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
        APPLY_PO_STAGE(9, accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
                x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s); \
    }

#else

#define APPLY_POST_OPS(accumulator, acc_elem_dt, sum_src, sum_elem_dt, x0, \
        x0_s, x1, x1_s, x2, x2_s, x3, x3_s, x4, x4_s, x5, x5_s) \
    {}

#endif // WITH_POST_OP

#endif
