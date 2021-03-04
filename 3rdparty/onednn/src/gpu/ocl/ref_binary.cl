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

#include "gpu/ocl/ocl_post_ops.h"
#include "gpu/ocl/ocl_types.h"

#undef DST_OFF
#define SRC0_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC0, x0, x1, x2, x3, x4, x5)
#define SRC1_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC1, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)

#if IS_TENSOR_OP && IS_DENSE && IS_SAME_MD && !WITH_BINARY_POST_OP

KERNEL_ATTR
__kernel void ref_binary(__global DATA_T *src0, __global DATA_T *src1,
        __global DST_DATA_T *dst POST_OP_ARGS, float src0_scale,
        float src1_scale) {
    int off = GWS_GET_IDX();

    POST_OP_DATA_T tmp_src0 = DATA_TO_REF(src0[off]);
    POST_OP_DATA_T tmp_src1 = DATA_TO_REF(src1[off]);
    POST_OP_DATA_T d = 0;

#if WITH_SRC0_SCALE
    tmp_src0 = tmp_src0 * src0_scale;
#endif
#if WITH_SRC1_SCALE
    tmp_src1 = tmp_src1 * src1_scale;
#endif

#if IS_ADD
    d = tmp_src0 + tmp_src1;
#elif IS_MUL
    d = tmp_src0 * tmp_src1;
#elif IS_MAX
    d = max(tmp_src0, tmp_src1);
#elif IS_MIN
    d = min(tmp_src0, tmp_src1);
#elif IS_DIV
    d = tmp_src0 / tmp_src1;
#endif

    POST_OP_DATA_T dst_data;
#if WITH_SUM
    dst_data = DATA_TO_REF(dst[off]);
#endif

    APPLY_POST_OPS(d, POST_OP_DATA_T, dst_data, POST_OP_DATA_T, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0);
    dst[off] = TO_DST(d);
}
#else
KERNEL_ATTR
__kernel void ref_binary(__global SRC0_DATA_T *src0, __global SRC1_DATA_T *src1,
        __global DST_DATA_T *dst POST_OP_ARGS, float src0_scale,
        float src1_scale) {

    // since gws = no. of total elems in A, id will be the logical offset
    int dims0[6] = {0};
    dims0[0] = GWS_GET_D0();
    dims0[1] = GWS_GET_D1();
    dims0[2] = GWS_GET_D2();
    dims0[3] = GWS_GET_D3();
    dims0[4] = GWS_GET_D4();
    dims0[5] = GWS_GET_D5();
    int d1_block = GWS_GET_D1_BLOCK();
    int d1_init = GWS_GET_D1();
    int dims0_po[6]
            = {dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]};

    int src1_off = 0;
    int src0_off = SRC0_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
    int dst_off = DST_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
#if TENSOR_OP
    src1_off = SRC1_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
#else
    int bcast_dims[] = {BCAST_DIM0, BCAST_DIM1, BCAST_DIM2, BCAST_DIM3,
            BCAST_DIM4, BCAST_DIM5};

    for (int d = 0; d < NDIMS; ++d) {
        dims0[d] *= (!bcast_dims[d]);
    }
    src1_off = SRC1_OFF(
            dims0[0], dims0[1], dims0[2], dims0[3], dims0[4], dims0[5]);
#endif

    // SRC1_D1 = IC for SRC1, using the dispatch ap
    int block_size
            = dims0[1] + d1_block > SRC1_D1 && min(d1_block, SRC0_D1) <= SRC1_D1
            ? (SRC1_D1 % d1_block)
            : d1_block;

    for (int ic = 0; ic < block_size; ++ic) {
        // using tmp vars to handle float calculations for bf16 datatypes
        // DATA_TO_REF is a macro defined in ocl_types.h according to datatype
        POST_OP_DATA_T tmp_src0 = DATA_TO_REF(src0[src0_off]);
        POST_OP_DATA_T tmp_src1 = DATA_TO_REF(src1[src1_off]);
        POST_OP_DATA_T d = 0;

#if WITH_SRC0_SCALE
        tmp_src0 = tmp_src0 * src0_scale;
#endif
#if WITH_SRC1_SCALE
        tmp_src1 = tmp_src1 * src1_scale;
#endif

#if IS_ADD
        d = tmp_src0 + tmp_src1;
#elif IS_MUL
        d = tmp_src0 * tmp_src1;
#elif IS_MAX
        d = max(tmp_src0, tmp_src1);
#elif IS_MIN
        d = min(tmp_src0, tmp_src1);
#elif IS_DIV
        d = tmp_src0 / tmp_src1;
#endif

        POST_OP_DATA_T dst_data;
        if (DST_D1 == DST_PD1 || d1_init + ic < DST_D1) {
#if WITH_SUM
            dst_data = DATA_TO_REF(dst[dst_off]);
#endif
            APPLY_POST_OPS(d, POST_OP_DATA_T, dst_data, POST_OP_DATA_T,
                    dims0_po[0], 1, dims0_po[1], 1, dims0_po[2], 1, dims0_po[3],
                    1, dims0_po[4], 1, dims0_po[5], 1);

            dst[dst_off] = TO_DST(d);
        }

#if USE_UNROLL_16B || SRC0_UNROLL_16B
        src0_off++;
        dst_off++;
        ++dims0_po[1];
        if (USE_UNROLL_16B && (SRC1_D1 > 1)) {
            src1_off++;
        } else if (SRC0_UNROLL_16B && (SRC1_D1 > 1)) {
            src1_off += SRC1_S1_0; // Equilvalent stride in plain format
        }
#endif
    }
}
#endif
