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

#define offset(mb, a, b, stride_mb, stride_a, stride_b) \
    ((mb) * (stride_mb) + (a) * (stride_a) + (b) * (stride_b))

__kernel void ref_matmul(__global SRC_DATA_T *A, __global WEI_DATA_T *B,
        __global DST_DATA_T *C, __global BIA_DATA_T *bia, __global int *a0,
        __global int *b0, __global int *c0, __global float *scales,
        long scale_stride, long MB, long M, long N, long K, long bia_stride_mb,
        long bia_stride_m, long bia_stride_n, long a_stride_mb, long a_stride_m,
        long a_stride_k, long b_stride_mb, long b_stride_k, long b_stride_n,
        long c_stride_mb, long c_stride_m, long c_stride_n POST_OP_ARGS) {

    int n = get_global_id(1);
    int mb = get_global_id(2);

    for (long m = 0; m < M; ++m) {
        ACC_DATA_T acc = 0;
        for (long k = 0; k < K; ++k) {
            acc += TO_ACC(SRC_TO_REF(A[offset(mb, m, k, a_stride_mb, a_stride_m,
                                  a_stride_k)])
                           - a0[0])
                    * TO_ACC(WEI_TO_REF(B[offset(mb, k, n, b_stride_mb,
                                     b_stride_k, b_stride_n)])
                            - b0[0]);
        }

        int c_off = offset(mb, m, n, c_stride_mb, c_stride_m, c_stride_n);
#if WITH_BIAS || NON_DEFAULT_ATTRS
        POST_OP_DATA_T temp = (POST_OP_DATA_T)acc;
#if WITH_BIAS
        int b_off = offset(mb, m, n, bia_stride_mb, bia_stride_m, bia_stride_n);
        temp += bia[b_off];
#endif
        temp *= scales[scale_stride * n];

        float dst_data;
#if WITH_SUM
        dst_data = convert_float(DATA_TO_REF(C[c_off]));
#endif

        const unsigned po_mb = DST_NDIMS > 2 ? mb : m;
        const unsigned po_oc = DST_NDIMS == 3 ? m : n;
        float po_acc = convert_float(temp);
        APPLY_POST_OPS(po_acc, float, dst_data, float, po_mb, 1, po_oc, 1, 0, 1,
                0, 1, 0, 1, 0, 1);

        po_acc += c0[0];
        C[c_off] = TO_DST(po_acc);
#else
        C[c_off] = TO_DST(acc);
#endif
    }
}
