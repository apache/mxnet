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

void get_strides(int mask, long dim0, long dim1, long dim2, long *str0,
        long *str1, long *str2) {
    int is_3d = dim0 > 1;
    long dims[3];
    dims[0] = (is_3d && mask & (1 << 0)) ? dim0 : 1;
    dims[1] = mask & (1 << 1) ? dim1 : 1;
    dims[2] = mask & (1 << 2) ? dim2 : 1;

    *str0 = dims[0] == 1 ? 0 : dims[1] * dims[2];
    *str1 = dims[1] == 1 ? 0 : 1;
    *str2 = dims[2] == 1 ? 0 : dims[1];
}

__kernel void ref_gemm(__global A_DATA_T *a, __global B_DATA_T *b,
        __global C_DATA_T *c, __global BIAS_DATA_T *bias, long offset_a0,
        long offset_b0, long offset_c0, long offset_bias0, int transa,
        int transb, long MB, long M, long N, long K, long stride_a,
        long stride_b, long stride_c, long lda, long ldb, long ldc,
        float eltwise_alpha, float eltwise_beta, float eltwise_scale,
        int bias_mask, __global int *a0, __global int *b0, __global int *c0,
        int c0_mask, __global float *scales, long scale_stride, float beta) {

    int n = get_global_id(1);
    int mb = get_global_id(2);

#if WITH_BIAS
    bias += offset_bias0;

    long b_strides[3];
    get_strides(
            bias_mask, MB, M, N, &b_strides[0], &b_strides[1], &b_strides[2]);
#endif

    a += offset_a0;
    b += offset_b0;
    c += offset_c0;

    long c0_strides[3];
    get_strides(
            c0_mask, MB, M, N, &c0_strides[0], &c0_strides[1], &c0_strides[2]);

    for (long m = 0; m < M; ++m) {
        ACC_DATA_T acc = 0;
        for (long k = 0; k < K; ++k) {
            long off_a = mb * stride_a + (transa ? m * lda + k : k * lda + m);
            long off_b = mb * stride_b + (transb ? k * ldb + n : n * ldb + k);
            acc += TO_ACC(A_TO_REF(a[off_a]) - a0[0])
                    * TO_ACC(B_TO_REF(b[off_b]) - b0[0]);
        }

        long off_c = mb * stride_c + n * ldc + m;
#if WITH_BIAS || NON_DEFAULT_ATTRS
        POST_OP_DATA_T temp = (POST_OP_DATA_T)acc;
#if WITH_BIAS
        long off_bias = mb * b_strides[0] + m * b_strides[1] + n * b_strides[2];
        temp += bias[off_bias];
#endif
        temp *= scales[scale_stride * n];
#if WITH_SUM
        temp += (POST_OP_DATA_T)(beta * C_TO_REF(c[off_c]));
#endif
#if WITH_ELTWISE
        temp = fwd_eltwise(temp, eltwise_alpha, eltwise_beta, eltwise_scale);
#endif
        long off_c0
                = mb * c0_strides[0] + m * c0_strides[1] + n * c0_strides[2];
        temp += c0[off_c0];
        c[off_c] = TO_C(temp);
#else
        c[off_c] = TO_C(acc);
#endif
    }
}
