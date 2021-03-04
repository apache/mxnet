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

__kernel void gen9_gemm_copy(long m, long n, __global SRC_DATA_T *a,
        long offseta, long lda, DATA_T alpha, __global DATA_T *b,
        long offsetb) {
    int idx = get_group_id(0);
    int idy = get_group_id(1) * COPY_UNROLL;
    int i;

#ifdef USE_TRANS
    offseta += (idy + idx * lda); // Transpose Copy
#else
    offseta += (idx + idy * lda); // Non-Transpose Copy
#endif

    offsetb += (idy * m + idx * COPY_UNROLL);

    n -= idy;

    for (i = 0; i < COPY_UNROLL; i++) {
        b[offsetb] = (i < n) ? (alpha * SRC_TO_REF(a[offseta])) : DATA_ZERO;

#ifdef USE_TRANS
        offseta++;
#else
        offseta += lda;
#endif

        offsetb++;
    }
}
