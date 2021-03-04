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

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)

#define OFF(dim, idx) \
    (dim % CONCAT2(DATA_B, idx)) * CONCAT2(DATA_SB, idx) \
            + (dim / CONCAT2(DATA_B, idx)) * CONCAT2(DATA_S, idx)

#if SOFTMAX_AXIS_IDX == 0
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(softmax_dim, 0) + OFF(dim0, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 1
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(softmax_dim, 1) + OFF(dim1, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 2
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(softmax_dim, 2) + OFF(dim2, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 3
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(softmax_dim, 3) \
            + OFF(dim3, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 4
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) \
            + OFF(softmax_dim, 4) + OFF(dim4, 5)
#elif SOFTMAX_AXIS_IDX == 5
#define DATA_OFF(dim0, dim1, dim2, dim3, dim4, softmax_dim) \
    OFF(dim0, 0) + OFF(dim1, 1) + OFF(dim2, 2) + OFF(dim3, 3) + OFF(dim4, 4) \
            + OFF(softmax_dim, 5)
#else
#error unsupported softmax dimension
#endif

#if IS_FWD
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))

__kernel void
ref_softmax_fwd_generic(__global DATA_T *src, __global DATA_T *dst) {

    const int dim[] = {
            (get_global_id(0) / GROUP_SIZE) % BLOCK_0,
            get_global_id(1) % BLOCK_1,
            get_global_id(2) % BLOCK_2,
            (get_global_id(0) / GROUP_SIZE) / BLOCK_0,
            get_global_id(1) / BLOCK_1,
            get_global_id(2) / BLOCK_2,
    };
    int local_id = get_local_id(0);

    // SOFTMAX_AXIS is the size of axis around which softmax operation is
    // performed

    // begin and end indices calculated according to thread's id
    int begin = local_id * (SOFTMAX_AXIS / GROUP_SIZE);
    int end = (local_id == GROUP_SIZE - 1)
            ? SOFTMAX_AXIS
            : (local_id + 1) * (SOFTMAX_AXIS / GROUP_SIZE);

    // initializing max_ to first value of subgroup
    int start_idx = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], begin);
    DEF_ACC_DATA_T max_ = TO_DEF_ACC_DATA_T(src[start_idx]);
    DEF_ACC_DATA_T denom_ = DATA_ZERO;

    // finding max value for each sub_group
    for (int i = begin; i < end; ++i) {
        size_t data_off = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);
        DEF_ACC_DATA_T temp = TO_DEF_ACC_DATA_T(src[data_off]);
        max_ = temp > max_ ? temp : max_;
    }

    // reduce using work_group_reduce if no. of subgroups > 1, for e.g.
    // if group_size is 32, there will be 2 sub-groups (size of each sub-group
    // is 16 which is an optimal value)
#if GROUP_SIZE == SUB_GROUP_SIZE
    max_ = sub_group_reduce_max(max_);
#else
    max_ = work_group_reduce_max(max_);
#endif

    // updating dst tensor and accumulating denom for last step
    for (int i = begin; i < end; ++i) {
        size_t data_off = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);
        DEF_ACC_DATA_T temp = TO_DEF_ACC_DATA_T(src[data_off]);
#if LOGSOFTMAX
        denom_ += exp(temp - max_);
#else
        dst[data_off] = TO_DATA_T(exp(temp - max_));
        denom_ += TO_DEF_ACC_DATA_T(dst[data_off]);
#endif
    }

#if GROUP_SIZE == SUB_GROUP_SIZE
    denom_ = sub_group_reduce_add(denom_);
#else
    denom_ = work_group_reduce_add(denom_);
#endif

#if LOGSOFTMAX
    denom_ = log(denom_);

#else
    denom_ = 1.0 / denom_;
#endif

    for (int i = begin; i < end; ++i) {
        size_t data_off = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);
#if LOGSOFTMAX
        DEF_ACC_DATA_T temp = TO_DEF_ACC_DATA_T(src[data_off]);
        dst[data_off] = TO_DATA_T(temp - max_ - denom_);
#else
        DEF_ACC_DATA_T temp = TO_DEF_ACC_DATA_T(dst[data_off]);
        dst[data_off] = TO_DATA_T(temp * denom_);
#endif
    }
}

#endif

#if IS_BWD
__kernel void ref_softmax_bwd_generic(__global DATA_T *dst,
        __global DATA_T *diff_src, __global DATA_T *diff_dst) {
    const int dim[] = {
            get_global_id(0) % BLOCK_0,
            get_global_id(1) % BLOCK_1,
            get_global_id(2) % BLOCK_2,
            get_global_id(0) / BLOCK_0,
            get_global_id(1) / BLOCK_1,
            get_global_id(2) / BLOCK_2,
    };

    DEF_ACC_DATA_T sbr = 0.f;
    for (int i = 0; i < SOFTMAX_AXIS; ++i) {
        size_t idx = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);
        DEF_ACC_DATA_T g_temp = TO_DEF_ACC_DATA_T(diff_dst[idx]);
#if LOGSOFTMAX
        sbr += g_temp;
#else
        DEF_ACC_DATA_T y_temp = TO_DEF_ACC_DATA_T(dst[idx]);
        sbr += g_temp * y_temp;
#endif
    }

    for (int i = 0; i < SOFTMAX_AXIS; ++i) {
        size_t idx = DATA_OFF(dim[0], dim[1], dim[2], dim[3], dim[4], i);
#if LOGSOFTMAX
        diff_src[idx] = TO_DATA_T(TO_DEF_ACC_DATA_T(diff_dst[idx])
                - exp(TO_DEF_ACC_DATA_T(dst[idx])) * sbr);
#else
        DEF_ACC_DATA_T inner_data = TO_DEF_ACC_DATA_T(diff_dst[idx]) - sbr;
        diff_src[idx] = TO_DATA_T(TO_DEF_ACC_DATA_T(dst[idx]) * inner_data);
#endif
    }
}
#endif
