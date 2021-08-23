/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file bounding_box-inl.cuh
 * \brief bounding box CUDA operators
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_INL_CUH_
#define MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_INL_CUH_
#include <cmath>
#include <cstdio>
#include <mxnet/operator_util.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "./bounding_box-common.h"

namespace mxnet {
namespace op {

template<typename DType>
struct valid_value {
  __host__ __device__ bool operator()(const DType x) {
    return static_cast<bool>(x);
  }
};

template<typename DType, typename FType>
int CopyIf(mshadow::Tensor<gpu, 1, DType> out,
           mshadow::Tensor<gpu, 1, DType> value,
           mshadow::Tensor<gpu, 1, FType> flag) {
  valid_value<FType> pred;
  DType *end_out = thrust::copy_if(thrust::device, value.dptr_, value.dptr_ + value.MSize(),
                                   flag.dptr_, out.dptr_, pred);
  return end_out - out.dptr_;
}

// compute line intersect along either height or width
template<typename DType>
MSHADOW_XINLINE DType Intersect2(const DType *a, const DType b1, const DType b2, int encode) {
  const DType a1 = a[0];
  const DType a2 = a[2];
  DType left, right;
  if (box_common_enum::kCorner == encode) {
    left = a1 > b1 ? a1 : b1;
    right = a2 < b2 ? a2 : b2;
  } else {
    const DType aw = a2 / 2;
    const DType bw = b2 / 2;
    const DType al = a1 - aw;
    const DType ar = a1 + aw;
    const DType bl = b1 - bw;
    const DType br = b1 + bw;
    left = bl > al ? bl : al;
    right = br < ar ? br : ar;
  }
  const DType w = right - left;
  return w > 0 ? w : DType(0);
}

template<typename DType, int N, bool check_class>
__launch_bounds__(512)
__global__ void nms_apply_kernel(const int topk, int32_t *index,
                                 const int32_t *batch_start,
                                 const DType *input,
                                 const DType *areas,
                                 const int num, const int stride,
                                 const int offset_box, const int offset_id,
                                 const float thresh, const bool force,
                                 const int encode, const int start_offset) {
  constexpr int block_size = 512;
  const int start = static_cast<int>(batch_start[blockIdx.x]) + start_offset;
  const int size_of_batch = static_cast<int>(batch_start[blockIdx.x + 1]) - start;
  const int end = min(min(size_of_batch, topk - start_offset), N * block_size);
  __shared__ int s_index[N * block_size];

  for (int i = threadIdx.x; i < end; i += block_size) {
    s_index[i] = static_cast<int>(index[start + i]);
  }

  __syncthreads();
  for (int ref = 0; ref < end; ++ref) {
    const int ref_area_offset = static_cast<int>(s_index[ref]);
    if (ref_area_offset >= 0) {
      const int ref_offset = ref_area_offset * stride + offset_box;
      int ref_id = 0;
      if (check_class) {
        ref_id = static_cast<int>(input[ref_offset - offset_box + offset_id]);
      }
      for (int i = 0; i < N; ++i) {
        const int my_pos = threadIdx.x + i * block_size;
        if (my_pos > ref && my_pos < end && s_index[my_pos] >= 0) {
          const int pos_area_offset = static_cast<int>(s_index[my_pos]);
          const int pos_offset = pos_area_offset * stride + offset_box;
          if (check_class) {
            const int pos_id = static_cast<int>(input[pos_offset - offset_box + offset_id]);
            if (ref_id != pos_id) continue;  // different class
          }
          DType intersect = Intersect(input + ref_offset, input + pos_offset, encode);
          intersect *= Intersect(input + ref_offset + 1, input + pos_offset + 1, encode);
          const DType iou = intersect /
                            (areas[ref_area_offset] + areas[pos_area_offset] - intersect);
          if (iou > thresh) {
            s_index[my_pos] = -1;
          }
        }
      }
      __syncthreads();
    }
  }

  for (int i = threadIdx.x; i < end; i += block_size) {
    index[start + i] = s_index[i];
  }
}

template<typename DType, int N, bool check_class>
__launch_bounds__(512)
__global__ void nms_apply_kernel_rest(const int topk, int32_t *index,
                                 const int32_t *batch_start,
                                 const DType *input,
                                 const DType *areas,
                                 const int num, const int stride,
                                 const int offset_box, const int offset_id,
                                 const float thresh, const bool force,
                                 const int encode, const int start_offset,
                                 const int blocks_per_batch) {
  constexpr int block_size = 512;
  const int batch = blockIdx.x / blocks_per_batch;
  const int start_ref = static_cast<int>(batch_start[batch]) + start_offset;
  const int block_offset = (N + blockIdx.x % blocks_per_batch) * block_size;
  const int start = start_ref + block_offset;

  const int size_of_batch = static_cast<int>(batch_start[batch + 1]) - start;
  const int end = min(size_of_batch, topk - start_offset - block_offset);
  const int my_pos = start + threadIdx.x;
  if (threadIdx.x < end && index[my_pos] >= 0) {
    const int pos_area_offset = static_cast<int>(index[my_pos]);
    const int pos_offset = pos_area_offset * stride + offset_box;
    DType my_box[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      my_box[i] = input[pos_offset + i];
    }
    const DType my_area = areas[pos_area_offset];
    int pos_id = 0;
    if (check_class) {
      pos_id = static_cast<int>(input[pos_offset - offset_box + offset_id]);
    }

    for (int ref = start_ref; ref < start_ref + N * block_size; ++ref) {
      const int ref_area_offset = static_cast<int>(index[ref]);
      if (ref_area_offset >= 0) {
        const int ref_offset = ref_area_offset * stride + offset_box;
        int ref_id = 0;
        if (check_class) {
          ref_id = static_cast<int>(input[ref_offset - offset_box + offset_id]);
          if (ref_id != pos_id) continue;  // different class
        }
        DType intersect = Intersect2(input + ref_offset, my_box[0], my_box[2], encode);
        intersect *= Intersect2(input + ref_offset + 1, my_box[1], my_box[3], encode);
        const DType iou = intersect /
          (areas[ref_area_offset] + my_area - intersect);
        if (iou > thresh) {
          index[my_pos] = -1;
          break;
        }
      }
    }
  }
}

template<typename DType>
void NMSApply(mshadow::Stream<gpu> *s,
              int num_batch, int topk,
              mshadow::Tensor<gpu, 1, int32_t>* sorted_index,
              mshadow::Tensor<gpu, 1, int32_t>* batch_start,
              mshadow::Tensor<gpu, 3, DType>* buffer,
              mshadow::Tensor<gpu, 1, DType>* areas,
              int num_elem, int width_elem,
              int coord_start, int id_index,
              float threshold, bool force_suppress,
              int in_format) {
  using namespace mxnet_op;
  constexpr int THRESHOLD = 1024;
  for (int ref = 0; ref < topk; ref += THRESHOLD) {
    constexpr int block_size = 512;
    constexpr int N = THRESHOLD / block_size;
    auto stream = mshadow::Stream<gpu>::GetStream(s);
    if (!force_suppress && id_index >= 0) {
      nms_apply_kernel<DType, N, true><<<num_batch, block_size, 0, stream>>>(topk,
                                                                      sorted_index->dptr_,
                                                                      batch_start->dptr_,
                                                                      buffer->dptr_,
                                                                      areas->dptr_,
                                                                      num_elem,
                                                                      width_elem,
                                                                      coord_start,
                                                                      id_index,
                                                                      threshold,
                                                                      force_suppress,
                                                                      in_format,
                                                                      ref);
      int blocks_per_batch = (topk - ref - THRESHOLD + block_size - 1)/block_size;
      int blocks = blocks_per_batch  * num_batch;
      if (blocks > 0) {
        nms_apply_kernel_rest<DType, N, true><<<blocks, block_size, 0, stream>>>(topk,
                                                                        sorted_index->dptr_,
                                                                        batch_start->dptr_,
                                                                        buffer->dptr_,
                                                                        areas->dptr_,
                                                                        num_elem,
                                                                        width_elem,
                                                                        coord_start,
                                                                        id_index,
                                                                        threshold,
                                                                        force_suppress,
                                                                        in_format,
                                                                        ref,
                                                                        blocks_per_batch);
      }
    } else {
      nms_apply_kernel<DType, N, false><<<num_batch, block_size, 0, stream>>>(topk,
                                                                       sorted_index->dptr_,
                                                                       batch_start->dptr_,
                                                                       buffer->dptr_,
                                                                       areas->dptr_,
                                                                       num_elem,
                                                                       width_elem,
                                                                       coord_start,
                                                                       id_index,
                                                                       threshold,
                                                                       force_suppress,
                                                                       in_format,
                                                                       ref);
      int blocks_per_batch = (topk - ref - THRESHOLD + block_size - 1)/block_size;
      int blocks = blocks_per_batch  * num_batch;
      if (blocks > 0) {
        nms_apply_kernel_rest<DType, N, false><<<blocks, block_size, 0, stream>>>(topk,
                                                                        sorted_index->dptr_,
                                                                        batch_start->dptr_,
                                                                        buffer->dptr_,
                                                                        areas->dptr_,
                                                                        num_elem,
                                                                        width_elem,
                                                                        coord_start,
                                                                        id_index,
                                                                        threshold,
                                                                        force_suppress,
                                                                        in_format,
                                                                        ref,
                                                                        blocks_per_batch);
      }
    }
  }
}

__launch_bounds__(512)
__global__ void nms_calculate_batch_start_kernel(int32_t * batch_start,
                                                 int32_t * valid_batch_id,
                                                 size_t N,
                                                 int num_batch) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 350
    const int32_t previous = tid > 0 ? __ldg(valid_batch_id + tid - 1) : -1;
    const int32_t my = __ldg(valid_batch_id + tid);
#else
    const int32_t previous = tid > 0 ? valid_batch_id[tid - 1] : -1;
    const int32_t my = valid_batch_id[tid];
#endif
    if (my > previous) {
      for (int32_t current = previous + 1; current <= my; ++current) {
        batch_start[current] = tid;
      }
    }
    if (tid == N - 1) {
      for (int32_t current = my + 1; current <= num_batch; ++current) {
        batch_start[current] = tid + 1;
      }
    }
  }
}

inline void NMSCalculateBatchStart(mshadow::Stream<gpu> *s,
                                   mshadow::Tensor<gpu, 1, int32_t>* batch_start,
                                   mshadow::Tensor<gpu, 1, int32_t>* valid_batch_id,
                                   int num_batch) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  auto stream = mshadow::Stream<gpu>::GetStream(s);
  constexpr int block_size = 512;
  const int num_elements = valid_batch_id->size(0);
  const int blocks = (num_elements + block_size - 1) / block_size;
  nms_calculate_batch_start_kernel<<<blocks, block_size, 0, stream>>>(batch_start->dptr_,
                                                                      valid_batch_id->dptr_,
                                                                      num_elements,
                                                                      num_batch);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BOUNDING_BOX_INL_CUH_
