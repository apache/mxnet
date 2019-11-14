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
  *  Copyright (c) 2017 by Contributors
  * \file bounding_box.cu
  * \brief Bounding box util functions and operators
  * \author Joshua Zhang
  */

#include <cub/cub.cuh>

#include "./bounding_box-inl.cuh"
#include "./bounding_box-inl.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

namespace {

using mshadow::Tensor;
using mshadow::Stream;

template <typename DType>
struct TempWorkspace {
  index_t scores_temp_space;
  DType* scores;
  index_t scratch_space;
  uint8_t* scratch;
  index_t buffer_space;
  DType* buffer;
  index_t nms_scratch_space;
  uint32_t* nms_scratch;
  index_t indices_temp_spaces;
  index_t* indices;
};

inline index_t ceil_div(index_t x, index_t y) {
  return (x + y - 1) / y;
}

inline index_t align(index_t x, index_t alignment) {
  return ceil_div(x, alignment)  * alignment;
}

template <typename DType>
__global__ void FilterAndPrepareAuxDataKernel(const DType* data, DType* out, DType* scores,
                                               index_t num_elements_per_batch,
                                               const index_t element_width,
                                               const index_t N,
                                               const float threshold,
                                               const int id_index, const int score_index,
                                               const int background_id) {
  index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool first_in_element = (tid % element_width == 0);
  index_t start_of_my_element = tid - (tid % element_width);

  if (tid < N) {
    DType my_score = data[start_of_my_element + score_index];
    bool filtered_out = my_score <= threshold;
    if (id_index != -1 && background_id != -1) {
      DType my_id = data[start_of_my_element + id_index];
      filtered_out = filtered_out || (my_id == background_id);
    }
    if (!filtered_out) {
      out[tid] = data[tid];
    } else {
      out[tid] = -1;
      my_score = -1;
    }

    if (first_in_element) {
      index_t offset = tid / element_width;
      scores[offset] = my_score;
    }
  }
}

template <typename DType>
void FilterAndPrepareAuxData(const Tensor<gpu, 3, DType>& data,
                             Tensor<gpu, 3, DType>* out,
                             const TempWorkspace<DType>& workspace,
                             const BoxNMSParam& param,
                             Stream<gpu>* s) {
  const int n_threads = 512;
  index_t N = data.shape_.Size();
  const auto blocks = ceil_div(N, n_threads);
  FilterAndPrepareAuxDataKernel<<<blocks,
                                   n_threads,
                                   0,
                                   Stream<gpu>::GetStream(s)>>>(
    data.dptr_, out->dptr_, workspace.scores,
    data.shape_[1], data.shape_[2], N,
    param.valid_thresh, param.id_index,
    param.score_index, param.background_id);
}

template <bool check_topk, bool check_score, typename DType>
__global__ void CompactDataKernel(const index_t* indices, const DType* source,
                                   DType* destination, const index_t topk,
                                   const index_t element_width,
                                   const index_t num_elements_per_batch,
                                   const int score_index,
                                   const index_t N) {
  const index_t tid_start = blockIdx.x * blockDim.x + threadIdx.x;
  for (index_t tid = tid_start; tid < N; tid += blockDim.x * gridDim.x) {
    const index_t my_element = tid / element_width;
    const index_t my_element_in_batch = my_element % num_elements_per_batch;
    if (check_topk && my_element_in_batch >= topk) {
      destination[tid] = -1;
    } else {
      DType ret;
      const index_t source_element = indices[my_element];
      DType score = 0;
      if (check_score) {
        score = source[source_element * element_width + score_index];
      }
      if (score >= 0) {
        ret = source[source_element * element_width + tid % element_width];
      } else {
        ret = -1;
      }
      destination[tid] = ret;
    }
  }
}

template <bool check_score, typename DType>
void CompactData(const Tensor<gpu, 1, index_t>& indices,
                 const Tensor<gpu, 3, DType>& source,
                 Tensor<gpu, 3, DType>* destination,
                 const index_t topk,
                 const int score_index,
                 Stream<gpu>* s) {
  const int n_threads = 512;
  const index_t max_blocks = 320;
  index_t N = source.shape_.Size();
  const auto blocks = std::min(ceil_div(N, n_threads), max_blocks);
  if (topk > 0) {
    CompactDataKernel<true, check_score><<<blocks, n_threads, 0,
                                            Stream<gpu>::GetStream(s)>>>(
      indices.dptr_, source.dptr_,
      destination->dptr_, topk,
      source.shape_[2], source.shape_[1],
      score_index, N);
  } else {
    CompactDataKernel<false, check_score><<<blocks, n_threads, 0,
                                             Stream<gpu>::GetStream(s)>>>(
      indices.dptr_, source.dptr_,
      destination->dptr_, topk,
      source.shape_[2], source.shape_[1],
      score_index, N);
  }
}

template <typename DType>
void WorkspaceForSort(const index_t num_elem,
                      const index_t topk,
                      const int alignment,
                      TempWorkspace<DType>* workspace) {
  const index_t sort_scores_temp_space =
    mxnet::op::SortByKeyWorkspaceSize<DType, index_t, gpu>(num_elem, false, false);
  const index_t sort_topk_scores_temp_space =
    mxnet::op::SortByKeyWorkspaceSize<DType, index_t, gpu>(topk, false, false);
  workspace->scratch_space = align(std::max(sort_scores_temp_space, sort_topk_scores_temp_space),
                                   alignment);
}

template <int encode, typename DType>
__global__ void CalculateGreedyNMSResultsKernel(const DType* data, uint32_t* result,
                                                 const index_t current_start,
                                                 const index_t num_elems,
                                                 const index_t num_batches,
                                                 const index_t num_blocks_per_row_batch,
                                                 const index_t num_blocks_per_row,
                                                 const index_t topk,
                                                 const index_t element_width,
                                                 const index_t num_elements_per_batch,
                                                 const int coord_index,
                                                 const int class_index,
                                                 const int score_index,
                                                 const float threshold);

template <typename DType>
__global__ void ReduceNMSResultTriangleKernel(uint32_t* nms_results,
                                               DType * data,
                                               const index_t score_index,
                                               const index_t element_width,
                                               const index_t num_batches,
                                               const index_t num_elems,
                                               const index_t start_index,
                                               const index_t topk);

template <typename DType>
__global__ void ReduceNMSResultRestKernel(DType* data,
                                           const uint32_t* nms_results,
                                           const index_t score_index,
                                           const index_t element_width,
                                           const index_t num_batches,
                                           const index_t num_elements_per_batch,
                                           const index_t start_index,
                                           const index_t topk,
                                           const index_t num_blocks_per_batch);

template <typename DType>
struct NMS {
  static constexpr int THRESHOLD = 512;

  void operator()(Tensor<gpu, 3, DType>* data,
                  Tensor<gpu, 2, uint32_t>* scratch,
                  const index_t topk,
                  const BoxNMSParam& param,
                  Stream<gpu>* s) {
    const int n_threads = 512;
    const index_t num_batches = data->shape_[0];
    const index_t num_elements_per_batch = data->shape_[1];
    const index_t element_width = data->shape_[2];
    for (index_t current_start = 0; current_start < topk; current_start += THRESHOLD) {
      const index_t n_elems = topk - current_start;
      const index_t num_blocks_per_row_batch = ceil_div(n_elems, n_threads);
      const index_t num_blocks_per_row =  num_blocks_per_row_batch * num_batches;
      const index_t n_blocks = THRESHOLD / (sizeof(uint32_t) * 8) * num_blocks_per_row;
      if (param.in_format == box_common_enum::kCorner) {
        CalculateGreedyNMSResultsKernel<box_common_enum::kCorner>
          <<<n_blocks, n_threads, 0, Stream<gpu>::GetStream(s)>>>(
            data->dptr_, scratch->dptr_, current_start, n_elems, num_batches,
            num_blocks_per_row_batch, num_blocks_per_row, topk, element_width,
            num_elements_per_batch, param.coord_start,
            param.force_suppress ? -1 : param.id_index,
            param.score_index, param.overlap_thresh);
      } else {
        CalculateGreedyNMSResultsKernel<box_common_enum::kCenter>
          <<<n_blocks, n_threads, 0, Stream<gpu>::GetStream(s)>>>(
            data->dptr_, scratch->dptr_, current_start, n_elems, num_batches,
            num_blocks_per_row_batch, num_blocks_per_row, topk, element_width,
            num_elements_per_batch, param.coord_start,
            param.force_suppress ? -1 : param.id_index,
            param.score_index, param.overlap_thresh);
      }
      ReduceNMSResultTriangleKernel<<<num_batches, THRESHOLD, 0, Stream<gpu>::GetStream(s)>>>(
          scratch->dptr_, data->dptr_, param.score_index,
          element_width, num_batches, num_elements_per_batch,
          current_start, topk);
      const index_t n_rest_elems = n_elems - THRESHOLD;
      const index_t num_rest_blocks_per_batch = ceil_div(n_rest_elems, n_threads);
      const index_t num_rest_blocks = num_rest_blocks_per_batch * num_batches;
      if (n_rest_elems > 0) {
        ReduceNMSResultRestKernel<<<num_rest_blocks, n_threads, 0, Stream<gpu>::GetStream(s)>>>(
            data->dptr_, scratch->dptr_, param.score_index, element_width,
            num_batches, num_elements_per_batch, current_start, topk,
            num_rest_blocks_per_batch);
      }
    }
  }
};

template <int encode, typename DType>
__device__ __forceinline__ DType calculate_area(const DType b0, const DType b1,
                                                const DType b2, const DType b3) {
  DType width = b2;
  DType height = b3;
  if (encode == box_common_enum::kCorner) {
    width -= b0;
    height -= b1;
  }
  if (width < 0 || height < 0) return 0;
  return width * height;
}

template <int encode, typename DType>
__device__ __forceinline__ DType calculate_intersection(const DType a0, const DType a1,
                                                        const DType a2, const DType a3,
                                                        const DType b0, const DType b1,
                                                        const DType b2, const DType b3) {
  DType wx, wy;
  if (encode == box_common_enum::kCorner) {
    const DType left = a0 > b0 ? a0 : b0;
    const DType bottom = a1 > b1 ? a1 : b1;
    const DType right = a2 < b2 ? a2 : b2;
    const DType top = a3 < b3 ? a3 : b3;
    wx = right - left;
    wy = top - bottom;
  } else {
    const DType al = 2 * a0 - a2;
    const DType ar = 2 * a0 + a2;
    const DType bl = 2 * b0 - b2;
    const DType br = 2 * b0 + b2;
    const DType left = bl > al ? bl : al;
    const DType right = br < ar ? br : ar;
    wx = right - left;
    const DType ab = 2 * a1 - a3;
    const DType at = 2 * a1 + a3;
    const DType bb = 2 * b1 - b3;
    const DType bt = 2 * b1 + b3;
    const DType bottom = bb > ab ? bb : ab;
    const DType top = bt < at ? bt : at;
    wy = top - bottom;
    wy = wy / 4;  // To compensate for both wx and wy being 2x too large
  }
  if (wx <= 0 || wy <= 0) {
    return 0;
  } else {
    return (wx * wy);
  }
}

template <int encode, typename DType>
__launch_bounds__(512)
__global__ void CalculateGreedyNMSResultsKernel(const DType* data, uint32_t* result,
                                                 const index_t current_start,
                                                 const index_t num_elems,
                                                 const index_t num_batches,
                                                 const index_t num_blocks_per_row_batch,
                                                 const index_t num_blocks_per_row,
                                                 const index_t topk,
                                                 const index_t element_width,
                                                 const index_t num_elements_per_batch,
                                                 const int coord_index,
                                                 const int class_index,
                                                 const int score_index,
                                                 const float threshold) {
  constexpr int max_elem_width = 20;
  constexpr int num_other_boxes = sizeof(uint32_t) * 8;
  __shared__ DType other_boxes[max_elem_width * num_other_boxes];
  __shared__ DType other_boxes_areas[num_other_boxes];
  const index_t my_row = blockIdx.x / num_blocks_per_row;
  const index_t my_block_offset_in_row = blockIdx.x % num_blocks_per_row;
  const index_t my_block_offset_in_batch = my_block_offset_in_row % num_blocks_per_row_batch;
  const index_t my_batch = (my_block_offset_in_row) / num_blocks_per_row_batch;
  const index_t my_element_in_batch = my_block_offset_in_batch * blockDim.x +
                                      current_start + threadIdx.x;

  // Load other boxes
  const index_t offset = (my_batch * num_elements_per_batch +
                         current_start + my_row * num_other_boxes) *
                         element_width;
  for (int i = threadIdx.x; i < element_width * num_other_boxes; i += blockDim.x) {
    other_boxes[i] = data[offset + i];
  }
  __syncthreads();

  if (threadIdx.x < num_other_boxes) {
    const int other_boxes_offset = element_width * threadIdx.x;
    const DType their_area = calculate_area<encode>(
        other_boxes[other_boxes_offset + coord_index + 0],
        other_boxes[other_boxes_offset + coord_index + 1],
        other_boxes[other_boxes_offset + coord_index + 2],
        other_boxes[other_boxes_offset + coord_index + 3]);
    other_boxes_areas[threadIdx.x] = their_area;
  }
  __syncthreads();

  if (my_element_in_batch >= topk) return;

  DType my_box[4];
  DType my_class = -1;
  DType my_score = -1;
  const index_t my_offset = (my_batch * num_elements_per_batch + my_element_in_batch) *
                            element_width;
  my_score = data[my_offset + score_index];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    my_box[i] = data[my_offset + coord_index + i];
  }
  if (class_index != -1) {
    my_class = data[my_offset + class_index];
  }
  DType my_area = calculate_area<encode>(my_box[0], my_box[1], my_box[2], my_box[3]);

  uint32_t ret = 0;
  if (my_score != -1) {
#pragma unroll
    for (int i = 0; i < num_other_boxes; ++i) {
      const int other_boxes_offset = element_width * i;
      if ((class_index == -1 || my_class == other_boxes[other_boxes_offset + class_index]) &&
          other_boxes[other_boxes_offset + score_index] != -1) {
        const DType their_area = other_boxes_areas[i];

        const DType intersect = calculate_intersection<encode>(
            my_box[0], my_box[1], my_box[2], my_box[3],
            other_boxes[other_boxes_offset + coord_index + 0],
            other_boxes[other_boxes_offset + coord_index + 1],
            other_boxes[other_boxes_offset + coord_index + 2],
            other_boxes[other_boxes_offset + coord_index + 3]);
        if (intersect > threshold * (my_area + their_area - intersect)) {
          ret = ret | (1u << i);
        }
      }
    }
  }
  result[(my_row * num_batches + my_batch) * topk + my_element_in_batch] = ~ret;
}

template <typename DType>
__launch_bounds__(NMS<DType>::THRESHOLD)
__global__ void ReduceNMSResultTriangleKernel(uint32_t* nms_results,
                                               DType * data,
                                               const index_t score_index,
                                               const index_t element_width,
                                               const index_t num_batches,
                                               const index_t num_elements_per_batch,
                                               const index_t start_index,
                                               const index_t topk) {
  constexpr int n_threads = NMS<DType>::THRESHOLD;
  constexpr int warp_size = 32;
  const index_t my_batch = blockIdx.x;
  const index_t my_element_in_batch = threadIdx.x + start_index;
  const index_t my_element = my_batch * topk + my_element_in_batch;
  const int my_warp = threadIdx.x / warp_size;
  const int my_lane = threadIdx.x % warp_size;

  __shared__ uint32_t current_valid_boxes[n_threads / warp_size];
  const uint32_t full_mask = 0xFFFFFFFF;
  const uint32_t my_lane_mask = 1 << my_lane;
  const uint32_t earlier_threads_mask = (1 << (my_lane + 1)) - 1;
  uint32_t valid = my_lane_mask;
  uint32_t valid_boxes = full_mask;

  uint32_t my_next_mask = my_element_in_batch < topk ?
    nms_results[my_element]:
    full_mask;
#pragma unroll
  for (int i = 0; i < n_threads / warp_size; ++i) {
    uint32_t my_mask = my_next_mask;
    my_next_mask = (((i + 1) < n_threads / warp_size) &&
                    (my_element_in_batch < topk)) ?
      nms_results[(i + 1) * topk * num_batches + my_element]:
      full_mask;
    if (my_warp == i && !__all_sync(full_mask, my_mask == full_mask)) {
      my_mask = my_mask | earlier_threads_mask;
      // Loop over warp_size - 1 because the last
      // thread does not contribute to the mask anyway
#pragma unroll
      for (int j = 0; j < warp_size - 1; ++j) {
          const uint32_t mask = __shfl_sync(full_mask, valid ? my_mask : full_mask, j);
          valid = valid & mask;
      }
      valid_boxes = __ballot_sync(full_mask, valid);
    }
    if (my_lane == 0 && my_warp == i) {
      current_valid_boxes[i] = valid_boxes;
    }
    __syncthreads();
    if ((my_warp > i) && (((~my_mask) & current_valid_boxes[i]) != 0)) {
      valid = 0;
    }
  }
  if (my_lane == 0) {
    nms_results[my_element] = valid_boxes;
  }
  if (valid == 0) {
    data[(my_batch * num_elements_per_batch + my_element_in_batch) * element_width +
         score_index] = -1;
  }
}

template <typename DType>
__launch_bounds__(512)
__global__ void ReduceNMSResultRestKernel(DType* data,
                                           const uint32_t* nms_results,
                                           const index_t score_index,
                                           const index_t element_width,
                                           const index_t num_batches,
                                           const index_t num_elements_per_batch,
                                           const index_t start_index,
                                           const index_t topk,
                                           const index_t num_blocks_per_batch) {
  constexpr int num_other_boxes = sizeof(uint32_t) * 8;
  constexpr int num_iterations = NMS<DType>::THRESHOLD / num_other_boxes;
  constexpr int warp_size = 32;
  const index_t my_block_offset_in_batch = blockIdx.x % num_blocks_per_batch;
  const index_t my_batch = blockIdx.x / num_blocks_per_batch;
  const index_t my_element_in_batch = my_block_offset_in_batch * blockDim.x +
                                      start_index + NMS<DType>::THRESHOLD + threadIdx.x;
  const index_t my_element = my_batch * topk + my_element_in_batch;

  if (my_element_in_batch >= topk) return;

  bool valid = true;

#pragma unroll
  for (int i = 0; i < num_iterations; ++i) {
    const uint32_t my_mask = nms_results[i * topk * num_batches + my_element];
    const uint32_t valid_boxes = nms_results[my_batch * topk + i * warp_size + start_index];

    const bool no_hit = (valid_boxes & (~my_mask)) == 0;
    valid = valid && no_hit;
  }

  if (!valid) {
    data[(my_batch * num_elements_per_batch + my_element_in_batch) * element_width +
          score_index] = -1;
  }
}

template <typename DType>
TempWorkspace<DType> GetWorkspace(const index_t num_batch,
                                  const index_t num_elem,
                                  const int width_elem,
                                  const index_t topk,
                                  const OpContext& ctx) {
  TempWorkspace<DType> workspace;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  const int alignment = 128;

  // Get the workspace size
  workspace.scores_temp_space = 2 * align(num_batch * num_elem * sizeof(DType), alignment);
  workspace.indices_temp_spaces = 2 * align(num_batch * num_elem * sizeof(index_t), alignment);
  WorkspaceForSort(num_elem, topk, alignment, &workspace);
  // Place for a buffer
  workspace.buffer_space = align(num_batch * num_elem * width_elem * sizeof(DType), alignment);
  workspace.nms_scratch_space = align(NMS<DType>::THRESHOLD / (sizeof(uint32_t) * 8) *
                                      num_batch * topk * sizeof(uint32_t), alignment);

  const index_t workspace_size = workspace.scores_temp_space +
                                 workspace.scratch_space +
                                 workspace.nms_scratch_space +
                                 workspace.indices_temp_spaces;

  // Obtain the memory for workspace
  Tensor<gpu, 1, DType> scratch_memory = ctx.requested[box_nms_enum::kTempSpace]
    .get_space_typed<gpu, 1, DType>(mshadow::Shape1(workspace_size), s);

  // Populate workspace pointers
  workspace.scores = scratch_memory.dptr_;
  workspace.scratch = reinterpret_cast<uint8_t*>(workspace.scores) +
                                                 workspace.scores_temp_space;
  workspace.buffer = reinterpret_cast<DType*>(workspace.scratch +
                                              workspace.scratch_space);
  workspace.nms_scratch = reinterpret_cast<uint32_t*>(
                            reinterpret_cast<uint8_t*>(workspace.buffer) +
                            workspace.buffer_space);
  workspace.indices = reinterpret_cast<index_t*>(
                            reinterpret_cast<uint8_t*>(workspace.nms_scratch) +
                            workspace.nms_scratch_space);
  return workspace;
}

template <typename DType>
__global__ void ExtractScoresKernel(const DType* data, DType* scores,
                                     const index_t N, const int element_width,
                                     const int score_index) {
  const index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    scores[tid] = data[tid * element_width + score_index];
  }
}

template <typename DType>
void CompactNMSResults(const Tensor<gpu, 3, DType>& data,
                       Tensor<gpu, 3, DType>* out,
                       Tensor<gpu, 1, index_t>* indices,
                       Tensor<gpu, 1, DType>* scores,
                       Tensor<gpu, 1, index_t>* sorted_indices,
                       Tensor<gpu, 1, DType>* sorted_scores,
                       Tensor<gpu, 1, char>* scratch,
                       const int score_index,
                       const index_t topk,
                       Stream<gpu>* s) {
  using mshadow::Shape1;
  constexpr int n_threads = 512;
  const index_t num_elements = scores->shape_.Size();
  const index_t num_elements_per_batch = data.shape_[1];
  const index_t num_batches = data.shape_[0];
  const int element_width = data.shape_[2];
  const index_t n_blocks = ceil_div(num_elements, n_threads);
  ExtractScoresKernel<<<n_blocks, n_threads, 0, Stream<gpu>::GetStream(s)>>>(
      data.dptr_, scores->dptr_, num_elements, element_width, score_index);
  *indices = mshadow::expr::range<index_t>(0, num_elements);
  for (index_t i = 0; i < num_batches; ++i) {
    // Sort each batch separately
    Tensor<gpu, 1, DType> scores_batch(scores->dptr_ + i * num_elements_per_batch,
                                       Shape1(topk),
                                       s);
    Tensor<gpu, 1, index_t> indices_batch(indices->dptr_ + i * num_elements_per_batch,
                                          Shape1(topk),
                                          s);
    Tensor<gpu, 1, DType> sorted_scores_batch(sorted_scores->dptr_ + i * num_elements_per_batch,
                                              Shape1(topk),
                                              s);
    Tensor<gpu, 1, index_t> sorted_indices_batch(sorted_indices->dptr_ + i * num_elements_per_batch,
                                                 Shape1(topk),
                                                 s);
    mxnet::op::SortByKey(scores_batch, indices_batch, false, scratch,
                         0, 8 * sizeof(DType), &sorted_scores_batch,
                         &sorted_indices_batch);
  }
  CompactData<true>(*sorted_indices, data, out, topk, score_index, s);
}

}  // namespace

void BoxNMSForwardGPU_notemp(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using mshadow::Shape1;
  using mshadow::Shape2;
  using mshadow::Shape3;
  CHECK_NE(req[0], kAddTo) << "BoxNMS does not support kAddTo";
  CHECK_NE(req[0], kWriteInplace) << "BoxNMS does not support in place computation";
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U) << "BoxNMS output: [output, temp]";
  const BoxNMSParam& param = nnvm::get<BoxNMSParam>(attrs.parsed);
  Stream<gpu> *s = ctx.get_stream<gpu>();
  mxnet::TShape in_shape = inputs[box_nms_enum::kData].shape_;
  int indim = in_shape.ndim();
  int num_batch = indim <= 2? 1 : in_shape.ProdShape(0, indim - 2);
  int num_elem = in_shape[indim - 2];
  int width_elem = in_shape[indim - 1];

  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<gpu, 3, DType> data = inputs[box_nms_enum::kData]
     .get_with_shape<gpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);
    Tensor<gpu, 3, DType> out = outputs[box_nms_enum::kOut]
     .get_with_shape<gpu, 3, DType>(Shape3(num_batch, num_elem, width_elem), s);

    // Special case for topk == 0
    if (param.topk == 0) {
      if (req[0] != kNullOp &&
          req[0] != kWriteInplace) {
        out = mshadow::expr::F<mshadow_op::identity>(data);
      }
      return;
    }

    index_t topk = param.topk > 0 ? std::min(param.topk, num_elem) : num_elem;
    const auto& workspace = GetWorkspace<DType>(num_batch, num_elem,
                                                width_elem, topk, ctx);

    FilterAndPrepareAuxData(data, &out, workspace, param, s);
    Tensor<gpu, 1, DType> scores(workspace.scores, Shape1(num_batch * num_elem), s);
    Tensor<gpu, 1, DType> sorted_scores(workspace.scores + scores.MSize(),
                                        Shape1(num_batch * num_elem), s);
    Tensor<gpu, 1, index_t> indices(workspace.indices, Shape1(num_batch * num_elem), s);
    Tensor<gpu, 1, index_t> sorted_indices(workspace.indices + indices.MSize(),
                                           Shape1(num_batch * num_elem), s);
    Tensor<gpu, 1, char> scratch(reinterpret_cast<char*>(workspace.scratch),
                                        Shape1(workspace.scratch_space), s);
    Tensor<gpu, 3, DType> buffer(workspace.buffer,
                                 Shape3(num_batch, num_elem, width_elem), s);
    Tensor<gpu, 2, uint32_t> nms_scratch(workspace.nms_scratch,
                                         Shape2(NMS<DType>::THRESHOLD / (sizeof(uint32_t) * 8),
                                                topk * num_batch),
                                         s);
    indices = mshadow::expr::range<index_t>(0, num_batch * num_elem);
    for (index_t i = 0; i < num_batch; ++i) {
      // Sort each batch separately
      Tensor<gpu, 1, DType> scores_batch(scores.dptr_ + i * num_elem,
                                         Shape1(num_elem),
                                         s);
      Tensor<gpu, 1, index_t> indices_batch(indices.dptr_ + i * num_elem,
                                            Shape1(num_elem),
                                            s);
      Tensor<gpu, 1, DType> sorted_scores_batch(sorted_scores.dptr_ + i * num_elem,
                                                Shape1(num_elem),
                                                s);
      Tensor<gpu, 1, index_t> sorted_indices_batch(sorted_indices.dptr_ + i * num_elem,
                                                   Shape1(num_elem),
                                                   s);
      mxnet::op::SortByKey(scores_batch, indices_batch, false, &scratch, 0,
                           8 * sizeof(DType), &sorted_scores_batch,
                           &sorted_indices_batch);
    }
    CompactData<false>(sorted_indices, out, &buffer, topk, -1, s);
    NMS<DType> nms;
    nms(&buffer, &nms_scratch, topk, param, s);
    CompactNMSResults(buffer, &out, &indices, &scores, &sorted_indices,
                      &sorted_scores, &scratch, param.score_index, topk, s);

    // convert encoding
    if (param.in_format != param.out_format) {
      if (box_common_enum::kCenter == param.out_format) {
        mxnet::op::mxnet_op::Kernel<corner_to_center, gpu>::Launch(s, num_batch * num_elem,
          out.dptr_ + param.coord_start, width_elem);
      } else {
        mxnet::op::mxnet_op::Kernel<center_to_corner, gpu>::Launch(s, num_batch * num_elem,
          out.dptr_ + param.coord_start, width_elem);
      }
    }
  });
}

void BoxNMSForwardGPU(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U) << "BoxNMS output: [output, temp]";
  if (req[1] == kNullOp) {
    BoxNMSForwardGPU_notemp(attrs, ctx, inputs, req, outputs);
    return;
  }
  BoxNMSForward<gpu>(attrs, ctx, inputs, req, outputs);
}


NNVM_REGISTER_OP(_contrib_box_nms)
.set_attr<FCompute>("FCompute<gpu>", BoxNMSForwardGPU);

NNVM_REGISTER_OP(_backward_contrib_box_nms)
.set_attr<FCompute>("FCompute<gpu>", BoxNMSBackward<gpu>);

NNVM_REGISTER_OP(_contrib_box_iou)
.set_attr<FCompute>("FCompute<gpu>", BoxOverlapForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_box_iou)
.set_attr<FCompute>("FCompute<gpu>", BoxOverlapBackward<gpu>);

NNVM_REGISTER_OP(_contrib_bipartite_matching)
.set_attr<FCompute>("FCompute<gpu>", BipartiteMatchingForward<gpu>);

NNVM_REGISTER_OP(_backward_contrib_bipartite_matching)
.set_attr<FCompute>("FCompute<gpu>", BipartiteMatchingBackward<gpu>);

NNVM_REGISTER_OP(_contrib_box_encode)
.set_attr<FCompute>("FCompute<gpu>", BoxEncodeForward<gpu>);

NNVM_REGISTER_OP(_contrib_box_decode)
.set_attr<FCompute>("FCompute<gpu>", BoxDecodeForward<gpu>);

}  // namespace op
}  // namespace mxnet
