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
 *  Copyright (c) 2019 by Contributors
 * \file multi_sum_sq.cu
 * \brief vectorized sums of squares norm over multiple arrays operators
 * \author Clement Fuji Tsang, Andrei Ivanov, Moises Hernandez
 */
#include "./multi_sum_sq-inl.h"
#include <cub/cub.cuh>

#define ILP 4
#define BLOCK_LIMIT 320
#define ARRAY_LIMIT 110

namespace mxnet {
namespace op {

// Shamelessly gotten from:
// https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_apply.cuh
// https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_l2norm_kernel.cu
// https://github.com/NVIDIA/apex/blob/master/csrc/type_shim.h

const int chunk_size = 32768;

template <typename DType>
struct MultiSumSqKernelParam {
  DType* addresses[ARRAY_LIMIT];
  int sizes[ARRAY_LIMIT];
  unsigned char block_to_tensor[BLOCK_LIMIT];
  int block_to_chunk[BLOCK_LIMIT];
  int max_chunks_per_tensor = -1;
};

template<typename DType>
__device__ __forceinline__ DType ReduceBlockIntoLanes(DType* x,
                                                      DType val) {
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  if (block_size >= 64) {
    x[tid] = val;
    __syncthreads();
  }

  #pragma unroll
  for (int i = (block_size >> 1); i >= 64; i >>= 1) {
    if (tid < i)
      x[tid] = x[tid] + x[tid+i];
    __syncthreads();
  }

  DType final;
  if (tid < 32) {
    if (block_size >= 64)
      final = x[tid] + x[tid+32];
    else
      final = val;

    #pragma unroll
    for (int i = 16; i >= 1; i >>= 1)
      final = final + __shfl_down_sync(0xffffffff, final, i);
  }
  return final;
}

template<typename DType>
__global__ void MultiSumSqKernel(int chunk_size,
                                 MultiSumSqKernelParam<DType> param,
                                 float* block_reductions,
                                 int start_tensor_id) {
  const int tensor_loc = param.block_to_tensor[blockIdx.x];
  const int chunk_len = param.block_to_chunk[blockIdx.x] * chunk_size;
  const int n = param.sizes[tensor_loc] - chunk_len;
  const DType* x = param.addresses[tensor_loc] + chunk_len;
  const auto i_max = n <= chunk_size ? n : chunk_size;
  __shared__ float vals[512];

  // Non-divergent exit condition for __syncthreads, not necessary here
  float val = 0;
  for (int i_start = 0;
       i_start < i_max;
       i_start += blockDim.x * ILP) {
    int i = i_start + threadIdx.x;
#pragma unroll
    for (int ii = 0; ii < ILP && i < i_max; ++ii, i += blockDim.x) {
      const auto incoming_val = static_cast<float>(x[i]);
      val += incoming_val * incoming_val;
    }
  }
  const float final = ReduceBlockIntoLanes(vals, val);

  if (threadIdx.x == 0) {
    block_reductions[(start_tensor_id + tensor_loc) * param.max_chunks_per_tensor +
                    param.block_to_chunk[blockIdx.x]] = final;
  }
}

template<typename DType>
__global__ void GlobalReductionKernel(MultiSumSqKernelParam<DType> param,
                                     float* block_reductions,
                                     float* output) {
  __shared__ float vals[512];
  float* reductions_this_tensor = block_reductions + blockIdx.x * param.max_chunks_per_tensor;
  float val = 0;
  for (int i = threadIdx.x; i < param.max_chunks_per_tensor; i += blockDim.x)
    val += reductions_this_tensor[i];

  float final = ReduceBlockIntoLanes(vals, val);

  if (threadIdx.x == 0)
    output[blockIdx.x] = final;
}

template<>
size_t GetRequiredStorageMultiSumSq<gpu>(const std::vector<TBlob> &inputs,
                                         int* param_max_chunks_per_tensor) {
  // find max num of chunks in tensors
  int max_chunks_per_tensor = -1;
  for (size_t t = 0; t < inputs.size(); t++) {
    int chunks_this_tensor = (inputs[t].shape_.Size() + chunk_size - 1) / chunk_size;
    if (chunks_this_tensor > max_chunks_per_tensor)
      max_chunks_per_tensor = chunks_this_tensor;
  }
  if (param_max_chunks_per_tensor != nullptr)
    *param_max_chunks_per_tensor = max_chunks_per_tensor;
  return inputs.size() * max_chunks_per_tensor * sizeof(float);
}

template<>
void MultiSumSqRun<gpu>(const std::vector<TBlob> &inputs, int n_inputs,
                        float *out_ptr, const OpContext &ctx) {
  const int block_size = 512;
  using namespace mxnet_op;
  auto s = ctx.get_stream<gpu>();
  auto stream = mshadow::Stream<gpu>::GetStream(s);

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MultiSumSqKernelParam<DType> param;
    size_t workspace_size = GetRequiredStorageMultiSumSq<gpu>(inputs,
                                                              &param.max_chunks_per_tensor);
    Tensor<gpu, 1, char> workspace =
      ctx.requested[multi_sum_sq::kTempSpace].get_space_typed<gpu, 1, char>(
        Shape1(workspace_size), s);
    Tensor<gpu, 1, float> block_reductions(reinterpret_cast<float*>(&workspace[0]),
      Shape1(n_inputs * param.max_chunks_per_tensor), s);
    CUDA_CALL(cudaMemsetAsync(block_reductions.dptr_, 0,
                              n_inputs * param.max_chunks_per_tensor* sizeof(float),
                              stream));

    int loc_block_info = 0;   // position in param.block_to_tensor and param.block_to_chunck
    int loc_tensor_info = 0;  // position in param.sizes and param.addresses
    int start_tensor_id = 0;
    for (int t = 0; t < n_inputs; t++, loc_tensor_info++) {  // array index in inputs
      param.sizes[loc_tensor_info] = inputs[t].shape_.Size();
      param.addresses[loc_tensor_info] = inputs[t].FlatTo2D<gpu, DType>(s).dptr_;
      const int chunks_this_tensor = (inputs[t].shape_.Size() - 1) / chunk_size;
      for (int chunk = 0; chunk <= chunks_this_tensor; ++chunk) {  // array chunk index
        param.block_to_tensor[loc_block_info] = loc_tensor_info;
        param.block_to_chunk[loc_block_info] = chunk;
        loc_block_info++;

        const bool last_curr_chunk = chunk == chunks_this_tensor;
        const bool tensors_full = last_curr_chunk && loc_tensor_info == (ARRAY_LIMIT-1);
        const bool blocks_full = (loc_block_info == BLOCK_LIMIT);
        const bool last_chunk = last_curr_chunk && t == n_inputs - 1;
        if (!(tensors_full || blocks_full || last_chunk))
          continue;
        MultiSumSqKernel<<<loc_block_info, block_size, 0, stream>>>
          (chunk_size, param, block_reductions.dptr_, start_tensor_id);
        MSHADOW_CUDA_POST_KERNEL_CHECK(MultiSumSqKernel);

        loc_block_info = 0;
        if (last_curr_chunk) {  // if you start from a new tensor
          loc_tensor_info = -1;
          start_tensor_id = t + 1;
        } else {  // if you start from the same tensor
          param.sizes[0] = param.sizes[loc_tensor_info];
          param.addresses[0] = param.addresses[loc_tensor_info];
          loc_tensor_info = 0;
          start_tensor_id = t;
        }
      }
    }
    // Global reduction
    GlobalReductionKernel<<<n_inputs, block_size, 0, stream>>>
      (param, block_reductions.dptr_, out_ptr);
  });
}

NNVM_REGISTER_OP(multi_sum_sq)
.set_attr<FCompute>("FCompute<gpu>", MultiSumSq<gpu>);

}  // namespace op
}  // namespace mxnet
