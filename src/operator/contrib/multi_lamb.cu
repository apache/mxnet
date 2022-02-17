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
 * \file multi_lamb.cu
 * \brief multi-tensor LAMB optimizer
 * \author Moises Hernandez
 */

#include "./multi_lamb-inl.h"

namespace mxnet {
namespace op {

#define BLOCK_SIZE_LAMB 512
#define ILP_LAMB        4

template <bool has_mixed_precision, typename MPDType, typename DType>
__global__ void KernelStep1(const MultiLAMBKernelParam<DType, MPDType> kernel_params,
                            const float beta1,
                            const float beta2,
                            const MPDType beta3,
                            const MPDType beta4,
                            const float epsilon,
                            const float clip_gradient,
                            const bool bias_correction,
                            const float rescale_grad,
                            float* temp_g,
                            int* block_to_tensor,
                            int* block_to_chunk) {
  const int tensor_id = block_to_tensor[blockIdx.x];
  const int chunck_id = block_to_chunk[blockIdx.x];
  const int start_pos = chunck_id * kernel_params.chunk_size + threadIdx.x;
  const int stop_pos  = chunck_id * kernel_params.chunk_size + kernel_params.chunk_size;

  MPDType biascorrection1, biascorrection2;
  if (bias_correction) {
    biascorrection1 = 1.0 - static_cast<MPDType>(pow(
                                beta1, static_cast<float>(kernel_params.step_count[tensor_id])));
    biascorrection2 = 1.0 - static_cast<MPDType>(pow(
                                beta2, static_cast<float>(kernel_params.step_count[tensor_id])));
  } else {
    biascorrection1 = static_cast<MPDType>(1.0);
    biascorrection2 = static_cast<MPDType>(1.0);
  }

  MPDType r_weight[ILP_LAMB];
  MPDType r_grad[ILP_LAMB];
  MPDType r_mean[ILP_LAMB];
  MPDType r_var[ILP_LAMB];
  MPDType r_g[ILP_LAMB];

  for (size_t i = start_pos; i < stop_pos && i < kernel_params.sizes[tensor_id];
       i += blockDim.x * ILP_LAMB) {
#pragma unroll
    for (int ii = 0; ii < ILP_LAMB; ii++) {
      int load_pos = i + ii * blockDim.x;
      if (load_pos < stop_pos && load_pos < kernel_params.sizes[tensor_id]) {
        r_weight[ii] = has_mixed_precision ?
                           kernel_params.weights32[tensor_id][load_pos] :
                           static_cast<MPDType>(kernel_params.weights[tensor_id][load_pos]);
        r_grad[ii]   = static_cast<MPDType>(kernel_params.grads[tensor_id][load_pos]);
        r_mean[ii]   = kernel_params.mean[tensor_id][load_pos];
        r_var[ii]    = kernel_params.var[tensor_id][load_pos];
      } else {
        r_weight[ii] = static_cast<MPDType>(0);
        r_grad[ii]   = static_cast<MPDType>(0);
        r_mean[ii]   = static_cast<MPDType>(0);
        r_var[ii]    = static_cast<MPDType>(0);
      }
    }
#pragma unroll
    for (int ii = 0; ii < ILP_LAMB; ii++) {
      r_grad[ii] = r_grad[ii] * static_cast<MPDType>(rescale_grad);
      if (clip_gradient >= 0.0f)
        r_grad[ii] = max(min(r_grad[ii], clip_gradient), -clip_gradient);
      r_mean[ii] = static_cast<MPDType>(beta1) * r_mean[ii] + beta3 * r_grad[ii];
      r_var[ii]  = static_cast<MPDType>(beta2) * r_var[ii] + beta4 * r_grad[ii] * r_grad[ii];
      r_g[ii]    = (r_mean[ii] / biascorrection1) /
                (sqrt(r_var[ii] / biascorrection2) + static_cast<MPDType>(epsilon));
      r_g[ii] = __fmaf_rn(kernel_params.wds[tensor_id], r_weight[ii], r_g[ii]);
    }
#pragma unroll
    for (int ii = 0; ii < ILP_LAMB; ii++) {
      int store_pos = i + ii * blockDim.x;
      if (store_pos < stop_pos && store_pos < kernel_params.sizes[tensor_id]) {
        kernel_params.mean[tensor_id][store_pos]                   = r_mean[ii];
        kernel_params.var[tensor_id][store_pos]                    = r_var[ii];
        temp_g[kernel_params.tensor2temp_g[tensor_id] + store_pos] = r_g[ii];
      }
    }
  }
}

template <bool has_mixed_precision, typename MPDType, typename DType>
__global__ void KernelStep2(const MultiLAMBKernelParam<DType, MPDType> kernel_params,
                            const float* sum_sq_weigths,
                            const float* sum_sq_temp_g,
                            const float* temp_g,
                            const float lower_bound,
                            const float upper_bound,
                            int* block_to_tensor,
                            int* block_to_chunk,
                            const OpReqType req) {
  const int tensor_id = block_to_tensor[blockIdx.x];
  const int chunck_id = block_to_chunk[blockIdx.x];
  const int start_pos = chunck_id * kernel_params.chunk_size + threadIdx.x;
  const int stop_pos  = chunck_id * kernel_params.chunk_size + kernel_params.chunk_size;

  MPDType r1 = sqrtf(sum_sq_weigths[tensor_id]);
  MPDType r2 = sqrtf(sum_sq_temp_g[tensor_id]);
  if (lower_bound >= 0)
    r1 = max(r1, lower_bound);
  if (upper_bound >= 0)
    r1 = min(r1, upper_bound);

  MPDType lr_adjusted;
  if (r1 == 0.0f || r2 == 0.0f)
    lr_adjusted = kernel_params.learning_rates[tensor_id];
  else
    lr_adjusted = kernel_params.learning_rates[tensor_id] * r1 / r2;

  MPDType r_weight[ILP_LAMB];
  MPDType r_g[ILP_LAMB];

  for (size_t i = start_pos; i < stop_pos && i < kernel_params.sizes[tensor_id];
       i += blockDim.x * ILP_LAMB) {
#pragma unroll
    for (int ii = 0; ii < ILP_LAMB; ii++) {
      int load_pos = i + ii * blockDim.x;
      if (load_pos < stop_pos && load_pos < kernel_params.sizes[tensor_id]) {
        r_weight[ii] = has_mixed_precision ?
                           kernel_params.weights32[tensor_id][load_pos] :
                           static_cast<MPDType>(kernel_params.weights[tensor_id][load_pos]);
        r_g[ii]      = temp_g[kernel_params.tensor2temp_g[tensor_id] + load_pos];
      }
    }
#pragma unroll
    for (int ii = 0; ii < ILP_LAMB; ii++) {
      r_weight[ii] -= lr_adjusted * r_g[ii];
    }
#pragma unroll
    for (int ii = 0; ii < ILP_LAMB; ii++) {
      int store_pos = i + ii * blockDim.x;
      if (store_pos < stop_pos && store_pos < kernel_params.sizes[tensor_id]) {
        if (has_mixed_precision)
          kernel_params.weights32[tensor_id][store_pos] = r_weight[ii];
        KERNEL_ASSIGN(kernel_params.out_data[tensor_id][store_pos], req, r_weight[ii]);
      }
    }
  }
}

template <typename MPDType, typename DType>
void CallKernel1(Stream<gpu>* s,
                 const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                 const MultiLAMBParam& param,
                 float* temp_g,
                 int* block_to_tensor,
                 int* block_to_chunk) {
  int nblocks            = kernel_params.nchunks;
  int* host_block2tensor = reinterpret_cast<int*>(malloc(kernel_params.nchunks * sizeof(int)));
  int* host_block2chunk  = reinterpret_cast<int*>(malloc(kernel_params.nchunks * sizeof(int)));
  int chunk_id           = 0;
  for (size_t index = 0; index < kernel_params.ntensors; ++index) {
    int current_chunk = 0;
    for (size_t j = 0; j < kernel_params.sizes[index]; j += kernel_params.chunk_size) {
      host_block2tensor[chunk_id] = index;
      host_block2chunk[chunk_id]  = current_chunk;
      current_chunk++;
      chunk_id++;
    }
  }
  cudaMemcpyAsync(block_to_tensor,
                  host_block2tensor,
                  kernel_params.nchunks * sizeof(int),
                  cudaMemcpyHostToDevice,
                  Stream<gpu>::GetStream(s));
  cudaMemcpyAsync(block_to_chunk,
                  host_block2chunk,
                  kernel_params.nchunks * sizeof(int),
                  cudaMemcpyHostToDevice,
                  Stream<gpu>::GetStream(s));

  bool has_mixed_precision = !std::is_same<DType, MPDType>::value;
  MPDType beta3            = 1.0 - param.beta1;
  MPDType beta4            = 1.0 - param.beta2;

  if (has_mixed_precision)
    KernelStep1<true>
        <<<nblocks, BLOCK_SIZE_LAMB, 0, Stream<gpu>::GetStream(s)>>>(kernel_params,
                                                                     param.beta1,
                                                                     param.beta2,
                                                                     beta3,
                                                                     beta4,
                                                                     param.epsilon,
                                                                     param.clip_gradient,
                                                                     param.bias_correction,
                                                                     param.rescale_grad,
                                                                     temp_g,
                                                                     block_to_tensor,
                                                                     block_to_chunk);
  else
    KernelStep1<false>
        <<<nblocks, BLOCK_SIZE_LAMB, 0, Stream<gpu>::GetStream(s)>>>(kernel_params,
                                                                     param.beta1,
                                                                     param.beta2,
                                                                     beta3,
                                                                     beta4,
                                                                     param.epsilon,
                                                                     param.clip_gradient,
                                                                     param.bias_correction,
                                                                     param.rescale_grad,
                                                                     temp_g,
                                                                     block_to_tensor,
                                                                     block_to_chunk);
}

template <typename MPDType, typename DType>
void CallKernel2(Stream<gpu>* s,
                 const MultiLAMBKernelParam<DType, MPDType>& kernel_params,
                 const MultiLAMBParam& param,
                 float* r1,
                 float* r2,
                 float* temp_g,
                 int* block_to_tensor,
                 int* block_to_chunk,
                 const OpReqType req) {
  size_t nblocks           = kernel_params.nchunks;
  bool has_mixed_precision = !std::is_same<DType, MPDType>::value;
  if (has_mixed_precision)
    KernelStep2<true><<<nblocks, BLOCK_SIZE_LAMB, 0, Stream<gpu>::GetStream(s)>>>(kernel_params,
                                                                                  r1,
                                                                                  r2,
                                                                                  temp_g,
                                                                                  param.lower_bound,
                                                                                  param.upper_bound,
                                                                                  block_to_tensor,
                                                                                  block_to_chunk,
                                                                                  req);
  else
    KernelStep2<false>
        <<<nblocks, BLOCK_SIZE_LAMB, 0, Stream<gpu>::GetStream(s)>>>(kernel_params,
                                                                     r1,
                                                                     r2,
                                                                     temp_g,
                                                                     param.lower_bound,
                                                                     param.upper_bound,
                                                                     block_to_tensor,
                                                                     block_to_chunk,
                                                                     req);
}

NNVM_REGISTER_OP(_multi_lamb_update)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs& attrs, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", MultiLAMBUpdate<gpu, false>);

NNVM_REGISTER_OP(_multi_mp_lamb_update)
    .set_attr<FIsCUDAGraphsCompatible>("FIsCUDAGraphsCompatible",
                                       [](const NodeAttrs& attrs, const bool) { return false; })
    .set_attr<FCompute>("FCompute<gpu>", MultiLAMBUpdate<gpu, true>);

}  // namespace op
}  // namespace mxnet
