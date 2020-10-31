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
 * Copyright (c) 2020 by Contributors
 * \file relu_lib.cu
 * \brief simple custom relu and noisy relu operator implemented using CUDA function
 */

#include <iostream>
#include "relu_lib.h"

using namespace mxnet::ext;

__global__ void relu_gpu_forward(float *out, float *in, int64_t N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
    out[tid] = in[tid] > 0 ? in[tid] : 0;
}

MXReturnValue forwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                         std::vector<MXTensor>* inputs,
                         std::vector<MXTensor>* outputs,
                         const OpResource& res) {
  float* in_data = inputs->at(0).data<float>();
  float* out_data = outputs->at(0).data<float>();

  mx_stream_t cuda_stream = res.get_cuda_stream();
  int64_t N = inputs->at(0).size();
  int num_block = (N + NumThreadPerBlock - 1) / NumThreadPerBlock;

  relu_gpu_forward<<<num_block,NumThreadPerBlock,0,cuda_stream>>>(out_data, in_data, N);

  return MX_SUCCESS;
}

__global__ void relu_gpu_backward(float *ingrad, float *outgrad, float *indata, int64_t N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
    ingrad[tid] = indata[tid] > 0 ? 1 * outgrad[tid] : 0;
}

MXReturnValue backwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                          std::vector<MXTensor>* inputs,
                          std::vector<MXTensor>* outputs,
                          const OpResource& res) {
  float* out_grad = inputs->at(0).data<float>();
  float* in_data = inputs->at(1).data<float>();
  float* in_grad = outputs->at(0).data<float>();

  mx_stream_t cuda_stream = res.get_cuda_stream();
  int64_t N = inputs->at(0).size();
  int num_block = (N + NumThreadPerBlock - 1) / NumThreadPerBlock;
  relu_gpu_backward<<<num_block,NumThreadPerBlock,0,cuda_stream>>>(in_grad, out_grad, in_data, N);

  return MX_SUCCESS;
}

__global__ void noisy_relu_gpu_forward(float *out, float *in, int64_t N, mx_gpu_rand_t* states, int step) {
    // the launcher logic ensures tid less than NumGPURandomStates
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread generates unique sequence of random numbers
    mx_gpu_rand_t thread_state = states[tid];
    // each thread works on <step> number of calculation
    int start = tid * step;
    int end = start + step;
    for (int i=start; i<end && i<N; ++i) {
        float noise = curand_normal(&thread_state);
        out[i] = in[i] + noise > 0 ? in[i] + noise : 0;
    }
}

MXReturnValue noisyForwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                              std::vector<MXTensor>* inputs,
                              std::vector<MXTensor>* outputs,
                              const OpResource& res) {
  float* in_data = inputs->at(0).data<float>();
  float* out_data = outputs->at(0).data<float>();

  mx_stream_t cuda_stream = res.get_cuda_stream();
  int64_t N = inputs->at(0).size();

  // below is mxnet recommended workflow to parallel random number generating
  int nthread = (N + NumRandomPerThread - 1) / NumRandomPerThread;
  // we should not launch more threads than mxnet supported random number GPU states
  int num_thread_need = nthread < MX_NUM_GPU_RANDOM_STATES ? nthread : MX_NUM_GPU_RANDOM_STATES;
  // each cuda thread processes [step * tid, step * id + step) snippet of input tensor
  int step = (N + num_thread_need - 1) / num_thread_need;
  // this can ensure number of parallel threads less than mxnet supported random number states
  int num_block = (num_thread_need + NumThreadPerBlock - 1) / NumThreadPerBlock;

  noisy_relu_gpu_forward<<<num_block,NumThreadPerBlock,0,cuda_stream>>>(
                                out_data, in_data, N, res.get_gpu_rand_states(), step);

  return MX_SUCCESS;
}
