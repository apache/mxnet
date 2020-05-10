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
#include "lib_api.h"

#define NumThreadPerBlock 256 // mxnet recommended cuda thread number per block

__global__ void relu_gpu_forward(float *out, float *in, int64_t N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
    out[tid] = in[tid] > 0 ? in[tid] : 0;
}

__global__ void relu_gpu_backward(float *ingrad, float *outgrad, float *indata, int64_t N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
    ingrad[tid] = indata[tid] > 0 ? 1 * outgrad[tid] : 0;
}

MXReturnValue forwardCPU(const std::unordered_map<std::string, std::string>& attrs,
                         std::vector<MXTensor>* inputs,
                         std::vector<MXTensor>* outputs,
                         const OpResource& res) {
  float* in_data = inputs->at(0).data<float>();
  float* out_data = outputs->at(0).data<float>();
  for (int i=0; i<inputs->at(0).size(); i++) {
    out_data[i] = in_data[i] > 0 ? in_data[i] : 0;
  }
  return MX_SUCCESS;
}

MXReturnValue backwardCPU(const std::unordered_map<std::string, std::string>& attrs,
                          std::vector<MXTensor>* inputs,
                          std::vector<MXTensor>* outputs,
                          const OpResource& res) {
  float* out_grad = inputs->at(0).data<float>();
  float* in_data = inputs->at(1).data<float>();
  float* in_grad = outputs->at(0).data<float>();
  for (int i=0; i<inputs->at(1).size(); i++) {
    in_grad[i] = in_data[i] > 0 ? 1 * out_grad[i] : 0;
  }
  return MX_SUCCESS;
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

MXReturnValue parseAttrs(const std::unordered_map<std::string, std::string>& attrs,
                         int* num_in, int* num_out) {
  *num_in = 1;
  *num_out = 1;
  return MX_SUCCESS;
}

MXReturnValue inferType(const std::unordered_map<std::string, std::string>& attrs,
                        std::vector<int>* intypes,
                        std::vector<int>* outtypes) {
  outtypes->at(0) = intypes->at(0);
  return MX_SUCCESS;
}

MXReturnValue inferShape(const std::unordered_map<std::string, std::string>& attrs,
                         std::vector<std::vector<unsigned int>>* inshapes,
                         std::vector<std::vector<unsigned int>>* outshapes) {
  outshapes->at(0) = inshapes->at(0);
  return MX_SUCCESS;
}

REGISTER_OP(my_relu)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setForward(forwardCPU, "cpu")
.setForward(forwardGPU, "gpu")
.setBackward(backwardCPU, "cpu")
.setBackward(backwardGPU, "gpu");

class MyStatefulReluCPU : public CustomStatefulOp {
  public:
    explicit MyStatefulReluCPU(const std::unordered_map<std::string, std::string>& attrs)
      : attrs_(attrs) {}
    MXReturnValue Forward(std::vector<MXTensor>* inputs,
                          std::vector<MXTensor>* outputs,
                          const OpResource& op_res) {
      return forwardCPU(attrs_, inputs, outputs, op_res);
    }
    MXReturnValue Backward(std::vector<MXTensor>* inputs,
                           std::vector<MXTensor>* outputs,
                           const OpResource& op_res) {
      return backwardCPU(attrs_, inputs, outputs, op_res);
    }
    ~MyStatefulReluCPU() {}
  private:
    const std::unordered_map<std::string, std::string> attrs_;
};

class MyStatefulReluGPU : public CustomStatefulOp {
  public:
    explicit MyStatefulReluGPU(const std::unordered_map<std::string, std::string>& attrs)
      : attrs_(attrs) {}
    MXReturnValue Forward(std::vector<MXTensor>* inputs,
                          std::vector<MXTensor>* outputs,
                          const OpResource& op_res) {
      return forwardGPU(attrs_, inputs, outputs, op_res);
    }
    MXReturnValue Backward(std::vector<MXTensor>* inputs,
                           std::vector<MXTensor>* outputs,
                           const OpResource& op_res) {
      return backwardGPU(attrs_, inputs, outputs, op_res);
    }
    ~MyStatefulReluGPU() {}
  private:
    const std::unordered_map<std::string, std::string> attrs_;
};

MXReturnValue createOpStateCPU(const std::unordered_map<std::string, std::string>& attrs,
                               CustomStatefulOp** op_inst) {
  *op_inst = new MyStatefulReluCPU(attrs);
  return MX_SUCCESS;
}

MXReturnValue createOpStateGPU(const std::unordered_map<std::string, std::string>& attrs,
                               CustomStatefulOp** op_inst) {
  *op_inst = new MyStatefulReluGPU(attrs);
  return MX_SUCCESS;
}

REGISTER_OP(my_state_relu)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setCreateOpState(createOpStateCPU, "cpu")
.setCreateOpState(createOpStateGPU, "gpu");

/*
 * Below is noisy ReLU operator example
 * noisy ReLU is made from ReLU extended to include Gaussian noise
 * forward - add Gaussian noise generated from normal distribution to each unit
 * backward - gradient doesn't need to change since noise is constant
 */

#define NumRandomPerThread 64 // mxnet recommended random numbers generated per thread

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

MXReturnValue noisyForwardCPU(const std::unordered_map<std::string, std::string>& attrs,
                              std::vector<MXTensor>* inputs,
                              std::vector<MXTensor>* outputs,
                              const OpResource& res) {
  float* in_data = inputs->at(0).data<float>();
  float* out_data = outputs->at(0).data<float>();

  mx_cpu_rand_t* states = res.get_cpu_rand_states();
  std::normal_distribution<float> dist_normal;

  for (int i=0; i<inputs->at(0).size(); ++i) {
    float noise = dist_normal(*states);
    out_data[i] = in_data[i] + noise > 0 ? in_data[i] + noise : 0;
  }
  return MX_SUCCESS;
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

REGISTER_OP(my_noisy_relu)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setForward(noisyForwardCPU, "cpu")
.setForward(noisyForwardGPU, "gpu")
.setBackward(backwardCPU, "cpu")
.setBackward(backwardGPU, "gpu");

MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
