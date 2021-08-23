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

#ifndef __EXAMPLE__RELU_LIB_H__
#define __EXAMPLE__RELU_LIB_H__

#include <iostream>
#include "mxnet/lib_api.h"

using namespace mxnet::ext;

#define NumThreadPerBlock 256 // mxnet recommended cuda thread number per block
#define NumRandomPerThread 64 // mxnet recommended random numbers generated per thread

class MyStatefulReluCPU : public CustomStatefulOp {
  public:
   explicit MyStatefulReluCPU(const std::unordered_map<std::string, std::string>& attrs);

   MXReturnValue Forward(std::vector<MXTensor>* inputs,
                         std::vector<MXTensor>* outputs,
                         const OpResource& op_res);
   MXReturnValue Backward(std::vector<MXTensor>* inputs,
                          std::vector<MXTensor>* outputs,
                          const OpResource& op_res);

  private:
    const std::unordered_map<std::string, std::string> attrs_;
};

class MyStatefulReluGPU : public CustomStatefulOp {
  public:
   explicit MyStatefulReluGPU(const std::unordered_map<std::string, std::string>& attrs);

    MXReturnValue Forward(std::vector<MXTensor>* inputs,
                          std::vector<MXTensor>* outputs,
                          const OpResource& op_res);
    
    MXReturnValue Backward(std::vector<MXTensor>* inputs,
                           std::vector<MXTensor>* outputs,
                           const OpResource& op_res);
    
  private:
    const std::unordered_map<std::string, std::string> attrs_;
};

MXReturnValue forwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                         std::vector<MXTensor>* inputs,
                         std::vector<MXTensor>* outputs,
                         const OpResource& res);

MXReturnValue backwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                          std::vector<MXTensor>* inputs,
                          std::vector<MXTensor>* outputs,
                          const OpResource& res);

/*
 * Below is noisy ReLU operator example
 * noisy ReLU is made from ReLU extended to include Gaussian noise
 * forward - add Gaussian noise generated from normal distribution to each unit
 * backward - gradient doesn't need to change since noise is constant
 */

MXReturnValue noisyForwardGPU(const std::unordered_map<std::string, std::string>& attrs,
                              std::vector<MXTensor>* inputs,
                              std::vector<MXTensor>* outputs,
                              const OpResource& res);

#endif
