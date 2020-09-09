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

REGISTER_OP(my_relu)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setForward(forwardCPU, "cpu")
.setForward(forwardGPU, "gpu")
.setBackward(backwardCPU, "cpu")
.setBackward(backwardGPU, "gpu");


MyStatefulReluCPU::MyStatefulReluCPU(const std::unordered_map<std::string, std::string>& attrs)
  : attrs_(attrs) {}

MXReturnValue MyStatefulReluCPU::Forward(std::vector<MXTensor>* inputs,
                                         std::vector<MXTensor>* outputs,
                                         const OpResource& op_res) {
  return forwardCPU(attrs_, inputs, outputs, op_res);
}

MXReturnValue MyStatefulReluCPU::Backward(std::vector<MXTensor>* inputs,
                                          std::vector<MXTensor>* outputs,
                                          const OpResource& op_res) {
  return backwardCPU(attrs_, inputs, outputs, op_res);
}

MyStatefulReluGPU::MyStatefulReluGPU(const std::unordered_map<std::string, std::string>& attrs)
  : attrs_(attrs) {}

MXReturnValue MyStatefulReluGPU::Forward(std::vector<MXTensor>* inputs,
                                         std::vector<MXTensor>* outputs,
                                         const OpResource& op_res) {
  return forwardGPU(attrs_, inputs, outputs, op_res);
}

MXReturnValue MyStatefulReluGPU::Backward(std::vector<MXTensor>* inputs,
                                          std::vector<MXTensor>* outputs,
                                          const OpResource& op_res) {
  return backwardGPU(attrs_, inputs, outputs, op_res);
}


MXReturnValue createOpStateCPU(const std::unordered_map<std::string, std::string>& attrs,
                               const MXContext& ctx,
                               const std::vector<std::vector<unsigned int> >& in_shapes,
                               const std::vector<int> in_types,
                               CustomStatefulOp** op_inst) {
  *op_inst = new MyStatefulReluCPU(attrs);
  return MX_SUCCESS;
}

MXReturnValue createOpStateGPU(const std::unordered_map<std::string, std::string>& attrs,
                               const MXContext& ctx,
                               const std::vector<std::vector<unsigned int> >& in_shapes,
                               const std::vector<int> in_types,
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

REGISTER_OP(my_noisy_relu)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setForward(noisyForwardCPU, "cpu")
.setForward(noisyForwardGPU, "gpu")
.setBackward(backwardCPU, "cpu")
.setBackward(backwardGPU, "gpu");

MXReturnValue initialize(int version) {
  if (version >= 20000) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    MX_ERROR_MSG << "MXNet version " << version << " not supported";
    return MX_FAIL;
  }
}
