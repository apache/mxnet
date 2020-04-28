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
 * Copyright (c) 2019 by Contributors
 * \file gemm_lib.cc
 * \brief Sample 2D gemm custom operator implementation library file
 */

#include <iostream>
#include "lib_api.h"

// main matrix multiplication routine
void gemm(const float* A, const float* B, float* C,
          const unsigned n, const unsigned k, const unsigned m) {
  unsigned i, j, kk;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      C[i*m+j] = 0;
      for (kk = 0; kk < k; kk++) {
        C[i*m+j] += A[i*k+kk] * B[kk*m+j];
      }
    }
  }
}

void transpose(const float* A, float* At, const unsigned n, const unsigned m) {
  unsigned i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      At[i*m+j] = A[j*n+i];
    }
  }
}

/*
 * Executes C = A * B
 * inputs[0] = A; inputs[1] = B; outputs[0] = C
 */
MXReturnValue forward(const std::unordered_map<std::string, std::string>& attrs,
                      std::vector<MXTensor>* inputs,
                      std::vector<MXTensor>* outputs,
                      const OpResource& res) {
  // simple example of using runtime data type
  if (inputs->at(0).dtype == kFloat32) {
    typedef float DType;
    // extract data pointers from tensors
    // if using dltensor repr, below lines can be changed to something like
    // DType* A = reinterpret_cast<DType*>(inputs[0].dltensor.data);
    DType* A = inputs->at(0).data<DType>();
    DType* B = inputs->at(1).data<DType>();
    DType* C = outputs->at(0).data<DType>();
    // set tensor shapes
    unsigned n = inputs->at(0).shape[0];
    unsigned k = inputs->at(0).shape[1];
    unsigned m = inputs->at(1).shape[1];

    gemm(A, B, C, n, k, m);
  }
  return MX_SUCCESS;
}

/*
 * Executes dA = dC * B.T; Executes dB = A.T * dC
 ***** gradient inputs
 * inputs[0] = dC
 ***** original inputs
 * inputs[1] = A; inputs[2] = B
 ***** original outputs
 * inputs[3] = C
 ***** gradient outputs
 * outputs[0] = dA; outputs[1] = dB
 */
MXReturnValue backward(const std::unordered_map<std::string, std::string>& attrs,
                       std::vector<MXTensor>* inputs,
                       std::vector<MXTensor>* outputs,
                       const OpResource& res) {
  // extract data pointers from tensors
  float* dC = inputs->at(0).data<float>();
  float* A = inputs->at(1).data<float>();
  float* B = inputs->at(2).data<float>();
  float* dA = outputs->at(0).data<float>();
  float* dB = outputs->at(1).data<float>();
  // set tensor shapes
  unsigned n = inputs->at(1).shape[0];
  unsigned k = inputs->at(1).shape[1];
  unsigned m = inputs->at(2).shape[1];
  // allocate temporary workspace memory through resource manager
  // for multiple arrays better to request a big memory pool
  void *workspace = res.alloc_cpu((k*n + m*k) * sizeof(float));
  float *At = static_cast<float*>(workspace);
  float *Bt = static_cast<float*>(workspace) + (k*n);

  transpose(A, At, k, n);
  transpose(B, Bt, m, k);
  gemm(dC, Bt, dA, n, m, k);
  gemm(At, dC, dB, k, n, m);

  return MX_SUCCESS;
}

MXReturnValue parseAttrs(const std::unordered_map<std::string, std::string>& attrs,
                         int* num_in, int* num_out) {
  *num_in = 2;
  *num_out = 1;
  return MX_SUCCESS;
}

MXReturnValue inferType(const std::unordered_map<std::string, std::string>& attrs,
                        std::vector<int> *intypes,
                        std::vector<int> *outtypes) {
  // validate inputs
  if (intypes->size() != 2) {
    std::cout << "Expected 2 inputs to inferType" << std::endl;
    return MX_FAIL;
  }
  for (unsigned i = 0; i < intypes->size(); i++) {
    if (intypes->at(i) != kFloat32) {
      std::cout << "Expected input " << i << " to have float32 type" << std::endl;
      return MX_FAIL;
    }
  }

  outtypes->at(0) = intypes->at(0);
  return MX_SUCCESS;
}

MXReturnValue inferShape(const std::unordered_map<std::string, std::string>& attrs,
                         std::vector<std::vector<unsigned int>>* inshapes,
                         std::vector<std::vector<unsigned int>>* outshapes) {
  // validate inputs
  if (inshapes->size() != 2) {
    std::cout << "Expected 2 inputs to inferShape" << std::endl;
    return MX_FAIL;
  }
  if (inshapes->at(0).size() != 2 || inshapes->at(1).size() != 2) {
    std::cout << "Expected 2D matrices for both inputs to inferShape" << std::endl;
    return MX_FAIL;
  }

  unsigned n = inshapes->at(0)[0];
  unsigned k = inshapes->at(0)[1];
  unsigned kk = inshapes->at(1)[0];
  unsigned m = inshapes->at(1)[1];
  if (k != kk) {
    std::cout << "Exected first input axis 1 equals to second input axis 0" << std::endl;
    return MX_FAIL;
  }

  outshapes->at(0) = {n, m};
  return MX_SUCCESS;
}

REGISTER_OP(my_gemm)
.setForward(forward, "cpu")
.setBackward(backward, "cpu")
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape);

/* ------------------------------------------------------------------------- */

class MyStatefulGemm : public CustomStatefulOp {
 public:
  explicit MyStatefulGemm(int count,
                          const std::unordered_map<std::string, std::string>& attrs)
    : count(count), attrs_(attrs) {}

  MXReturnValue Forward(std::vector<MXTensor>* inputs,
                        std::vector<MXTensor>* outputs,
                        const OpResource& op_res) {
    std::cout << "Info: keyword + number of forward: " << ++count << std::endl;
    return forward(attrs_, inputs, outputs, op_res);
  }

  MXReturnValue Backward(std::vector<MXTensor>* inputs,
                         std::vector<MXTensor>* outputs,
                         const OpResource& op_res) {
    return backward(attrs_, inputs, outputs, op_res);
  }

  ~MyStatefulGemm() {}

 private:
  int count;
  const std::unordered_map<std::string, std::string> attrs_;
};

MXReturnValue createOpState(const std::unordered_map<std::string, std::string>& attrs,
                            CustomStatefulOp** op_inst) {
  // testing passing of keyword arguments
  int count = attrs.count("test_kw") > 0 ? std::stoi(attrs.at("test_kw")) : 0;
  // creating stateful operator instance
  *op_inst = new MyStatefulGemm(count, attrs);
  std::cout << "Info: stateful operator created" << std::endl;
  return MX_SUCCESS;
}

MXReturnValue mutateInputs(const std::unordered_map<std::string, std::string>& attrs,
                           std::vector<int>* input_indices) {
  // input_indices.push_back(1);  // mark mutate input
  return MX_SUCCESS;
}

REGISTER_OP(state_gemm)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setMutateInputs(mutateInputs)
.setCreateOpState(createOpState, "cpu");

MXReturnValue initialize(int version) {
  if (version >= 10700) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
