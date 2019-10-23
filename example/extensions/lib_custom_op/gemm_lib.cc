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
MXReturnValue forward(std::map<std::string, std::string> attrs,
                      std::vector<MXTensor> inputs,
                      std::vector<MXTensor> outputs,
                      OpResource res) {
  // simple example of using runtime data type
  if (inputs[0].dtype == kFloat32) {
    typedef float DType;
    // extract data pointers from tensors
    // if using dltensor repr, below lines can be changed to something like
    // DType* A = reinterpret_cast<DType*>(inputs[0].dltensor.data);
    DType* A = inputs[0].data<DType>();
    DType* B = inputs[1].data<DType>();
    DType* C = outputs[0].data<DType>();
    // set tensor shapes
    unsigned n = inputs[0].shape[0];
    unsigned k = inputs[0].shape[1];
    unsigned m = inputs[1].shape[1];

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
MXReturnValue backward(std::map<std::string, std::string> attrs,
                       std::vector<MXTensor> inputs,
                       std::vector<MXTensor> outputs,
                       OpResource res) {
  // extract data pointers from tensors
  float* dC = inputs[0].data<float>();
  float* A = inputs[1].data<float>();
  float* B = inputs[2].data<float>();
  float* dA = outputs[0].data<float>();
  float* dB = outputs[1].data<float>();
  // set tensor shapes
  unsigned n = inputs[1].shape[0];
  unsigned k = inputs[1].shape[1];
  unsigned m = inputs[2].shape[1];
  // allocate temporary workspace memory through resource manager
  // for multiple arrays better to request a big memory pool
  void *workspace = res.alloc((k*n + m*k) * sizeof(float));
  float *At = static_cast<float*>(workspace);
  float *Bt = static_cast<float*>(workspace) + (k*n);

  transpose(A, At, k, n);
  transpose(B, Bt, m, k);
  gemm(dC, Bt, dA, n, m, k);
  gemm(At, dC, dB, k, n, m);

  return MX_SUCCESS;
}

MXReturnValue parseAttrs(std::map<std::string, std::string> attrs, int* num_in, int* num_out) {
  *num_in = 2;
  *num_out = 1;
  return MX_SUCCESS;
}

MXReturnValue inferType(std::map<std::string, std::string> attrs,
                        std::vector<int> &intypes,
                        std::vector<int> &outtypes) {
  // validate inputs
  if (intypes.size() != 2) {
    std::cout << "Expected 2 inputs to inferType" << std::endl;
    return MX_FAIL;
  }
  for (unsigned i = 0; i < intypes.size(); i++) {
    if (intypes[i] != kFloat32) {
      std::cout << "Expected input " << i << " to have float32 type" << std::endl;
      return MX_FAIL;
    }
  }

  outtypes[0] = intypes[0];
  return MX_SUCCESS;
}

MXReturnValue inferShape(std::map<std::string, std::string> attrs,
                         std::vector<std::vector<unsigned int>> &inshapes,
                         std::vector<std::vector<unsigned int>> &outshapes) {
  // validate inputs
  if (inshapes.size() != 2) {
    std::cout << "Expected 2 inputs to inferShape" << std::endl;
    return MX_FAIL;
  }
  if (inshapes[0].size() != 2 || inshapes[1].size() != 2) {
    std::cout << "Expected 2D matrices for both inputs to inferShape" << std::endl;
    return MX_FAIL;
  }

  unsigned n = inshapes[0][0];
  unsigned k = inshapes[0][1];
  unsigned kk = inshapes[1][0];
  unsigned m = inshapes[1][1];
  if (k != kk) {
    std::cout << "Exected first input axis 1 equals to second input axis 0" << std::endl;
    return MX_FAIL;
  }

  outshapes[0] = {n, m};
  return MX_SUCCESS;
}

REGISTER_OP(my_gemm)
.setForward(forward)
.setBackward(backward)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape);

/* ------------------------------------------------------------------------- */

class MyStatefulGemm : public CustomStatefulOp {
 public:
  explicit MyStatefulGemm(int count) : count(count) {}

  MXReturnValue Forward(std::vector<MXTensor> inputs,
                        std::vector<MXTensor> outputs,
                        OpResource op_res) {
    ++count;
    std::cout << "Info: keyword + number of forward: " << count << std::endl;
    std::map<std::string, std::string> attrs;
    return forward(attrs, inputs, outputs, op_res);
  }

  MXReturnValue Backward(std::vector<MXTensor> inputs,
                         std::vector<MXTensor> outputs,
                         OpResource op_res) {
    std::map<std::string, std::string> attrs;
    return backward(attrs, inputs, outputs, op_res);
  }

  ~MyStatefulGemm() {}

 private:
  int count;
};

MXReturnValue createOpState(std::map<std::string, std::string> attrs,
                            CustomStatefulOp** op_inst) {
  int count = 0;
  if (attrs.count("test_kw") > 0)
    count = std::stoi(attrs["test_kw"]);
  *op_inst = new MyStatefulGemm(count);
  std::cout << "Info: stateful operator created" << std::endl;
  return MX_SUCCESS;
}

MXReturnValue mutateInputs(std::map<std::string, std::string> attrs,
                           std::vector<int> &input_indices) {
  // input_indices.push_back(1);  // mark mutate input
  return MX_SUCCESS;
}

REGISTER_OP(state_gemm)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setMutateInputs(mutateInputs)
.setCreateOpState(createOpState);

MXReturnValue initialize(int version) {
  if (version >= 10400) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
