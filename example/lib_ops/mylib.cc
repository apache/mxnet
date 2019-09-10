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
 * \file mylib.cc
 * \brief Sample custom operator implementation
 * library file
 */

#include <iostream>
#include "lib_api.h"

/*
 * main matrix multiplication routine
 */
void gemm(float* A, float* B, float* C, unsigned n, unsigned k, unsigned m) {
  unsigned i,j,kk;
  for (i=0;i<n;i++) {
    for (j=0;j<m;j++) {
      C[i*m+j] = 0;
      for (kk=0;kk<k;kk++) {
        C[i*m+j] += A[i*k+kk] * B[kk*m+j];
      }
    }
  }
}

void transpose(float* A, float* At, unsigned n, unsigned m) {
  unsigned i,j;
  for (i=0; i < n; i++) {
    for (j=0; j < m; j++) {
      At[i*n+j] = A[j*m+i];
    }
  }
}

/*
 * Executes C = A * B
 * inputs[0] = A
 * inputs[1] = B
 * outputs[0] = C
 */
MXReturnValue forward(std::map<std::string,std::string> attrs,
               std::vector<MXTensor> inputs, std::vector<MXTensor> outputs,
               OpResource res) {
  //validate inputs
  for(int i=0; i<inputs.size(); i++) {
    if(inputs[i].dtype != kFloat32) {
      std::cout << "Expected input " << i << " to have float32 type" << std::endl;
      return MX_FAIL;
    }
  }
  
  //extract data pointers from tensors
  float* A = inputs[0].getData<float>();
  float* B = inputs[1].getData<float>();
  float* C = outputs[0].getData<float>();

  //set tensor shapes
  unsigned n = inputs[0].shape[0];
  unsigned k = inputs[0].shape[1];
  unsigned m = inputs[1].shape[1];

  gemm(A, B, C, n, k, m);
  
  return MX_SUCCESS;
}

/*
 * Executes dA = dC * B.T
 * Executes dB = A.T * dC
 ***** gradient inputs
 * inputs[0] = dC
 ***** original inputs
 * inputs[1] = A
 * inputs[2] = B
 ***** original outputs
 * inputs[3] = C
 ***** gradient outputs
 * outputs[0] = dA
 * outputs[1] = dB
 */
MXReturnValue backward(std::map<std::string,std::string> attrs,
               std::vector<MXTensor> inputs, std::vector<MXTensor> outputs,
               OpResource res) {
  //validate inputs
  for(int i=0; i<inputs.size(); i++) {
    if(inputs[i].dtype != kFloat32) {
      std::cout << "Expected input " << i << " to have float32 type" << std::endl;
      return MX_FAIL;
    }
  }
  
  //extract data pointers from tensors
  float* dC = inputs[0].getData<float>();
  float* A = inputs[1].getData<float>();
  float* B = inputs[2].getData<float>();
  float* dA = outputs[0].getData<float>();
  float* dB = outputs[1].getData<float>();
  //set tensor shapes
  unsigned n = inputs[0].shape[0];
  unsigned k = inputs[0].shape[1];
  unsigned m = inputs[1].shape[1];

  //gemm(input1, input2, output, n, k, m);
  
  return MX_SUCCESS;
}

MXReturnValue parseAttrs(std::map<std::string,std::string> attrs,
               int* num_in, int* num_out) {
  /*
  if(attrs.find("myParam") == attrs.end()) {
    std::cout << "Missing param 'myParam'" << std::endl;
    return 0;
  }
  */
  *num_in = 2;
  *num_out = 1;

  return MX_SUCCESS;
}

MXReturnValue inferType(std::map<std::string,std::string> attrs, std::vector<int> &intypes,
              std::vector<int> &outtypes) {
  // validate inputs
  if (intypes.size() != 2) {
    std::cout << "Expected 2 inputs to inferType" << std::endl;
    return MX_FAIL;
  }

  if (intypes[0] != intypes[1]) {
    std::cout << "Expected 2 inputs to have same data type for inferType" << std::endl;
    return MX_FAIL;
  }

  outtypes[0] = intypes[0];

  std::cout << "intypes[0]=" << intypes[0] << "  outtypes[0]=" << outtypes[0] << std::endl;
  std::cout << "intypes=" << intypes.size() << "  outtypes=" << outtypes.size() << std::endl;

  return MX_SUCCESS;
}

MXReturnValue inferShape(std::map<std::string,std::string> attrs, std::vector<std::vector<unsigned int>> &inshapes,
               std::vector<std::vector<unsigned int>> &outshapes) {
  // validate inputs
  if (inshapes.size() != 2) {
    std::cout << "Expected 2 inputs to inferShape" << std::endl;
    return MX_FAIL;
  }

  if (inshapes[0].size() != 2) {
    std::cout << "Expected 2D for first input to inferShape" << std::endl;
    return MX_FAIL;
  }

  if (inshapes[1].size() != 2) {
    std::cout << "Expected 2D for second input to inferShape" << std::endl;
    return MX_FAIL;
  }
  
  unsigned n = inshapes[0][0];
  unsigned k = inshapes[0][1];
  unsigned kk = inshapes[1][0];
  unsigned m = inshapes[1][1];

  std::cout << "inshapes[0][0]=" << n << "  inshapes[0][1]=" << k << std::endl;
  std::cout << "inshapes[1][0]=" << kk << "  inshapes[1][1]=" << m << std::endl;
  
  if (k != kk)
    return MX_FAIL;
  
  outshapes[0].push_back(n);
  outshapes[0].push_back(m);

  return MX_SUCCESS;
}

REGISTER_OP(gemm)
.setForward(forward)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape);

REGISTER_OP(warpctc)
.setForward(forward)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape);

MXReturnValue initialize(int version) {
  if (version >= 10400) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
