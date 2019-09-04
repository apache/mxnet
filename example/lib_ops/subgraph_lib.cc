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
 * \file subgraph_lib.cc
 * \brief subgraph operator implementation
 * library file
 */

#include <iostream>
#include "lib_api.h"

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

MXReturnValue parseAttrs(std::map<std::string,std::string> attrs,
               int* num_in, int* num_out) {
  *num_in = 2;
  *num_out = 1;
  return MX_SUCCESS;
}

MXReturnValue inferType(std::map<std::string,std::string> attrs, std::vector<int> &intypes,
              std::vector<int> &outtypes) {
  outtypes[0] = intypes[0];
  return MX_SUCCESS;
}

MXReturnValue inferShape(std::map<std::string,std::string> attrs, std::vector<std::vector<unsigned int>> &inshapes,
               std::vector<std::vector<unsigned int>> &outshapes) {
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

MXReturnValue myFCompute(std::map<std::string,std::string> attrs,
               std::vector<MXTensor> inputs, std::vector<MXTensor> outputs,
               OpResource res) {
  //extract data pointers from tensors
  float* input1 = inputs[0].getData<float>();
  float* input2 = inputs[1].getData<float>();
  float* output = outputs[0].getData<float>();
  //set tensor shapes
  unsigned n = inputs[0].shape[0];
  unsigned k = inputs[0].shape[1];
  unsigned m = inputs[1].shape[1];

  gemm(input1, input2, output, n, k, m);

  return MX_SUCCESS;
}

MXReturnValue mutateInputs(std::map<std::string,std::string> attrs,
               std::vector<int> &input_indices) {
  input_indices.push_back(1);
  std::cout << "the 1st input is marked as mutate input by library author" << std::endl;
  return MX_SUCCESS;
}

REGISTER_OP(subgraph_op)
.setFCompute(myFCompute)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setMutateInputs(mutateInputs);

MXReturnValue initialize(int version) {
  if (version >= 10400) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}

