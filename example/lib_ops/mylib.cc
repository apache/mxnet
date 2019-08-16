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
 * Copyright (c) 2015 by Contributors
 * \file mylib.cc
 * \brief Sample library file
 */

#include <iostream>
#include "lib_api.h"

void gemm(double* A, double* B, double* C, unsigned n, unsigned k, unsigned m) {
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

int myFCompute(std::map<std::string,std::string> attrs,
               std::vector<MXTensor> inputs, std::vector<MXTensor> outputs) {

  double* input1 = inputs[0].getData<double>();
  double* input2 = inputs[1].getData<double>();
  double* output = outputs[0].getData<double>();
  unsigned n = inputs[0].shape[0];
  unsigned k = inputs[0].shape[1];
  unsigned m = inputs[1].shape[1];

  gemm(input1, input2, output, n, k, m);
  
  return 1;
}

int parseAttrs(std::map<std::string,std::string> attrs,
               int* num_in, int* num_out) {

  if(attrs.find("myParam") == attrs.end()) {
    std::cout << "Missing param 'myParam'" << std::endl;
    return 0;
  }

  *num_in = 2;
  *num_out = 1;

  return 1; //no error
}

int inferType(std::map<std::string,std::string> attrs, std::vector<int> &intypes,
              std::vector<int> &outtypes) {
  outtypes[0] = intypes[0];
  
  return 1; //no error
}

int inferShape(std::map<std::string,std::string> attrs, std::vector<std::vector<unsigned int>> &inshapes,
               std::vector<std::vector<unsigned int>> &outshapes) {
  unsigned n = inshapes[0][0];
  unsigned k = inshapes[0][1];
  unsigned kk = inshapes[1][0];
  unsigned m = inshapes[1][1];

  if(k != kk) return 0;
  
  outshapes[0].push_back(n);
  outshapes[0].push_back(m);

  return 1; //no error
}

REGISTER_OP(sam)
.setFCompute(myFCompute)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape);

int initialize(int version) {
  if (version >= 10400) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return 1;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return 0;
  }
}

