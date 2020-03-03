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
 * \file transsparse_lib.cc
 * \brief Sample 2D transpose custom operator.
 */

#include <iostream>
#include "lib_api.h"

void transpose(MXTensor src, MXTensor dst) {
  MXInSparse* A = src.data<MXInSparse>();
  MXOutSparse* B = dst.data<MXOutSparse>(); 

  std::vector<int64_t> shape = src.shape;
  int64_t h = shape[0];
  int64_t w = shape[1];
  if(src.stype == kCSRStorage) {
    // To do: fix type.
    float *Aval = (float*) (A->data);
    std::vector<int64_t> rowPtr(w + 2, 0);

    // count column
    for(int i = 0; i < A->data_len; i++) {
      rowPtr[A->indices[i] + 2]++;
    }

    // Accumulated sum
    for(int i = 2; i < rowPtr.size(); i++) {
      rowPtr[i] += rowPtr[i - 1];
    }

    // Get the dst sparse matrix.
    B->m_col_idx.resize(A->data_len);
    B->m_row_ptr.resize(w + 1);
    B->m_data.resize(A->data_len);
    for(int i = 0; i < h; i++) {
      for(int j = A->indptr[i]; j < A->indptr[i + 1]; j++) {
        int index = rowPtr[A->indices[j] + 1]++;
	B->m_data[index] = Aval[j];
	B->m_col_idx[index] = i;
      }
    }
    memcpy(B->m_row_ptr.data(), rowPtr.data(), sizeof(int64_t) * (w + 1));
  }
}

MXReturnValue forward(std::map<std::string, std::string> attrs,
                      std::vector<MXTensor> inputs,
                      std::vector<MXTensor> outputs,
                      OpResource res) {

  // The data types and storage types of inputs and outputs should be the same.  
  if(inputs[0].dtype != outputs[0].dtype || inputs[0].stype != outputs[0].stype)
    return MX_FAIL;

  transpose(inputs[0], outputs[0]);
  return MX_SUCCESS;
}

MXReturnValue backward(std::map<std::string, std::string> attrs,
                       std::vector<MXTensor> inputs,
                       std::vector<MXTensor> outputs,
                       OpResource res) {
  return MX_SUCCESS;
}

MXReturnValue parseAttrs(std::map<std::string, std::string> attrs, int* num_in, int* num_out) {
  *num_in = 1;
  *num_out = 1;
  return MX_SUCCESS;
}

MXReturnValue inferType(std::map<std::string, std::string> attrs,
                        std::vector<int> &intypes,
                        std::vector<int> &outtypes) {
  // validate inputs
  if (intypes.size() != 1) {
    std::cout << "Expected 1 inputs to inferType" << std::endl;
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
  if (inshapes.size() != 1) {
    std::cout << "Expected 1 inputs to inferShape" << std::endl;
    return MX_FAIL;
  }

  outshapes[0].push_back(inshapes[0][1]);
  outshapes[0].push_back(inshapes[0][0]);
  return MX_SUCCESS;
}

REGISTER_OP(my_transcsr)
.setForward(forward, "cpu")
.setBackward(backward, "cpu")
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape);

/* ------------------------------------------------------------------------- */

class MyStatefulTransCSR : public CustomStatefulOp {
 public:
  explicit MyStatefulTransCSR(int count) : count(count) {}

  MXReturnValue Forward(std::vector<MXTensor> inputs,
                        std::vector<MXTensor> outputs,
                        OpResource op_res) {
    std::cout << "Info: keyword + number of forward: " << ++count << std::endl;
    std::map<std::string, std::string> attrs;
    return forward(attrs, inputs, outputs, op_res);
  }

  MXReturnValue Backward(std::vector<MXTensor> inputs,
                         std::vector<MXTensor> outputs,
                         OpResource op_res) {
    std::map<std::string, std::string> attrs;
    return backward(attrs, inputs, outputs, op_res);
  }

  ~MyStatefulTransCSR() {}

 private:
  int count;
};

MXReturnValue createOpState(std::map<std::string, std::string> attrs,
                            CustomStatefulOp** op_inst) {
  // testing passing of keyword arguments
  int count = attrs.count("test_kw") > 0 ? std::stoi(attrs["test_kw"]) : 0;
  // creating stateful operator instance
  *op_inst = new MyStatefulTransCSR(count);
  std::cout << "Info: stateful operator created" << std::endl;
  return MX_SUCCESS;
}

MXReturnValue mutateInputs(std::map<std::string, std::string> attrs,
                           std::vector<int> &input_indices) {
  // input_indices.push_back(1);  // mark mutate input
  return MX_SUCCESS;
}

REGISTER_OP(state_transcsr)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setMutateInputs(mutateInputs)
.setCreateOpState(createOpState, "cpu");

MXReturnValue initialize(int version) {
  if (version >= 10400) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
