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

void transpose(MXTensor& src, MXTensor& dst, const OpResource& res) {
  MXSparse* A = src.data<MXSparse>();
  MXSparse* B = dst.data<MXSparse>(); 
  std::vector<int64_t> shape = src.shape;
  int64_t h = shape[0];
  int64_t w = shape[1];
  if(src.stype == kCSRStorage) {
    float *Aval = (float*) (A->data);
    // Here we need one more element to help calculate index(line 57).
    std::vector<int64_t> rowPtr(w + 2, 0);
    // count column
    for(int i = 0; i < A->data_len; i++) {
      rowPtr[A->indices[i] + 2]++;
    }
    // Accumulated sum. After this for loop, rowPtr[1:w+2) stores the correct 
    // result of transposed rowPtr.
    for(int i = 2; i < rowPtr.size(); i++) {
      rowPtr[i] += rowPtr[i - 1];
    }
    
    // Alloc memory for sparse data, where 0 is the index
    // of B in output vector.
    res.alloc_sparse(B, 0, A->data_len, w + 1);
    float *Bval = (float*) (B->data);
    for(int i = 0; i < h; i++) {
      for(int j = A->indptr[i]; j < A->indptr[i + 1]; j++) {
        // Helps calculate index and after that rowPtr[0:w+1) stores the 
        // correct result of transposed rowPtr.
        int index = rowPtr[A->indices[j] + 1]++;
        Bval[index] = Aval[j];
        B->indices[index] = i;
      }
    }
    memcpy(B->indptr, rowPtr.data(), sizeof(int64_t) * (w + 1));
  }
}

MXReturnValue forward(const std::unordered_map<std::string, std::string>& attrs,
                      std::vector<MXTensor>* inputs,
                      std::vector<MXTensor>* outputs,
                      const OpResource& res) {
  // The data types and storage types of inputs and outputs should be the same.  
  if(inputs->at(0).dtype != outputs->at(0).dtype ||
     inputs->at(0).stype != outputs->at(0).stype) {
    std::cout << "Error! Expected all inputs and outputs to be the same type." 
              << "Found input storage type:" << inputs->at(0).stype
              << " Found output storage type:" << outputs->at(0).stype
              << " Found input data type:" << inputs->at(0).dtype
              << " Found output data type:" << outputs->at(0).dtype << std::endl;
    return MX_FAIL;
  }

  transpose(inputs->at(0), outputs->at(0), res);
  return MX_SUCCESS;
}

MXReturnValue backward(const std::unordered_map<std::string, std::string>& attrs,
                       std::vector<MXTensor>* inputs,
                       std::vector<MXTensor>* outputs,
                       const OpResource& res) {
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
  // validate inputs
  if (intypes->size() != 1) {
    std::cout << "Expected 1 inputs to inferType" << std::endl;
    return MX_FAIL;
  }
  if (intypes->at(0) != kFloat32) {
    std::cout << "Expected input to have float32 type" << std::endl;
    return MX_FAIL;
  }

  outtypes->at(0) = intypes->at(0);
  return MX_SUCCESS;
}

MXReturnValue inferSType(const std::unordered_map<std::string, std::string>& attrs,
                         std::vector<int>* instypes,
                         std::vector<int>* outstypes) {
  if (instypes->at(0) != kCSRStorage) {
    std::cout << "Expected storage type is kCSRStorage" << std::endl;
    return MX_FAIL;
  }
  outstypes->at(0) = instypes->at(0);
  return MX_SUCCESS;
}

MXReturnValue inferShape(const std::unordered_map<std::string, std::string>& attrs,
                         std::vector<std::vector<unsigned int>>* inshapes,
                         std::vector<std::vector<unsigned int>>* outshapes) {
  // validate inputs
  if (inshapes->size() != 1) {
    std::cout << "Expected 1 inputs to inferShape" << std::endl;
    return MX_FAIL;
  }

  outshapes->at(0).push_back(inshapes->at(0)[1]);
  outshapes->at(0).push_back(inshapes->at(0)[0]);
  return MX_SUCCESS;
}

REGISTER_OP(my_transposecsr)
.setForward(forward, "cpu")
.setBackward(backward, "cpu")
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferSType(inferSType)
.setInferShape(inferShape);

/* ------------------------------------------------------------------------- */

class MyStatefulTransposeCSR : public CustomStatefulOp {
  public:
    explicit MyStatefulTransposeCSR(int count,
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

  private:
    int count;
    const std::unordered_map<std::string, std::string> attrs_;
};

MXReturnValue createOpState(const std::unordered_map<std::string, std::string>& attrs,
                            CustomStatefulOp** op_inst) {
  // testing passing of keyword arguments
  int count = attrs.count("test_kw") > 0 ? std::stoi(attrs.at("test_kw")) : 0;
  // creating stateful operator instance
  *op_inst = new MyStatefulTransposeCSR(count, attrs);
  std::cout << "Info: stateful operator created" << std::endl;
  return MX_SUCCESS;
}

REGISTER_OP(my_state_transposecsr)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferSType(inferSType)
.setInferShape(inferShape)
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
