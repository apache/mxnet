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
 *  Copyright (c) 2019 by Contributors
 * \file np_elemwise_unary_op_basic.h
 * \brief Function definition of elementwise unary operators
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_ELEMWISE_UNARY_OP_BASIC_H_
#define MXNET_OPERATOR_NUMPY_NP_ELEMWISE_UNARY_OP_BASIC_H_

#if MXNET_USE_TVM_OP
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP
#include <vector>
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

#if MXNET_USE_TVM_OP
static constexpr int max_dim = 5;

inline bool IsIntType(const int dtype) {
  return (dtype == mshadow::kUint8 ||
          dtype == mshadow::kInt32 ||
          dtype == mshadow::kInt8 ||
          dtype == mshadow::kInt64);
}

TBlob padding(const TBlob& tblob, const int& max_dim) {
  TShape tshape(max_dim, 1);
  int ndim = tblob.shape_.ndim();
  for (int i = max_dim - ndim; i < max_dim; ++i) {
    tshape[i] = tblob.size(i - max_dim + ndim);
  }
  return tblob.reshape(tshape);
}

template<const char* func>
void TVMOpSincCompute(const nnvm::NodeAttrs& attrs,
                      const mxnet::OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (outputs[0].Size() == 0U) return;
  TBlob idata, odata;
  idata = padding(inputs[0], max_dim);
  odata = padding(outputs[0], max_dim);
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {idata, odata});

}

template<const char* func>
void TVMSincBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (IsIntType(outputs[0].type_flag_)) {
    LOG(FATAL) << "This Operator only support float dtype in backward, if you want"
                  "to get backward gradient, you should transform input type for floating point.";
  }
  if (inputs[0].Size() == 0U) return;
  using namespace mshadow;
  TBlob out_grad;
  TBlob in_data;
  TBlob out_data;
  TBlob in_grad;
  out_grad = padding(inputs[0], max_dim);
  in_data = padding(inputs[1], max_dim);
  out_data = padding(inputs[2], max_dim);
  in_grad = padding(outputs[0], max_dim);
  std::string funcname = std::string(func);
  funcname += "req_";
  MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
    if (req_type == kWriteTo) {
      funcname += "kWriteTo";
    } else {
      funcname += "kAddTo";
    }
    tvm::runtime::TVMOpModule::Get()->Call(funcname, ctx,
                                           {out_grad, in_data, out_data, in_grad, in_grad});
  })
}
#endif  // MXNET_USE_TVM_OP

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_ELEMWISE_UNARY_OP_BASIC_H_
