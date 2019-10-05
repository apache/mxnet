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
 * \file np_true_divide.h
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_TRUE_DIVIDE_H_
#define MXNET_OPERATOR_NUMPY_NP_TRUE_DIVIDE_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <vector>
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

inline bool IsIntType(const int dtype) {
  return (dtype == mshadow::kUint8 ||
          dtype == mshadow::kInt32 ||
          dtype == mshadow::kInt8 ||
          dtype == mshadow::kInt64);
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBroadcastBackwardUseInFloat(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs) {
  CHECK(!IsIntType(outputs[0].type_flag_)) << "Cannot compute ingrad if type of input is `int`.\n";
  BinaryBroadcastBackwardUseIn<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
}

template<typename xpu, typename OP>
void BinaryScalarOpComputeFloat(const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const std::vector<TBlob> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<TBlob> &outputs) {
  CHECK(!IsIntType(outputs[0].type_flag_)) << "Cannot compute ingrad if type of input is `int`.\n";
  BinaryScalarOp::Compute<xpu, OP>(attrs, ctx, inputs, req, outputs);
}

template<typename xpu, typename OP>
void BinaryScalarOpBackwardFloat(const nnvm::NodeAttrs &attrs,
                            const OpContext &ctx,
                            const std::vector<TBlob> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<TBlob> &outputs) {
  CHECK(!IsIntType(outputs[0].type_flag_)) << "Cannot compute ingrad if type of input is `int`.\n";
  BinaryScalarOp::Backward<xpu, OP>(attrs, ctx, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_TRUE_DIVIDE_H_
