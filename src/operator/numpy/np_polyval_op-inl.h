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
 * \file np_polyval_op-inl.h
 * \brief Implement polyval
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_POLYVAL_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_POLYVAL_OP_INL_H_

#ifdef MXNET_USE_TVM_OP

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include <string>
#include <vector>
#include <type_traits>
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tvmop/op_module.h"

namespace mxnet {
namespace op {

template<bool cuda>
void TVMPolyvalForward(const nnvm::NodeAttrs& attrs,
                       const mxnet::OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  std::string funcname = "polyval";
  MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, DType, {
    if (std::is_integral<DType>::value) {
      funcname += "_horner";
    }
    if (cuda) {
      funcname += "_cuda";
    }
    tvm::runtime::TVMOpModule::Get()->Call(
        funcname, ctx, {inputs[0], inputs[1], outputs[0]});
  })
}

template<bool cuda>
void TVMPolyvalBackward(const nnvm::NodeAttrs& attrs,
                        const mxnet::OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  MSHADOW_TYPE_SWITCH(inputs[1].type_flag_, PType, {
    MSHADOW_TYPE_SWITCH(inputs[2].type_flag_, XType, {
      if (!std::is_same<PType, XType>::value ||
      std::is_integral<PType>::value ||
      std::is_integral<XType>::value) {
        return;
      }
    })
  })
  std::string funcname = "backward_polyval";
  if (cuda) {
    funcname += "_cuda";
  }
  funcname += "req_";
  MXNET_ASSIGN_REQ_SWITCH(req[0], req_p, {
    if (req_p == kWriteTo) {
      funcname += "kWriteTo";
    } else {
      funcname += "kAddTo";
    }
  })
  tvm::runtime::TVMOpModule::Get()->Call(
      funcname, ctx,
      {inputs[0], inputs[1], inputs[2],
       outputs[0], outputs[0],
       outputs[1], outputs[1]});
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TVM_OP
#endif  // MXNET_OPERATOR_NUMPY_NP_POLYVAL_OP_INL_H_
