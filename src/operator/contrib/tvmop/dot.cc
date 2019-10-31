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
 * \file dot.cc
 * \brief
 * \author Haozheng Fan
 */
#ifdef MXNET_USE_TVM_OP
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include <string>
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../tvmop/op_module.h"
#include "../../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

int SplitSch(const ::tvm::runtime::TVMOpConfig& config,
             const ::std::string& name,
             const std::vector<int>& size) {
  const ::tvm::runtime::OtherOptionSpace& space = config.get_space(name);
  int weight = config.get_weight(name);
  int num_space = space.size(), num_size = size.size();
  for (int i = 0; i < num_space; ++i) {
    bool flag = true;
    for (int j = 0; j < num_size; ++j) {
      if (size[j] % space[i].get_val() != 0) {
        flag = false;
        break;
      }
    }
    if (flag) {
      return i * weight;
    }
  }
  return -1;
}

std::string DotSch(const std::string name,
                   const nnvm::NodeAttrs& attrs,
                   const mxnet::ShapeVector& in_attrs,
                   const mxnet::ShapeVector& out_attrs) {
  const ::tvm::runtime::TVMOpConfig& config = tvm::runtime::GetOpConfig(name);
  int m = in_attrs[0][0];
  int k = in_attrs[0][1];
  int n = in_attrs[1][1];
  int idx_bn = SplitSch(config, "bn", {m, n});
  int idx_factor = SplitSch(config, "factor", {k});
  int idx = idx_bn + idx_factor;
  if (idx_bn == -1 || idx_factor == -1) {
    return "fallback";
  }
  return "index_" + std::to_string(idx);
}

void TVMDotForward(const nnvm::NodeAttrs& attrs,
                const mxnet::OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  std::string funcname = "dot";
  std::string sch = DotSch(funcname, attrs, {inputs[0].shape_, inputs[1].shape_},
                           {outputs[0].shape_});
  tvm::runtime::TVMOpModule::Get()->Call(funcname + sch, ctx, {inputs[0], inputs[1], outputs[0]});
}

void TVMDotFallbackForward(const nnvm::NodeAttrs& attrs,
                        const mxnet::OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  std::string funcname = "dot";
  std::string sch = "fallback";
  tvm::runtime::TVMOpModule::Get()->Call(funcname + sch, ctx, {inputs[0], inputs[1], outputs[0]});
}

bool TVMDotShape(const nnvm::NodeAttrs& attrs,
              mxnet::ShapeVector *in_attrs,
              mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);
  CHECK_EQ(a_shape.ndim(), 2U);
  CHECK_EQ(b_shape.ndim(), 2U);
  mxnet::TShape tmp_shape(2, -1);
  tmp_shape[1] = b_shape[0];
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, tmp_shape);

  tmp_shape[0] = a_shape[1];
  tmp_shape[1] = -1;
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, tmp_shape);

  tmp_shape[0] = a_shape[0];
  tmp_shape[1] = b_shape[1];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, tmp_shape);
  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs);
}

NNVM_REGISTER_OP(_contrib_tvm_dot)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs) {
        return std::vector<std::string>{"a", "b"};
      })
    .set_attr<mxnet::FInferShape>("FInferShape", TVMDotShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
    .set_attr<mxnet::FCompute>("FCompute<cpu>", TVMDotForward);

NNVM_REGISTER_OP(_contrib_tvm_dot_fallback)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs) {
        return std::vector<std::string>{"a", "b"};
      })
    .set_attr<mxnet::FInferShape>("FInferShape", TVMDotShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
    .set_attr<mxnet::FCompute>("FCompute<cpu>", TVMDotFallbackForward);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP
