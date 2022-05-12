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
 * \file dnnl_pow_mul_scalar.cc
 * \brief DNNL pow_mul_scalar operator based on subgraph
 */

#if MXNET_USE_ONEDNN == 1

#include <string>
#include <utility>
#include <vector>

#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/nn/dnnl/dnnl_pow_mul_scalar-inl.h"
#include "operator/subgraph/common.h"

namespace mxnet {
namespace op {

bool DNNLPowMulScalarShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector* in_attrs,
                           mxnet::ShapeVector* out_attrs) {
  return ElemwiseShape<1, 1>(attrs, in_attrs, out_attrs);
}

bool DNNLPowMulScalarType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_types,
                          std::vector<int>* out_types) {
  CHECK_EQ(in_types->size(), 1U);
  CHECK_EQ(out_types->size(), 1U);
  const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
  bool scalar_is_int                  = param.is_int;
  if (common::is_int(in_types->at(0)) && !scalar_is_int) {
    TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
  } else if (in_types->at(0) == mshadow::kBool) {
    TYPE_ASSIGN_CHECK(*out_types, 0, scalar_is_int ? mshadow::kInt32 : mshadow::kFloat32);
  } else {
    TYPE_ASSIGN_CHECK(*out_types, 0, in_types->at(0));
    TYPE_ASSIGN_CHECK(*in_types, 0, out_types->at(0));
  }
  return out_types->at(0) != -1;
}

inline static bool DNNLPowMulScalarStorageType(const nnvm::NodeAttrs& attrs,
                                               const int dev_mask,
                                               DispatchMode* dispatch_mode,
                                               std::vector<int>* in_attrs,
                                               std::vector<int>* out_attrs) {
  return DNNLStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs, out_attrs);
}

NNVM_REGISTER_OP(_sg_onednn_pow_mul_scalar)
    .describe(R"code(_sg_onednn_pow_mul_scalar)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) { return 1; })
    .set_num_outputs([](const NodeAttrs& attrs) { return 1; })
    .set_attr_parser(ParamParser<DNNLPowMulScalarParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"input"};
                                     })
    .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                      [](const NodeAttrs& attrs) {
                                        return std::vector<std::string>{"output"};
                                      })
    .set_attr<mxnet::FInferShape>("FInferShape", DNNLPowMulScalarShape)
    .set_attr<nnvm::FInferType>("FInferType", DNNLPowMulScalarType)
    .set_attr<FInferStorageType>("FInferStorageType", DNNLPowMulScalarStorageType)
    .set_attr<FComputeEx>("FComputeEx<cpu>", DNNLPowMulScalarForward<true>)
    .set_attr<bool>("TIsDNNL", true)
    .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
