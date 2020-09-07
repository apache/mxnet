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
 * \file np_moments_op.cc
 * \brief move some op here from np_reduce_op_value.cc.
 */

#if MXNET_USE_TVM_OP
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP

#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyMomentsParam);
DMLC_REGISTER_PARAMETER(NumpyWeightedAverageParam);

inline bool NumpyMomentsShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape> *in_attrs,
                              std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyMomentsParam& param = nnvm::get<NumpyMomentsParam>(attrs.parsed);
  mxnet::TShape out_shape = NumpyReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, out_shape);

  return shape_is_known(out_attrs->at(0)) && shape_is_known(out_attrs->at(1));
}

inline bool NumpyMomentsType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_attrs,
                             std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);
  const NumpyMomentsParam &param = nnvm::get<NumpyMomentsParam>(attrs.parsed);

  if (param.dtype.has_value()) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  }
  TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0));

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_npi_std)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<NumpyMomentsParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyMomentsShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyMomentsType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"std", "mean"};
  })
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
  [](const NodeAttrs& attrs) {
    return 1;
  })
.add_argument("a", "NDArray-or-Symbol", "The input")
.add_arguments(NumpyMomentsParam::__FIELDS__())
.set_attr<FCompute>("FCompute<cpu>", NumpyMomentsForward<cpu, true>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

NNVM_REGISTER_OP(_npi_var)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<NumpyMomentsParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyMomentsShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyMomentsType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"var", "mean"};
  })
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
  [](const NodeAttrs& attrs) {
    return 1;
  })
.add_argument("a", "NDArray-or-Symbol", "The input")
.add_arguments(NumpyMomentsParam::__FIELDS__())
.set_attr<FCompute>("FCompute<cpu>", NumpyMomentsForward<cpu, false>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

inline bool NumpyWeightedAverageType(const nnvm::NodeAttrs& attrs,
                                     std::vector<int> *in_attrs,
                                     std::vector<int> *out_attrs) {
  const auto &param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), (param.weighted ? 2U : 1U));
  CHECK_EQ(out_attrs->size(), 2U);

  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  if (param.weighted) {
    TYPE_ASSIGN_CHECK(*in_attrs, 1, in_attrs->at(0));
  }
  TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0));

  return in_attrs->at(0) != -1 && out_attrs->at(0) != -1 &&
         (!param.weighted || (in_attrs->at(1) != -1)) &&
         out_attrs->at(1) != -1;
}

NNVM_REGISTER_OP(_npi_average)
.set_num_inputs(
  [](const NodeAttrs& attrs) {
    const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
    return param.weighted ? 2 : 1;
  })
.set_num_outputs(2)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
  [](const NodeAttrs& attrs) {
    const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
    return param.returned ? 2 : 1;
  })
.set_attr_parser(ParamParser<NumpyWeightedAverageParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyWeightedAverageShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyWeightedAverageType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
    return param.weighted ?
    std::vector<std::string>{"a", "weights"} :
    std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "The input")
.add_argument("weights", "NDArray-or-Symbol", "The weights to calculate average")
.add_arguments(NumpyWeightedAverageParam::__FIELDS__())
.set_attr<FCompute>("FCompute<cpu>", NumpyWeightedAverageForward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_np_average"});

NNVM_REGISTER_OP(_backward_np_average)
.set_num_outputs(
  [](const NodeAttrs& attrs) {
    const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
    return param.weighted ? 2 : 1;
  })
.set_attr_parser(ParamParser<NumpyWeightedAverageParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_num_inputs(
  [](const NodeAttrs& attrs) {
    const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
    return param.weighted ? 6 : 5;
  })
.set_attr<FCompute>("FCompute<cpu>", NumpyWeightedAverageBackward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
});

}  // namespace op
}  // namespace mxnet
