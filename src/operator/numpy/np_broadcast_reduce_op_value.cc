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
 * \file np_reduce_op_value.cc
 * \brief CPU Implementation of broadcast and reduce functions based on value.
 */

#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyReduceAxesParam);
DMLC_REGISTER_PARAMETER(NumpyReduceAxesNoDTypeParam);
DMLC_REGISTER_PARAMETER(NumpyMomentsParam);

inline bool NumpySumType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyReduceAxesParam &param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);

  if (param.dtype.has_value()) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  }

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_np_sum)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesShape)
.set_attr<nnvm::FInferType>("FInferType", NumpySumType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "The input")
.add_arguments(NumpyReduceAxesParam::__FIELDS__())
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesCompute<cpu, mshadow_op::sum, true>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_np_sum"});

NNVM_REGISTER_OP(_backward_np_sum)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesBackwardUseNone<cpu>);

inline bool NumpyReduceAxesNoDTypeType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_np_max)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesNoDTypeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesNoDTypeShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyReduceAxesNoDTypeType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "The input")
.add_arguments(NumpyReduceAxesNoDTypeParam::__FIELDS__())
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesNoDTypeCompute<cpu, mshadow::red::maximum>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_np_max"});

NNVM_REGISTER_OP(_backward_np_max)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesNoDTypeParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesNoDTypeBackward<cpu, mshadow_op::eq>);

NNVM_REGISTER_OP(_np_min)
.describe(R"code()code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesNoDTypeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesNoDTypeShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyReduceAxesNoDTypeType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
[](const NodeAttrs& attrs) {
return std::vector<std::string>{"a"};
})
.add_argument("a", "NDArray-or-Symbol", "The input")
.add_arguments(NumpyReduceAxesNoDTypeParam::__FIELDS__())
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesNoDTypeCompute<cpu, mshadow::red::minimum>)
.set_attr<FResourceRequest>("FResourceRequest",
[](const NodeAttrs& attrs) {
return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_np_min"});

NNVM_REGISTER_OP(_backward_np_min)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesNoDTypeParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_num_inputs(3)
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesNoDTypeBackward<cpu, mshadow_op::eq>);

NNVM_REGISTER_OP(_np_prod)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesShape)
.set_attr<nnvm::FInferType>("FInferType", NumpySumType)
.add_arguments(NumpyReduceAxesParam::__FIELDS__())
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "The input")
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesCompute<cpu, mshadow_op::product, true>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_np_prod"});

NNVM_REGISTER_OP(_backward_np_prod)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesBackwardUseInOut<cpu, mshadow_op::rdiv>);

inline bool IsIntType(const int dtype) {
  return (dtype == mshadow::kUint8 ||
          dtype == mshadow::kInt32 ||
          dtype == mshadow::kInt8 ||
          dtype == mshadow::kInt64);
}

inline bool NumpyMeanType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_attrs,
                          std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const NumpyReduceAxesParam &param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);

  if (param.dtype.has_value()) {
    if (IsIntType(in_attrs->at(0)) && !IsIntType(param.dtype.value())) {
      LOG(FATAL) << "Output cannot be float type when input is integer type for now";
    }
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  }

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

NNVM_REGISTER_OP(_npi_mean)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyMeanType)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.add_argument("a", "NDArray-or-Symbol", "The input")
.add_arguments(NumpyReduceAxesParam::__FIELDS__())
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesCompute<cpu, mshadow_op::sum, true, true>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_np_mean"});

NNVM_REGISTER_OP(_backward_np_mean)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyReduceAxesParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_num_inputs(1)
.set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesBackwardUseNone<cpu, true>);

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
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

bool NumpyBroadcastToShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::shape_is_known(ishape)) return false;
  const BroadcastToParam& param = nnvm::get<BroadcastToParam>(attrs.parsed);
  CHECK(mxnet::shape_is_known(param.shape))
      << "the objective shape for broadcasting array must be known";
  CHECK_LE(ishape.ndim(), param.shape.ndim())
      << "shape " << ishape << " is not broadcastable to " << param.shape;
  for (int i = param.shape.ndim() - 1; i >= 0; --i) {
    int j = i - param.shape.ndim() + ishape.ndim();
    if (j < 0) break;
    CHECK(ishape[j] == param.shape[i] || ishape[j] == 1)
        << "shape " << ishape << " is not broadcastable to " << param.shape;
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, param.shape);
  return true;
}

NNVM_REGISTER_OP(_np_broadcast_to)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"array"};
  })
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n,
    const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_np_broadcast_to", n, ograds, {}, n->attrs.dict);
  })
.add_argument("array", "NDArray-or-Symbol", "The input")
.set_attr_parser(ParamParser<BroadcastToParam>)
.add_arguments(BroadcastToParam::__FIELDS__())
.set_attr<mxnet::FInferShape>("FInferShape", NumpyBroadcastToShape)
.set_attr<FCompute>("FCompute<cpu>", NumpyBroadcastToForward<cpu>);

NNVM_REGISTER_OP(_backward_np_broadcast_to)
.set_attr_parser(ParamParser<BroadcastToParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyBroadcastToBackward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  });

}  // namespace op
}  // namespace mxnet
