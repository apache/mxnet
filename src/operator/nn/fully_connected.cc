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
 * \file fully_connected.cc
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_fully_connected-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {

static bool FullyConnectedShape(const nnvm::NodeAttrs& attrs,
                                std::vector<TShape> *in_shape,
                                std::vector<TShape> *out_shape) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  using namespace mshadow;
  if (!param.no_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  TShape dshape = (*in_shape)[fullc::kData];
  TShape oshape = (*out_shape)[0];
  // require data to be known
  if (dshape.ndim() ==  0) return false;

  index_t num_input;
  if (!param.flatten) {
    num_input = dshape[dshape.ndim()-1];
  } else {
    num_input = dshape.ProdShape(1, dshape.ndim());
  }
  SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, Shape2(param.num_hidden, num_input));
  if (!param.no_bias) {
    SHAPE_ASSIGN_CHECK(*in_shape, fullc::kBias, Shape1(param.num_hidden));
  }

  if (!param.flatten) {
    TShape result_shape(dshape);
    result_shape[dshape.ndim()-1] = param.num_hidden;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, result_shape);
  } else {
    SHAPE_ASSIGN_CHECK(*out_shape, 0, Shape2(dshape[0], param.num_hidden));
  }
  if (oshape.ndim() != 0) {
    dshape[0] = oshape[0];
    SHAPE_ASSIGN_CHECK(*in_shape, fullc::kData, dshape);
  }
  return true;
}

static bool FullyConnectedType(const nnvm::NodeAttrs& attrs,
                               std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_GE(in_type->size(), 1U);
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
      attrs, in_type, out_type, -1);
}

struct FullyConnectedGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[fullc::kData]);
    heads.push_back(n->inputs[fullc::kWeight]);
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

DMLC_REGISTER_PARAMETER(FullyConnectedParam);

NNVM_REGISTER_OP(FullyConnected)
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T + b`.

If ``flatten`` is set to be true, then the shapes are:

- **data**: `(batch_size, x1, x2, ..., xn)`
- **weight**: `(num_hidden, x1 * x2 * ... * xn)`
- **bias**: `(num_hidden,)`
- **out**: `(batch_size, num_hidden)`

If ``flatten`` is set to be false, then the shapes are:

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(num_hidden, input_dim)`
- **bias**: `(num_hidden,)`
- **out**: `(x1, x2, ..., xn, num_hidden)`

The learnable parameters include both ``weight`` and ``bias``.

If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const FullyConnectedParam& params = nnvm::get<FullyConnectedParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_num_outputs(1)
.set_attr_parser(ParamParser<FullyConnectedParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  const FullyConnectedParam& params = nnvm::get<FullyConnectedParam>(attrs.parsed);
  if (!params.no_bias) {
    return std::vector<std::string>{"data", "weight", "bias"};
  } else {
    return std::vector<std::string>{"data", "weight"};
  }
})
.set_attr<nnvm::FInferShape>("FInferShape", FullyConnectedShape)
.set_attr<nnvm::FInferType>("FInferType", FullyConnectedType)
.set_attr<FCompute>("FCompute<cpu>", FullyConnectedCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", FullyConnectedGrad{"_backward_FullyConnected"})
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(FullyConnectedParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_FullyConnected)
.set_num_outputs([](const NodeAttrs& attrs) {
  const FullyConnectedParam& params = nnvm::get<FullyConnectedParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{1, 0}};
})
.set_attr_parser(ParamParser<FullyConnectedParam>)
.set_attr<FCompute>("FCompute<cpu>", FullyConnectedGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
