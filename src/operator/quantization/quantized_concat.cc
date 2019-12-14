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
 * Copyright (c) 2018 by Contributors
 * \file quantized_concat.cc
 * \brief
*/

#include "../nn/concat-inl.h"

namespace mxnet {
namespace op {

static bool QuantizedConcatShape(const nnvm::NodeAttrs& attrs, mxnet::ShapeVector* in_shape,
                                 mxnet::ShapeVector* out_shape) {
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), static_cast<size_t>(param_.num_args * 3));
  CHECK_EQ(out_shape->size(), 3U);
  mxnet::TShape dshape;
  index_t size = 0;
  bool has_unknown_dim_size = false;
  int axis = -1;
  for (int i = 0; i < param_.num_args; ++i) {
    mxnet::TShape tmp = (*in_shape)[i];
    if (tmp.ndim() > 0) {
      axis = CheckAxis(param_.dim, tmp.ndim());
      has_unknown_dim_size = !mxnet::dim_size_is_known(tmp, axis) || has_unknown_dim_size;
      size += tmp[axis];
      tmp[axis] = -1;
      shape_assign(&dshape, tmp);
    }
  }

  mxnet::TShape tmp = (*out_shape)[0];
  if (tmp.ndim() > 0) {
    axis = CheckAxis(param_.dim, tmp.ndim());
    tmp[axis] = -1;
    shape_assign(&dshape, tmp);
  }

  if (!mxnet::ndim_is_known(dshape)) return false;

  for (int i = 0; i < param_.num_args; ++i) {
    CHECK(shape_assign(&(*in_shape)[i], dshape))
        << "Incompatible input shape: expected " << dshape << ", got " << (*in_shape)[i];
  }

  if (!has_unknown_dim_size) dshape[axis] = size;
  CHECK(shape_assign(&(*out_shape)[0], dshape))
      << "Incompatible output shape: expected " << dshape << ", got " << (*out_shape)[0];

  for (int i = param_.num_args; i < param_.num_args * 3; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, mxnet::TShape{1});
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape{1});
  SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape{1});
  return shape_is_known(dshape);
}

static bool QuantizedConcatType(const nnvm::NodeAttrs& attrs, std::vector<int>* in_type,
                                std::vector<int>* out_type) {
  const ConcatParam& param_ = nnvm::get<ConcatParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), static_cast<size_t>(param_.num_args * 3));
  CHECK_EQ(out_type->size(), 3U);
  int dtype = mshadow::kUint8;

  for (int i = 0; i < param_.num_args; ++i) {
    if (in_type->at(i) == mshadow::kInt8) {
      dtype = mshadow::kInt8;
    } else {
      TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kUint8);
    }
  }
  TYPE_ASSIGN_CHECK(*out_type, 0, dtype);
  TYPE_ASSIGN_CHECK(*out_type, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_type, 2, mshadow::kFloat32);

  return true;
}

NNVM_REGISTER_OP(_contrib_quantized_concat)
.describe(R"code(Joins input arrays along a given axis.

The dimensions of the input arrays should be the same except the axis along
which they will be concatenated.
The dimension of the output array along the concatenated axis will be equal
to the sum of the corresponding dimensions of the input arrays.
All inputs with different min/max will be rescaled by using largest [min, max] pairs.
If any input holds int8, then the output will be int8. Otherwise output will be uint8.

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  return params.num_args * 3;
})
.set_num_outputs(3)
.set_attr_parser(ParamParser<ConcatParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  const ConcatParam& params = nnvm::get<ConcatParam>(attrs.parsed);
  std::vector<std::string> ret;
  for (int i = 0; i < params.num_args; ++i) {
    ret.push_back(std::string("arg") + std::to_string(i));
  }
  for (int i = 0; i < params.num_args; ++i) {
    ret.push_back(std::string("arg") + std::to_string(i) + "_min");
    ret.push_back(std::string("arg") + std::to_string(i) + "_max");
  }
  return ret;
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "min_output", "max_output"};
})
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<nnvm::FInferType>("FInferType", QuantizedConcatType)
.set_attr<mxnet::FInferShape>("FInferShape", QuantizedConcatShape)
.set_attr<std::string>("key_var_num_args", "num_args")
.add_argument("data", "NDArray-or-Symbol[]", "List of arrays to concatenate")
.add_arguments(ConcatParam::__FIELDS__());

NNVM_REGISTER_OP(Concat)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  nnvm::NodePtr node = nnvm::Node::Create();
  if (param.dim > 0) {
    node->attrs.op = Op::Get("_contrib_quantized_concat");
    node->attrs.name = "quantized_" + attrs.name;
  } else {
    LOG(INFO) << "Currently, quantized concat only supports dim>0, exclude "
              << attrs.name << " which dim is " << param.dim;
    node->attrs.op = nullptr;
    node->attrs.name = attrs.name;
  }
  node->attrs.dict = attrs.dict;
  if (node->op() != nullptr && node->op()->attr_parser != nullptr) {
    node->op()->attr_parser(&(node->attrs));
  }
  return node;
});

}  // namespace op
}  // namespace mxnet
