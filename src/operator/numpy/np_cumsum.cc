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
 * \file np_cumsum.cc
 * \brief CPU implementation of numpy-compatible cumsum operator
 */

#include "./np_cumsum-inl.h"

namespace mxnet {
namespace op {

inline bool CumsumShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector *in_attrs,
                        mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const CumsumParam &param = nnvm::get<CumsumParam>(attrs.parsed);

  if (param.axis.has_value()) {
    return ElemwiseShape<1, 1>(attrs, in_attrs, out_attrs);
  } else {
    TShape out_shape(1, in_attrs->at(0).Size());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
    return shape_is_known(out_attrs->at(0));
  }
}

inline bool CumsumType(const nnvm::NodeAttrs& attrs,
                       std::vector<int> *in_attrs,
                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const CumsumParam &param = nnvm::get<CumsumParam>(attrs.parsed);

  if (param.dtype.has_value()) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype.value());
  } else {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
    TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  }

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(CumsumParam);

NNVM_REGISTER_OP(_np_cumsum)
.set_attr_parser(ParamParser<CumsumParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", CumsumShape)
.set_attr<nnvm::FInferType>("FInferType", CumsumType)
.set_attr<FCompute>("FCompute<cpu>", CumsumForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_np_cumsum"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("a", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(CumsumParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_np_cumsum)
.set_attr_parser(ParamParser<CumsumParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", CumsumBackward<cpu>);

}  // namespace op
}  // namespace mxnet
