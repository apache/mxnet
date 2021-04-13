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
 * \file np_diff.cc
 * \brief CPU implementation of numpy-compatible diff operator
 */

#include "./np_diff-inl.h"

namespace mxnet {
namespace op {

inline TShape NumpyDiffShapeImpl(const TShape& ishape,
                                 const int n,
                                 const int axis) {
  CHECK_GE(n, 0);
  int axis_checked = CheckAxis(axis, ishape.ndim());

  TShape oshape = ishape;
  if (n >= ishape[axis_checked]) {
    oshape[axis_checked] = 0;
  } else {
    oshape[axis_checked] -= n;
  }
  return oshape;
}

inline bool DiffShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape>* in_attrs,
                      std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const DiffParam& param = nnvm::get<DiffParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyDiffShapeImpl((*in_attrs)[0], param.n, param.axis));
  return shape_is_known(out_attrs->at(0));
}

inline bool DiffType(const nnvm::NodeAttrs& attrs,
                     std::vector<int>* in_attrs,
                     std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

DMLC_REGISTER_PARAMETER(DiffParam);

NNVM_REGISTER_OP(_npi_diff)
.set_attr_parser(ParamParser<DiffParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", DiffShape)
.set_attr<nnvm::FInferType>("FInferType", DiffType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", DiffForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
                            ElemwiseGradUseNone{"_backward_npi_diff"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("a", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(DiffParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_npi_diff)
.set_attr_parser(ParamParser<DiffParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", DiffBackward<cpu>);

}  // namespace op
}  // namespace mxnet
