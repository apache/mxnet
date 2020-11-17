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
 * \file np_pad_op.cc
 * \brief CPU Implementation of numpy pad operations
 */

#include <vector>
#include "./np_pad_op-inl.h"
#include "../nn/concat-inl.h"

namespace mxnet {
namespace op {

inline bool NumpyPadOpShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector* in_attrs,
                            mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }
  const NumpyPadParam& param = nnvm::get<NumpyPadParam>(attrs.parsed);

  mxnet::TShape oshape = NumpyPadShapeImpl(ishape, param.pad_width);

  if (shape_is_none(oshape)) {
    LOG(FATAL) << "Pad does not exist.";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyPadOpType(const nnvm::NodeAttrs &attrs,
                           std::vector<int> *in_attrs,
                           std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  return (*out_attrs)[0] != -1;
}

DMLC_REGISTER_PARAMETER(NumpyPadParam);

NNVM_REGISTER_OP(_npi_pad)
.set_attr_parser(ParamParser<NumpyPadParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", NumpyPadOpShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyPadOpType)
.set_attr<FCompute>("FCompute<cpu>", NumpyPadOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_npi_pad"})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyPadParam::__FIELDS__())
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  });

NNVM_REGISTER_OP(_backward_npi_pad)
.set_attr_parser(ParamParser<NumpyPadParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", NumpyPadOpBackward<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  });

}  // namespace op
}  // namespace mxnet
