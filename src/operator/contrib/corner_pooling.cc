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
 * \file corner_pooling.cc
 * \brief corner pooling operator
 * \author Jiajie Tang 
*/

#include "./corner_pooling-inl.h"

#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

void CornerPoolingParamParser(nnvm::NodeAttrs *attrs) {
  CornerPoolingParam param;
  param.Init(attrs->dict);
  attrs->parsed = std::move(param);
}


static bool CornerPoolingType(const nnvm::NodeAttrs &attrs,
                              std::vector<int> *in_attrs,
                              std::vector<int> *out_attrs) {
  out_attrs->at(0) = in_attrs->at(0);
  return true;
}

static bool CornerPoolingShape(const nnvm::NodeAttrs &attrs,
                               std::vector<TShape> *in_shape,
                               std::vector<TShape> *out_shape) {
  CHECK_EQ(in_shape->size(), 1U);

  const TShape &dshape = (*in_shape)[0];

  CHECK_EQ(dshape.ndim(), 4U)
      << "CornerPooling: Input data should be  4D in (batch, channel, h, w)";
  TShape oshape = dshape;
  if (dshape.ndim() == 0) return false;
  out_shape->clear();
  out_shape->push_back(oshape);
  return true;
}




DMLC_REGISTER_PARAMETER(CornerPoolingParam);

NNVM_REGISTER_OP(CornerPooling)
.describe(R"code(Performs corner pooling over a 4D input with the shape of (NCHW).
    
Four corner pooling options are supported by ``corner_pooling_type``:

- **left**: left corner pooling
- **right**: right corner pooling
- **top**: top corner pooling
- **bottom**: bottom corner pooling

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs([](const NodeAttrs& attrs) {
  const CornerPoolingParam &param = nnvm::get<CornerPoolingParam>(attrs.parsed);
  return GetNumOutputs(param);
})
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  const CornerPoolingParam &param = nnvm::get<CornerPoolingParam>(attrs.parsed);
  if (GetNumOutputs(param) == 2)
    return std::vector<std::string>{"output", "workspace"};
  else
    return std::vector<std::string>{"output"};
})
.set_attr_parser(CornerPoolingParamParser)
.set_attr<nnvm::FInferType>("FInferType", CornerPoolingType)
.set_attr<nnvm::FInferShape>("FInferShape", CornerPoolingShape)
.set_attr<FCompute>("FCompute<cpu>", CornerPoolingCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
                           ElemwiseGradUseInOut{"_backward_CornerPooling"})
.add_argument("data", "NDArray-or-Symbol",
              "Input data to the corner pooling operator.")
.add_arguments(CornerPoolingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_CornerPooling)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>(
    "FInplaceOption",
    [](const NodeAttrs &attrs) {
  return std::vector<std::pair<int, int> >{{1, 0}};
})
.set_attr_parser(CornerPoolingParamParser)
.set_attr<FCompute>("FCompute<cpu>", CornerPoolingGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
