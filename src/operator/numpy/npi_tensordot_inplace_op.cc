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
 * \file npi_tensordot_inplace_op.cc
 * \brief CPU Implementation of numpy-compatible tensordot
 */

#include "npi_tensordot_inplace_op-inl.h"

namespace mxnet {
namespace op {

inline bool TensordotInplaceOpShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);

  if (!ndim_is_known(a_shape) || !ndim_is_known(b_shape)) {
    return false;
  }

  const TensordotInplaceParam& param = nnvm::get<TensordotInplaceParam>(attrs.parsed);  
  const int& axes = param.axes;    

  Tuple<int> a_axes_summed;
  Tuple<int> b_axes_summed;  
  getSummedAxes(a_axes_summed, b_axes_summed, axes, a_shape);

  Tuple<int> a_axes_remained;
  Tuple<int> b_axes_remained;
  Tuple<int> a_axes;
  Tuple<int> b_axes;
  getReorderedAxes(a_axes_summed, a_axes_remained, a_axes, b_axes_summed, b_axes_remained, 
    b_axes, a_shape, b_shape);

  CHECK_EQ(a_axes_summed.ndim(), b_axes_summed.ndim());

  mxnet::TShape out_shape(a_axes_remained.ndim() + b_axes_remained.ndim(), -1);
  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    out_shape[i] = a_shape[a_axes_remained[i]];
  }
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    out_shape[a_axes_remained.ndim() + i] = b_shape[b_axes_remained[i]];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);

  mxnet::TShape tem_shape1(a_axes.ndim(), -1);
  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    tem_shape1[a_axes_remained[i]] = out_shape[i];
  }
  for (int i = 0; i < a_axes_summed.ndim(); i++) {
    tem_shape1[a_axes_summed[i]] = b_shape[b_axes_summed[i]];
  }
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, tem_shape1);

  mxnet::TShape tem_shape2(b_axes.ndim(), -1);
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    tem_shape2[b_axes_remained[i]] = out_shape[a_axes_remained.ndim() + i];
  }
  for (int i = 0; i < b_axes_summed.ndim(); i++) {
    tem_shape2[b_axes_summed[i]] = a_shape[a_axes_summed[i]];
  }
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, tem_shape2);

  return shape_is_known(*in_attrs) && shape_is_known(*out_attrs); 
}

DMLC_REGISTER_PARAMETER(TensordotInplaceParam);                                           

NNVM_REGISTER_OP(tensordot_inplace)
.add_alias("_npi_tensordot_inplace")                                                        
.describe(R"code(This operators implements the numpy-compatible inplace tensordot function                 
)code" ADD_FILELINE) // description. 
.set_attr_parser(mxnet::op::ParamParser<TensordotInplaceParam>)  
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", TensordotInplaceOpShape)
.set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", TensordotInplaceOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", mxnet::op::ElemwiseGradUseIn{"_backward_tensordot_inplace"})
.add_argument("a", "NDArray-or-Symbol", "First input")
.add_argument("b", "NDArray-or-Symbol", "Second input")
.add_arguments(TensordotInplaceParam::__FIELDS__()); // abc  

NNVM_REGISTER_OP(_backward_tensordot_inplace)
.set_attr_parser(mxnet::op::ParamParser<TensordotInplaceParam>) 
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", TensordotInplaceOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
