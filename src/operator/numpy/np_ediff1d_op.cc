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
 * \file np_dediff1d_op.cc
 * \brief CPU implementation of numpy-compatible ediff1d operator
 */

#include "./np_ediff1d_op-inl.h"

namespace mxnet {
namespace op {

inline bool EDiff1DType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  CHECK_GE(in_attrs->size(), 1U);
  CHECK_LE(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));

  const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
  if (param.to_begin_arr_given && param.to_end_arr_given) {
      TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
      TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(2));
  } else if (param.to_begin_arr_given || param.to_end_arr_given) {
      TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  }

  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  return out_attrs->at(0) != -1 && in_attrs->at(0) != -1;
}

inline TShape NumpyEDiff1DShapeImpl(std::vector<TShape>* in_attrs,
                                    const bool to_begin_arr_given,
                                    const bool to_end_arr_given,
                                    dmlc::optional<double> to_begin_scalar,
                                    dmlc::optional<double> to_end_scalar) {
  size_t out = (in_attrs->at(0).Size() > 0)? in_attrs->at(0).Size() - 1: 0;
  // case 1: when both `to_begin` and `to_end` are arrays
  if (to_begin_arr_given && to_end_arr_given) {
      out += in_attrs->at(1).Size() + in_attrs->at(2).Size();
  // case 2: only one of the parameters is an array
  } else if (to_begin_arr_given || to_end_arr_given) {
      out += in_attrs->at(1).Size();
      // if the other one is a scalar
      if (to_begin_scalar.has_value() || to_end_scalar.has_value()) {
          out += 1;
      }
  // case 3: neither of the parameters is an array
  } else {
      // case 3.1: both of the parameters are scalars
      if (to_begin_scalar.has_value() && to_end_scalar.has_value()) {
          out += 2;
      // case 3.2: only one of the parameters is a scalar
      } else if (to_begin_scalar.has_value() || to_end_scalar.has_value()) {
          out += 1;
      }
      // case 3.3: they are both `None` -- skip
  }
  TShape oshape = TShape(1, out);
  return oshape;
}

inline bool EDiff1DShape(const nnvm::NodeAttrs& attrs,
                         std::vector<TShape>* in_attrs,
                         std::vector<TShape>* out_attrs) {
  CHECK_GE(in_attrs->size(), 1U);
  CHECK_LE(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyEDiff1DShapeImpl(in_attrs,
                                           param.to_begin_arr_given,
                                           param.to_end_arr_given,
                                           param.to_begin_scalar,
                                           param.to_end_scalar));
  return shape_is_known(out_attrs->at(0));
}

DMLC_REGISTER_PARAMETER(EDiff1DParam);

NNVM_REGISTER_OP(_npi_ediff1d)
.set_attr_parser(ParamParser<EDiff1DParam>)
.set_num_inputs(
  [](const nnvm::NodeAttrs& attrs) {
     const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
     int num_inputs = 1;
     if (param.to_begin_arr_given) num_inputs += 1;
     if (param.to_end_arr_given) num_inputs += 1;
     return num_inputs;
  })
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
    int num_inputs = 1;
    if (param.to_begin_arr_given) num_inputs += 1;
    if (param.to_end_arr_given) num_inputs += 1;
    if (num_inputs == 1) return std::vector<std::string>{"input1"};
    if (num_inputs == 2) return std::vector<std::string>{"input1", "input2"};
    return std::vector<std::string>{"input1", "input2", "input3"};
  })
.set_attr<mxnet::FInferShape>("FInferShape",  EDiff1DShape)
.set_attr<nnvm::FInferType>("FInferType", EDiff1DType)
.set_attr<FCompute>("FCompute<cpu>", EDiff1DForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_npi_backward_ediff1d"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("input1", "NDArray-or-Symbol", "Source input")
.add_argument("input2", "NDArray-or-Symbol", "Source input")
.add_argument("input3", "NDArray-or-Symbol", "Source input")
.add_arguments(EDiff1DParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_backward_ediff1d)
.set_attr_parser(ParamParser<EDiff1DParam>)
.set_num_inputs(
  [](const nnvm::NodeAttrs& attrs) {
     const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
     int num_inputs = 2;
     if (param.to_begin_arr_given) num_inputs += 1;
     if (param.to_end_arr_given) num_inputs += 1;
     return num_inputs;
  })
.set_num_outputs(
  [](const nnvm::NodeAttrs& attrs) {
     const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
     int num_inputs = 1;
     if (param.to_begin_arr_given) num_inputs += 1;
     if (param.to_end_arr_given) num_inputs += 1;
     return num_inputs;
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<mxnet::FCompute>("FCompute<cpu>", EDiff1DBackward<cpu>);

}  // namespace op
}  // namespace mxnet
