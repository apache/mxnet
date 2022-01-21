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
 * \file np_true_divide.cc
 * \brief CPU Implementation of true_divide operator.
 */

#include "./np_true_divide-inl.h"

namespace mxnet {
namespace op {

int TrueDivideOutType(int ltype, int rtype) {
  if (common::is_float(ltype) && common::is_float(rtype)) {
    // If both inputs are float, return the one with the higher precision
    return common::type_promotion(ltype, rtype);
  } else if (common::is_float(ltype) || common::is_float(rtype)) {
    // If only one of the inputs is float, return that float type
    return (common::is_float(ltype)) ? ltype : rtype;
  }
  // If neither of the inputs is float, return the default dtype
  return mxnet::common::GetDefaultDtype();
}

template <int num_inputs>
bool TrueDivideType(const nnvm::NodeAttrs& attrs,
                    std::vector<int>* in_attrs,
                    std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(num_inputs));
  CHECK_GT(in_attrs->size(), 0U);
  CHECK_EQ(out_attrs->size(), 1U);

  for (const int dtype : *in_attrs) {
    if (dtype == -1)
      return false;
  }

  const int lhs_dtype = in_attrs->at(0);
  const int rhs_dtype =
      (num_inputs == 2) ?
          in_attrs->at(1) :
          (common::is_float(lhs_dtype) ? lhs_dtype : mxnet::common::GetDefaultDtype());
  TYPE_ASSIGN_CHECK(*out_attrs, 0, TrueDivideOutType(lhs_dtype, rhs_dtype));
  return true;
}

#if MXNET_USE_ONEDNN == 1
void NumpyDivideBroadcastComputeCPU(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  TrueDivideBroadcastCompute<cpu>(attrs, ctx, inputs, req, outputs);
}
#endif  // MXNET_USE_ONEDNN

NNVM_REGISTER_OP(_npi_true_divide)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"lhs", "rhs"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", TrueDivideType<2>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>", TrueDivideBroadcastCompute<cpu>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FComputeEx>("FComputeEx<cpu>", NumpyBinaryOperatorComputeExCPU<op::mshadow_op::div>)
    .set_attr<FInferStorageType>("FInferStorageType", NumpyBinaryBroadcastStorageType)
#endif  // MXNET_USE_ONEDNN
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_broadcast_div"})
    .add_argument("lhs", "NDArray-or-Symbol", "Dividend array")
    .add_argument("rhs", "NDArray-or-Symbol", "Divisor array");

NNVM_REGISTER_OP(_backward_npi_broadcast_div)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>",
                        NumpyBinaryBackwardUseIn<cpu, mshadow_op::div_grad, mshadow_op::div_rgrad>);

NNVM_REGISTER_OP(_npi_true_divide_scalar)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", TrueDivideType<1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>", TrueDivideScalarCompute<cpu, op::mshadow_op::true_divide>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_div_scalar"})
    .add_argument("data", "NDArray-or-Symbol", "source input")
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_rtrue_divide_scalar)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", TrueDivideType<1>)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
#ifdef _WIN32
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
#endif
    .set_attr<FCompute>("FCompute<cpu>", TrueDivideScalarCompute<cpu, mshadow_op::rtrue_divide>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rdiv_scalar"})
    .add_argument("data", "NDArray-or-Symbol", "source input")
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
